# -*- coding: utf-8 -*-
import os, glob, json
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _parse_date_like(x):
    import pandas as pd
    if x is None:
        return None
    if isinstance(x, int):
        return pd.Timestamp(f"{x}-01-01")
    x = str(x)
    if len(x) == 4 and x.isdigit():
        return pd.Timestamp(f"{x}-01-01")
    return pd.to_datetime(x)


def load_scaler(path: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    with open(path, "r") as f:
        obj = json.load(f)

   
    if all(isinstance(v, dict) and "mean" in v for v in obj.values()):
        mean = {k: v["mean"] for k, v in obj.items()}
        std  = {k: v["std"]  for k, v in obj.items()}
        return mean, std

    
    elif "mean" in obj and "std" in obj:
        return obj["mean"], obj["std"]

    else:
        raise ValueError(f"Unsupported scaler format in {path}")


class FloodWindowDataset(Dataset):
    """
   
      - past_fast:      [Lp, F_fast]      (P, Tmean, SWd, PET ...)
      - future_weather: [To, F_future]    (P, Tmean)
     
      - geo:            [G]               
      - runoff:         [To, 1]           
     self.index: [(station, t0, t0_date)]
    """

    def __init__(
        self,
        root_dir: str = "",
        static_csv: str = "",
        scaler_weather_json: str = "",
        scaler_q_json: str = "",
        scaler_static_json: str = "",
        fast_cols: List[str] = ("P","Tmean","SWd","LWd","RH","PET"),
        future_cols: List[str] = ("P","Tmean"),
        
        geo_cols: Optional[List[str]] = None,  
        runoff_col: str = "runoff_mm",
        past_len: int = 365,
        future_len: int = 7,
        stride: int = 3,
        start_date: str = "1984-01-01",
        end_date:   str = "2006-12-31",
        stats_csv_path: str = "geo_statics",
        verbose: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.static_csv = static_csv
        self.fast_cols = list(fast_cols)
        self.future_cols = list(future_cols)
        
        self.runoff_col = runoff_col
        self.past_len = past_len
        self.future_len = future_len
        self.stride = stride
        self.start_date = start_date
        self.end_date = end_date
        self.verbose = verbose
        self.stats_csv_path = stats_csv_path

       
        self.w_mean, self.w_std,  = load_scaler(scaler_weather_json)
        q_mean_dict, q_std_dict = load_scaler(scaler_q_json)
        self.s_mean, self.s_std   = load_scaler(scaler_static_json)
        self.q_mean = q_mean_dict["runoff"]
        self.q_std = q_std_dict["runoff"]
       
        df_static = pd.read_csv(static_csv)
        if "stations" not in df_static.columns:
            raise ValueError("must includee 'stations' column")
        if geo_cols is None:
            geo_cols = [c for c in df_static.columns if c != "stations"]
        self.geo_cols = list(geo_cols)

       
        self.geo_log1p_cols = {"area", "pcp_intensity", "re_mean_annual_runoff_mm"}

        
        self.geo_dict: Dict[str, np.ndarray] = {}
        for _, row in df_static.iterrows():
            st = str(row["stations"])
            vec = []
            for c in self.geo_cols:
                val = row[c]
                if c in self.geo_log1p_cols:
                    val = np.log1p(val) if pd.notna(val) else np.nan
                m = self.s_mean.get(c, 0.0)
                s = self.s_std.get(c, 1.0)
                if pd.isna(val):
                    vec.append(np.nan)
                else:
                    vec.append((float(val) - m) / (s + 1e-8))
            self.geo_dict[st] = np.array(vec, dtype=np.float32)

 
        csv_files = sorted(glob.glob(os.path.join(root_dir, "GRDC_*.csv")))
        if len(csv_files) == 0:
            raise FileNotFoundError(f"no {root_dir} in GRDC_*.csv")

     
        self.station_to_file: Dict[str, str] = {}
        for f in csv_files:
            st = os.path.splitext(os.path.basename(f))[0]  # "GRDC_XXXX"
            self.station_to_file[st] = f

        
        self.series: Dict[str, Dict[str, np.ndarray]] = {}
        
        self.index: List[Tuple[str, int, pd.Timestamp]] = []
       
        stats_rows = []

        required_cols = set([self.runoff_col, "date"]) \
                        | set(self.fast_cols) | set(self.future_cols) | set(self.slow_cols)

        sdt = _parse_date_like(self.start_date)
        edt = _parse_date_like(self.end_date)

        for f in csv_files:
            st = os.path.splitext(os.path.basename(f))[0]  # "GRDC_XXXX"
           
            df = pd.read_csv(f)
            print(st)
            if "date" not in df.columns:
                if self.verbose:
                    print(f"[SKIP] {st} ")
                continue
            missing = [c for c in required_cols if c not in df.columns]
            if len(missing) > 0:
                if self.verbose:
                    print(f"[SKIP] {st} : {missing}")
                continue

            dt = pd.to_datetime(df["date"])
            mask = (dt >= sdt) & (dt <= edt)
            if mask.sum() < (self.past_len + self.future_len + 1):
               
                continue
            df = df.loc[mask].reset_index(drop=True)
            dt = dt.loc[mask].reset_index(drop=True)

           
            fast_arr_cols = []
            for c in self.fast_cols:
                arr = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float32)
                if c == "P":
                    arr = np.log1p(np.clip(arr, a_min=0.0, a_max=None))  
                fast_arr_cols.append(self._z_weather(arr, c))
            fast_arr = np.stack(fast_arr_cols, axis=1) 

           
            future_arr_cols = []
            for c in self.future_cols:
                arr = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float32)
                if c == "P":
                    arr = np.log1p(np.clip(arr, a_min=0.0, a_max=None))  
                future_arr_cols.append(self._z_weather(arr, c))
            future_arr = np.stack(future_arr_cols, axis=1)  # [N, F_future]

            
           
            q_raw = df[self.runoff_col].to_numpy(dtype=np.float32)
            q_log = np.log1p(np.clip(q_raw, a_min=0.0, a_max=None))
            
            q_std_arr = (q_log - self.q_mean) / (self.q_std + 1e-8)
            
            
            if st not in self.geo_dict:
               
                st2 = st.replace("GRDC_", "GRDC__")
                if st2 in self.geo_dict:
                    st_key = st2
                else:
                    if self.verbose:
                        print(f"[SKIP] {st} no ({st}/{st2})")
                    continue
            else:
                st_key = st
            geo_vec = self.geo_dict[st_key]
            
            if np.isnan(geo_vec).any():
                if self.verbose:
                    print(f"[SKIP] {st} clear nan")
                continue

            
            N = len(df)
            start_t0 = self.past_len - 1
            end_t0   = N - 1 - self.future_len
            if end_t0 < start_t0:
                continue
            t0_all = np.arange(start_t0, end_t0 + 1)

            valid_q = np.isfinite(q_std_arr).astype(np.int8)
            prefix_q = np.zeros(N + 1, dtype=np.int32)
            prefix_q[1:] = np.cumsum(valid_q)
            fsum = prefix_q[t0_all + self.future_len + 1] - prefix_q[t0_all + 1]
            runoff_ok = (fsum == self.future_len)

            t0_candidates = t0_all[runoff_ok]
            
            if self.stride > 1 and len(t0_candidates) > 0:
                t0_candidates = t0_candidates[::self.stride]

       
            self.series[st] = dict(
                fast=fast_arr.astype(np.float32),
                future=future_arr.astype(np.float32),
                slow=slow_arr.astype(np.float32),
                q=q_std_arr.astype(np.float32),
                dates=dt.values
            )
          
            for t0 in t0_candidates:
                t0_date = pd.Timestamp(dt.iloc[int(t0)])
                self.index.append((st, int(t0), t0_date))

            
            total_days = len(df)
            miss_rate_any = 1.0 - (np.isfinite(fast_arr).all(axis=1) &
                                    np.isfinite(future_arr).all(axis=1) &
                                    np.isfinite(slow_arr).all(axis=1) &
                                    np.isfinite(q_std_arr)).mean()
            stats_rows.append({
                "station": st,
                "total_days_in_range": total_days,
                "valid_windows": int(len(t0_candidates)),
                "miss_rate_any_col": float(miss_rate_any)
            })

       
        self.stats = pd.DataFrame(stats_rows).sort_values("valid_windows", ascending=False).reset_index(drop=True)
        os.makedirs(os.path.dirname(self.stats_csv_path), exist_ok=True)
        self.stats.to_csv(self.stats_csv_path, index=False)
        if self.verbose:
            print(f"✅ stats saved to: {self.stats_csv_path}")
            print(f"✅ total samples (windows): {len(self.index)} from {len(self.series)} stations")

    # ---------- Z-score (weather) ----------
    def _z_weather(self, arr: np.ndarray, col: str) -> np.ndarray:
        m = self.w_mean.get(col, 0.0)
        s = self.w_std.get(col, 1.0)
        arr = arr.astype(np.float32)
        return (arr - m) / (s + 1e-8)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        st, t0, _ = self.index[i]
        S = self.series[st]
        Lp, To = self.past_len, self.future_len

        past_fast = S["fast"][t0 - Lp + 1 : t0 + 1, :]               # [Lp, F_fast]
        future    = S["future"][t0 + 1 : t0 + To + 1, :]             # [To, F_future]
        
        runoff    = S["q"][t0 + 1 : t0 + To + 1]                     # [To]

        # geo
        st_key = st
        if st_key not in self.geo_dict:
            st2 = st.replace("GRDC_", "GRDC__")
            st_key = st2 if st2 in self.geo_dict else st
        geo_vec = self.geo_dict[st_key]

        return {
            "past_fast":      torch.from_numpy(past_fast),                  # [Lp, F_fast]
            "future_weather": torch.from_numpy(future),                     # [To, F_future]
            
            "geo":            torch.from_numpy(geo_vec.astype(np.float32)), # [G]
            "runoff":         torch.from_numpy(runoff.reshape(-1, 1))       # [To, 1]
        }

    
    def get_station_data(self, station: str, start=None, end=None):
        
        sdt = _parse_date_like(start) if start is not None else None
        edt = _parse_date_like(end) if end is not None else None

        rows = [(st, t0, d) for (st, t0, d) in self.index if st == station]
        if sdt is not None:
            rows = [(st, t0, d) for (st, t0, d) in rows if d >= sdt]
        if edt is not None:
            rows = [(st, t0, d) for (st, t0, d) in rows if d <= edt]
        if len(rows) == 0:
            return {"t0": np.array([], dtype=int), "dates": np.array([], dtype="datetime64[ns]")}
        rows.sort(key=lambda x: x[2])
        t0_list = np.array([t0 for (_, t0, _) in rows], dtype=int)
        dates   = np.array([d for (_, _, d) in rows], dtype="datetime64[ns]")

        return {"t0": t0_list, "dates": dates}
    

    def get_station_windows(
        self,
        station: str,
        start=None,
        end=None,
        return_inverse_runoff: bool = False,
        return_dates: bool = True,
    ):
        """
       

        return dict:
        - past_fast:      [N, Lp, F_fast]
        - future_weather: [N, To, F_future]
        - past_slow_vec:  [N, F_slow]
        - geo:            [N, G]
        - runoff:         [N, To, 1]  
        - t0:             [N]          
        - dates_t0:       [N] datetime64[ns] 
        - dates_out:      [N, To] 
        """
        
        sel = self.get_station_data(station, start=start, end=end)
        t0_list = sel["t0"]
        dates_t0 = sel["dates"]
        if len(t0_list) == 0:
         
            out = {
                "past_fast":      np.zeros((0, self.past_len, len(self.fast_cols)), dtype=np.float32),
                "future_weather": np.zeros((0, self.future_len, len(self.future_cols)), dtype=np.float32),
                "past_slow_vec":  np.zeros((0, len(self.slow_cols)), dtype=np.float32),
                "geo":            np.zeros((0, len(self.geo_cols)), dtype=np.float32),
                "runoff":         np.zeros((0, self.future_len, 1), dtype=np.float32),
                "t0":             t0_list,
            }
            if return_dates:
                out["dates_t0"]  = dates_t0
                out["dates_out"] = np.zeros((0, self.future_len), dtype="datetime64[ns]")
            return out

        if station not in self.series:
            
            st_key = station
            if st_key not in self.series:
                raise KeyError(f"{station} ")
        else:
            st_key = station

        S  = self.series[st_key]
        Lp = self.past_len
        To = self.future_len

 
        past_fast = np.stack([S["fast"][t - Lp + 1 : t + 1, :]          for t in t0_list], axis=0)  # [N,Lp,F_fast]
        future    = np.stack([S["future"][t + 1 : t + To + 1, :]         for t in t0_list], axis=0)  # [N,To,F_future]
        slow_vec  = np.stack([S["slow"][t, :]                            for t in t0_list], axis=0)  # [N,F_slow]
        runoff_z  = np.stack([S["q"][t + 1 : t + To + 1]                 for t in t0_list], axis=0)  # [N,To]
        runoff_z  = runoff_z[..., None].astype(np.float32)  # [N,To,1]

        if st_key in self.geo_dict:
            geo_vec = self.geo_dict[st_key].astype(np.float32)
        else:
            
            st2 = st_key.replace("GRDC_", "GRDC__")
            geo_vec = self.geo_dict[st2].astype(np.float32)
        geo = np.repeat(geo_vec[None, :], repeats=len(t0_list), axis=0)   

     
        runoff = runoff_z
        if return_inverse_runoff:
            
            
            log1p_q = runoff_z * (self.q_std + 1e-8) + self.q_mean
            runoff  = np.expm1(log1p_q).astype(np.float32)  # mm/day

        out = {
            "past_fast":      past_fast.astype(np.float32),
            "future_weather": future.astype(np.float32),
            "past_slow_vec":  slow_vec.astype(np.float32),
            "geo":            geo.astype(np.float32),
            "runoff":         runoff.astype(np.float32),
            "t0":             t0_list
        }

        if return_dates:
            
            out["dates_t0"] = dates_t0
            
            dates_out = np.stack([S["dates"][t + 1 : t + To + 1] for t in t0_list], axis=0)
            out["dates_out"] = dates_out

        return out


