
"""
caculate the idx over threshold in different lead time 
- input : sample [S, N, 7, 1], target [N, 7, 1]
- threshold file thresholds_all.csv (T2_emp / T2_gev)

- out put CSV: predicted_exceed_indices.csv and observed_exceed_indices.csv
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True,
                    help="npz file, such as :/home/huangbh/test_sample")
    ap.add_argument("--thresholds_csv", type=str, required=True,
                    help="threeshold table, /home/huangbh/GRDC_train_v7/return_levels/thresholds_all.csv")
    ap.add_argument("--return_col", type=str, default="T2_emp",
                    help="return year T2_emp / T2_gev")
    ap.add_argument("--tau", type=float, default=0.5,
                    help="threshold of ensemble ")
    ap.add_argument("--leads", type=str, default="1,2,3,4,5,6,7",
                    help="evaluated lead table")
    ap.add_argument("--out_dir", type=str, default="/home/huangbh/eval_results/summaries",
                    help="output")
    ap.add_argument("--cfg", type=float, default=1.0,
                    help="cfg")
    return ap.parse_args()

def load_threshold_map(thr_csv: str, return_col: str) -> dict:
    df = pd.read_csv(thr_csv)
    
    cols = {c.lower(): c for c in df.columns}
    st_col = cols.get("station", "station")
    if return_col not in df.columns:
        raise ValueError(f" {return_col}  {thr_csv} :{df.columns.tolist()}")
    
    m = {}
    for _, r in df.iterrows():
        st = str(r[st_col])
        if not st.startswith("GRDC_"):
            st = f"GRDC_{st}"
        m[st] = float(r[return_col])
    return m

def build_date_to_index_map(dates_array: np.ndarray) -> dict:
    
    date_map = {}
    for i in range(len(dates_array)):
        
        anchor_date_np = dates_array[i, 0]
       
        anchor_date_str = pd.to_datetime(str(anchor_date_np)).strftime('%Y-%m-%d')
        date_map[anchor_date_str] = i
    return date_map

def to_json_list(idx_arr: np.ndarray) -> str:
    
    return json.dumps(idx_arr.astype(int).tolist(), ensure_ascii=False)



def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    thr_map = load_threshold_map(args.thresholds_csv, args.return_col)
    thr_map_model = load_threshold_map('/home/huangbh/eval_results/summaries/model_empirical_thresholds_lead1_21statics_ALD.csv', 'T5_model_emp')
    
    
    leads = [int(x) for x in args.leads.split(",") if x.strip()]
    for L in leads:
        if L < 1 or L > 7:
            raise ValueError(f"lead must in 1..7, {L}")

    npz_files = sorted(glob.glob(os.path.join(args.data_root, "GRDC_*.npz")))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"{args.data_root} not find GRDC_*.npz")

    rows_pred, rows_obs = [], []

    for path in npz_files:
        st_name = os.path.splitext(os.path.basename(path))[0]   # e.g., GRDC_4123082
        if st_name not in thr_map:
            print(f"[skip] {st_name}: no stataion({args.thresholds_csv})")
            continue
        thr = thr_map[st_name]
        
        
        thr_model = thr_map_model.get(st_name, thr)
        

        try:
            data = np.load(path)
            sample = data["mu"]   # [S, N, 7, 1]
            target = data["target"]   # [N, 7, 1]
            
        except Exception as e:
            print(f"[skip] {st_name}: read faliure - {e}")
            continue
      
       
        q_pred = sample
        
        N = q_pred.shape[0]
        if target.shape[0] != N:
            
            N = min(N, target.shape[0])
            q_pred = q_pred[:N, :,0]
            target = target[:N, :, 0]

        
        for L in leads:
            lidx = L - 1
            pred_series = q_pred[:, lidx]           # [N]
            obs_series  = target[:, lidx]        # [N]

           
            valid_pred = np.isfinite(pred_series)
            valid_obs  = np.isfinite(obs_series)

            obs_condition = valid_obs & (obs_series >= thr) #& (obs_series < thr_5)
           
            pred_condition = valid_pred & (pred_series >= thr_model) #& (pred_series < thr_5)
            
            obs_idx  = np.where(obs_condition)[0]
            pred_idx = np.where(pred_condition)[0]

            rows_pred.append({
                "station": st_name,
                "return_col": args.return_col,
                "tau": args.tau,
                "lead": L,
                "thr_mm": thr,
                "n_windows": int(N),
                "n_exceed_pred": int(pred_idx.size),
                "exceed_indices_json": to_json_list(pred_idx),
            })
            rows_obs.append({
                "station": st_name,
                "return_col": args.return_col,
                "lead": L,
                "thr_mm": thr,
                "n_windows": int(N),
                "n_exceed_obs": int(obs_idx.size),
                "exceed_indices_json": to_json_list(obs_idx),
            })

    df_pred = pd.DataFrame(rows_pred).sort_values(["station", "lead"]).reset_index(drop=True)
    df_obs  = pd.DataFrame(rows_obs ).sort_values(["station", "lead"]).reset_index(drop=True)

    out_pred = os.path.join(args.out_dir, f"predicted_exceed_indices_tau_{args.cfg}_{args.return_col}_ALD.csv")
    out_obs  = os.path.join(args.out_dir, f"observed_exceed_indices_{args.cfg}_{args.return_col}_ALD.csv")
    df_pred.to_csv(out_pred, index=False)
    df_obs.to_csv(out_obs, index=False)

   

if __name__ == "__main__":
    main()

