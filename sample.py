import os, math, json, glob, argparse
from typing import List, Dict, Any, Optional, Tuple
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import argparse, sys, datetime, yaml
from util import instantiate_from_config
from dataset import FloodWindowDataset  
from ddim import DDIMSampler  


def load_dataset_from_yaml(yaml_path: str) -> Tuple[FloodWindowDataset, dict]:
    cfg = OmegaConf.load(yaml_path)
    data_cfg = cfg.data.params.data  
    dataset = instantiate_from_config(data_cfg)
    if not isinstance(dataset, FloodWindowDataset):
        raise TypeError("instantiate_from_config(data_cfg) not FloodWindowDataset ")
    return dataset, data_cfg

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

def inverse_runoff(z, mean, std):
    
    import torch
    import numpy as np
    if isinstance(z, np.ndarray):
        z = torch.from_numpy(z)
    elif not torch.is_tensor(z):
        raise TypeError(f"Unsupported input type: {type(z)}")
    if not torch.is_tensor(mean):
        mean = torch.tensor(mean, dtype=z.dtype, device=z.device)
    if not torch.is_tensor(std):
        std = torch.tensor(std, dtype=z.dtype, device=z.device)
    log1p_q = z * (std + 1e-8) + mean
    q = torch.expm1(log1p_q)
    #q = log1p_q
    return q.detach().cpu().numpy().astype(np.float32)

def read_stations_arg(stations: Optional[str], stations_list: Optional[str], dataset: FloodWindowDataset) -> List[str]:
    if stations_list:
        with open(stations_list, "r") as f:
            sts = [line.strip() for line in f if line.strip()]
        return sts
    if stations:
        return [s.strip() for s in stations.split(",") if s.strip()]
   
    sts = sorted({st for (st, _, _) in dataset.index})
    return sts

def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd.get("global_step", None)
        state_dict = pl_sd["state_dict"]
    else:
        state_dict = None
        global_step = None
    model = load_model_from_config(config.model, state_dict)
    return model, global_step

def _to_tensor(x, device):
    import torch, numpy as np
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    if torch.is_tensor(x):
        return x.to(device)
    raise TypeError(f"Unsupported type: {type(x)}")

# ------------------ sample  one step ------------------
@torch.no_grad()
def convsample_ddim(model, steps, shape, cond, eta, freemodel,
                    only_nonelabel=False, device=None,
                    guidance_scale=1.5, temperature=1.0, verbose=False):
  
    model.to(device).eval()
    
    cond = [c.to(device) if torch.is_tensor(c) else c for c in cond]

    
    FUTURE_IDX = 1
    uncond = []
    for i, c in enumerate(cond):
        if i == FUTURE_IDX and torch.is_tensor(c) and c is not None:
            uncond.append(torch.zeros_like(c))
        else:
            uncond.append(c if not torch.is_tensor(c) else c.clone())

    if only_nonelabel:
        if torch.is_tensor(cond[FUTURE_IDX]) and cond[FUTURE_IDX] is not None:
            cond[FUTURE_IDX] = torch.zeros_like(cond[FUTURE_IDX])

    sampler = DDIMSampler(model, freemodel)
    samples, _ = sampler.sample(
        S=steps,
        batch_size=shape[0],
        shape=shape[1:],
        conditioning=cond,
        freemodel=freemodel,
        eta=eta,
        verbose=verbose,
        temperature=temperature,
        unconditional_guidance_scale=guidance_scale,
        unconditional_conditioning=None,   
        
    )
    return samples


def worker(rank: int, world_args: Dict[str, Any]):
   
    os.environ["CUDA_VISIBLE_DEVICES"] = world_args["cuda_visible"]
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    
    dataset_yaml  = world_args["dataset_yaml"]
    model_ckpt    = world_args["model_ckpt"]
    steps         = world_args["steps"]
    eta           = world_args["eta"]
    num_samples   = world_args["num_samples"]   
    batch_size    = world_args["batch_size"]   
    out_dir       = world_args["out_dir"]
    stations_sub  = world_args["stations_split"][rank]
    start         = world_args["start"]
    end           = world_args["end"]
    cfg_scale     = world_args["cfg_scale"]

    
    dataset, data_cfg = load_dataset_from_yaml(dataset_yaml)
    T_out = dataset.future_len
    q_scaler_path = data_cfg["params"]["scaler_q_json"] if "params" in data_cfg else data_cfg["scaler_q_json"]
    q_mean_dict, q_std_dict = load_scaler(q_scaler_path)
    q_mean = q_mean_dict["runoff"]
    q_std = q_std_dict["runoff"]
   
    
    config_model = OmegaConf.load(dataset_yaml)
    model, _ = load_model(config_model, model_ckpt, True, True)
    
    freemodel = None
    os.makedirs(out_dir, exist_ok=True)

    
    
    for st in stations_sub:
        pack = dataset.get_station_windows(st, start=start, end=end,
                                           return_inverse_runoff=False, return_dates=True)
        t0_list = pack["t0"]
        #print('dates_out',pack['dates_out'])
        if len(t0_list) == 0:
            print(f"[GPU{rank}] {st}: no valid windows in [{start}, {end}] -> skip")
            continue
        
        target_z = pack["runoff"]                     # [Nw, T_out, 1] (torch or np)
        target_mm = inverse_runoff(target_z, q_mean, q_std)  # [Nw, T_out, 1] (np)
        print(target_mm[0])
        Nw = target_mm.shape[0]
        
        S = int(num_samples)
        Bmax = int(batch_size)
        
        
        samples_chunks = []
        targets_chunks = []

        for i in range(0, Nw, Bmax):
            j = min(i + Bmax, Nw)
            B = j - i

            
            past_fast      = _to_tensor(pack["past_fast"][i:j], device)        # [B,Lp,Ff]
            future_weather = _to_tensor(pack["future_weather"][i:j], device)   # [B,T_out,Fut]
           
            geo            = _to_tensor(pack["geo"][i:j], device)              # [B,G]
            
            past_fast_rep      = torch.tile(past_fast, (S,1,1))
            future_weather_rep = torch.tile(future_weather, (S,1,1))
            
            geo_rep            = torch.tile(geo, (S,1))
            
            cond_rep = [past_fast_rep, future_weather_rep, geo_rep]

            # sample once
            samples_z_flat = convsample_ddim(
                    model=model, steps=steps, shape=[S*B, T_out,1], cond=cond_rep,
                    eta=eta, freemodel=freemodel, only_nonelabel=False, device=device,
                    guidance_scale=cfg_scale, temperature=1.1,
                )

            
            samples_z = samples_z_flat.view(S, B, T_out, 1)
          
            samples_mm = inverse_runoff(samples_z, q_mean, q_std)     # [S,B,T,1] (np)

            samples_chunks.append(samples_mm)
            targets_chunks.append(target_mm[i:j])                     # [B,T,1]   (np)

            
        samples_all = np.concatenate(samples_chunks, axis=1)   # [S, Nw, T, 1]
        targets_all = np.concatenate(targets_chunks, axis=0)   # [Nw, T, 1]

        
        
        save_path = os.path.join(out_dir, f"{st}_{cfg_scale}_{start}.npz")
        np.savez_compressed(
            save_path,
            sample=samples_all,                     # [S, Nw, T_out, 1]
            target=targets_all,                     # [Nw, T_out, 1]
            t0_dates=pack["dates_t0"].astype("datetime64[D]"),
            out_dates=pack["dates_out"].astype("datetime64[D]")
        )
        print(f"[GPU{rank}] ✓ saved {save_path}  (S={samples_all.shape[0]}, Nw={targets_all.shape[0]})")
                        

def parse_args():
    ap = argparse.ArgumentParser()                                                     
    ap.add_argument("--dataset_config", type=str, help="sample.yaml",             
                    default="sample.yaml")  # same as traing.yaml  start_date :'2009-01-01'
    ap.add_argument("--model_ckpt",     type=str, 
                    default="model.ckpt") # the trained model parameters
    ap.add_argument("--steps",          type=int,   default=100)
    ap.add_argument("--eta",            type=float, default=0.5)
    ap.add_argument("--num-samples",    type=int,   default=50)
    ap.add_argument("--batch-size",     type=int,   default=50)
    ap.add_argument("--cfg-scale",      type=float, default=1.0)
    ap.add_argument("--ngpus",          type=int,   default=8, help=" CUDA_VISIBLE_DEVICES or torch.cuda.device_count()")
    ap.add_argument("--start",          type=str,   default="2009-01-01")
    ap.add_argument("--end",            type=str,   default="2021-12-31")
    ap.add_argument("--stations",       type=str,   default=None, help="")
    ap.add_argument("--stations-list",  type=str,   default=None, help="txt")
    ap.add_argument("--out-dir",        type=str,  
                    default="")
    return ap.parse_args()


def list_done_stations(out_dir: str):
    
    done = set()

   
    for p in glob.glob(os.path.join(out_dir, "GRDC_*.npz")):
        st = os.path.splitext(os.path.basename(p))[0]  # GRDC_xxxxxx
        done.add(st)

    
    for p in glob.glob(os.path.join(out_dir, "GRDC_*", "GRDC_*.npz")):
        st = os.path.splitext(os.path.basename(p))[0]
        done.add(st)

    return done




def main():
    args = parse_args()

    stations = []
    

    
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible is not None and visible.strip():
        dev_list = [d.strip() for d in visible.split(",") if d.strip()]
        ngpus_env = len(dev_list)
        cuda_visible = ",".join(dev_list)
    else:
        ngpus_env = torch.cuda.device_count()
        cuda_visible = ",".join(str(i) for i in range(ngpus_env))

    ngpus = args.ngpus or ngpus_env

    assert ngpus >= 1, "No GPU detected."


    splits = [stations[i::ngpus] for i in range(ngpus)]
    fg_list = [1.0]
    for w in fg_list:
        world_args = dict(
            cuda_visible=cuda_visible,
            dataset_yaml=args.dataset_config,
            model_ckpt=args.model_ckpt,
            steps=args.steps,
            eta=args.eta,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            out_dir=args.out_dir,
            stations_split=splits,
            start=args.start,
            end=args.end,
            cfg_scale=w,
        )
        print(f"Total stations={len(stations)}, ngpus={ngpus}")
        for gi in range(ngpus):
            print(f"  GPU{gi}: {len(splits[gi])} stations")

        mp.spawn(worker, args=(world_args,), nprocs=ngpus, join=True)

if __name__ == "__main__":
    main()
