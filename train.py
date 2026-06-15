# this diffusion training frame comes from the openAI github  https://github.com/openai/improved-diffusion, Multi-GPU training  based  on  pytorch_lightning
import sys
import os
import argparse, datetime, glob

import torch
from torch.utils.data import random_split, DataLoader, Dataset
from util import add_dict_to_argparser, instantiate_from_config
import numpy as np
from omegaconf import OmegaConf
import pytorch_lightning as pl
from torch.utils.data import Subset
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor, EarlyStopping
from pytorch_lightning.utilities import rank_zero_only
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from typing import Optional, Callable
#torch.set_float32_matmul_precision('medium')
#torch.set_float32_matmul_precision('high')

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, data=None, num_workers=None,valid_indices_path=None):
        """
        LightningDataModule for MeteorologicalRunoffDataset.

        Args:
            batch_size: Batch size for DataLoader.
            data: Configuration for the MeteorologicalRunoffDataset.
            num_workers: Number of workers for DataLoader.
        """
        super().__init__()
        self.batch_size = batch_size
        self.data = data  
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        
    def setup(self, stage: Optional[str]=None):
        
        self.train_dataset = instantiate_from_config(self.data)
    

    def train_dataloader(self):
        """
        Create DataLoader for training.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        """
        if auto-valiidation # we use manual validation  
        Create DataLoader for validation.
        """
        return []   # sample for val
    
    def test_dataloader(self):
        
        return []


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            
            #print(self.config.pretty())
            
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            #print(self.lightning_config.pretty())
            
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass



def main():
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    opt = create_argparser().parse_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    ckpt = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths)-paths[::-1].index("logs")+1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs+opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs")+1]
    else:
        if opt.name:
            name = "_"+opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_"+cfg_name
        else:
            name = ""
        nowname = now+name+opt.postfix
        logdir = os.path.join("logs", nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    configs = OmegaConf.merge(*configs)
    lightning_config = configs.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config["strategy"] = "auto"
    trainer_config["accelerator"] = "gpu"
    trainer_config["devices"] = [0,1,2,3,4,5,6,7]
    trainer_config['precision'] = '32' 
    trainer_config['max_epochs'] = 200
    trainer_config['log_every_n_steps'] = 50
    if trainer_config["accelerator"] == 'cpu':
        print(f"Running on cpu")
        cpu = True
    else:
        gpuinfo = len(trainer_config["devices"])
        print(f"Running on GPUs {gpuinfo}")
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    model = instantiate_from_config(configs.model)

    # trainer and callbacks
    trainer_kwargs = dict()
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "epoch={epoch:02d}-step={step:06d}",
            "verbose": True,
            "save_last": True,
            "save_on_train_epoch_end": True,
        }
    }
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 3

    modelckpt_cfg = lightning_config.get("modelcheckpoint", OmegaConf.create())
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    trainer_kwargs["callbacks"] = [instantiate_from_config(modelckpt_cfg)]
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "scripts.Runoff_train_ddpm.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": configs,
                "lightning_config": lightning_config,
            }
        },
        "image_logger": {
            "target": "scripts.Runoff_train_ddpm.ImageLogger",
            "params": {
                "batch_frequency": 1000,
                "max_images": 300,
                "clamp": False,
                "save_path": None,
                "scale": True,
            }
        },
        "learning_rate_logger": {
            "target": "scripts.Runoff_train_ddpm.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                #"log_momentum": True
            }
        },
        
    }
    """
    'NaN_callback':{
            "target":'scripts.Runoff_train_ddpm.EarlyStopping',
            "params":{
                "monitor":'val/loss_simple',
                "mode":"min",
                "check_finite": True,
            }
        }
    """
    callbacks_cfg = lightning_config.get('callbacks', OmegaConf.create())
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    trainer_kwargs["callbacks"].extend([instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg])
    trainer_kwargs = {**vars(trainer_opt), **trainer_kwargs}
    trainer = Trainer(**trainer_kwargs)
    ##trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    data = instantiate_from_config(configs.data)
    data.setup()
    if not cpu:
        ngpu = gpuinfo
    else:
        ngpu=1
    bs, base_lr = configs.data.params.batch_size, configs.model.base_learning_rate
    accumulate_grad_batches = opt.accumulate_grad_batches or 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    model.learning_rate = 1e-3
    
    
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    import signal
    signal.signal(signal.SIGUSR1, melk)
    if opt.train:
        trainer.fit(model, data, ckpt_path=ckpt)
            
    if not opt.no_test:
        trainer.test(model, data)

def create_argparser():
    defaults = dict(
        mode='train',
        deterministic=True,
        train=True,
        no_test=True,
        accumulate_grad_batches=1,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=['train.yaml'],
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="train_for_hydroforcast",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    return parser
if __name__ == "__main__":
    main()