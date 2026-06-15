# this DDPM frame comes from the openAI github  https://github.com/openai/improved-diffusion   We overwrote the original DDPM architecture as our training framework (class RunoffDDPM)
import os
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
import torch.nn.functional as F
from util import  exists, default, count_params, instantiate_from_config
from modules.ema import LitEma
from util import make_beta_schedule, extract_into_tensor




__conditioning_keys__ = {
                         'runoff': 'y'}

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 monitor=None,
                 use_ema=True,
                 runoff_key="",
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 geo_key='',
                 past_fast_key = '',
                 future_weather_key = ''   
                
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        self.use_small_t_weight = False 
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.runoff_key = runoff_key
        self.geo_key = geo_key
        self.past_fast_key = past_fast_key
        self.future_weather_key = future_weather_key
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")





    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None,cond=None,weights=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t, cond)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")
        elem_raw = self.get_loss(model_out, target, mean=False)
        elem = elem_raw   
      
        
       
        if elem.dim() == 4:
            loss_simple = elem.mean(dim=[1, 2, 3])
        elif elem.dim() == 3:
            loss_simple = elem.mean(dim=[1, 2])
        else:
            loss_simple = elem.mean(dim=[1])
        log_prefix = 'train' if self.training else 'val'
        if self.use_small_t_weight:
            T = self.num_timesteps
            t_weight = 1.0 + self.small_t_alpha * (1.0 - t.float() / max(T - 1, 1))
            loss_simple = loss_simple * t_weight
        
        loss_dict.update({f'{log_prefix}/loss_simple': loss_simple.mean()})
        if self.learn_logvar:
           
            logvar_t = self.logvar[t].to(self.device)
            
            
            loss = loss_simple * torch.exp(-logvar_t) + logvar_t
            
            loss_dict.update({f'{log_prefix}/logvar_value': self.logvar.data.mean()})
        else:
            loss = loss_simple
        
        loss = loss.mean() * self.l_simple_weight
        
        #loss_vlb = (self.lvlb_weights[t] * elem).mean()
        vlb_w = self.lvlb_weights[t]
        while vlb_w.dim() < elem.dim():
            vlb_w = vlb_w.unsqueeze(-1)

        vlb_elem = vlb_w * elem_raw

        if vlb_elem.dim() == 4:
            loss_vlb = vlb_elem.mean(dim=[1,2,3]).mean()
        elif vlb_elem.dim() == 3:
            loss_vlb = vlb_elem.mean(dim=[1,2]).mean()
        else:
            loss_vlb = vlb_elem.mean(dim=[1]).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        if isinstance(x, torch.Tensor):
            x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.runoff_key)
        cond_dist = self.get_input(batch, self.runoff_key)
        weather = self.get_input(batch, self.runoff_key)
        cond = [weather, cond_dist]
        loss, loss_dict = self(x, cond=cond)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,batch_size=128,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,batch_size=128,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, batch_size=128, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, batch_size=128, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)


    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
  

class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'runoff']

    def forward(self, x, t, c_concat: list = None,  force_uncond=False,force_drop_future=False):
        
        if self.conditioning_key == 'runoff':
            ppast_fast = c_concat[0]
            future_weather = c_concat[1]
            geo_data = c_concat[2]
            #past_slow_vec = c_concat[3]
            out = self.diffusion_model(x,t, ppast_fast, future_weather, geo_data,force_uncond=force_uncond,force_drop_future=force_drop_future)    
        else:
            raise NotImplementedError()

        return out
    
class RunoffDDPM(DDPM):
    def __init__(self, flood_alpha=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)  
        self.flood_alpha = flood_alpha
    def forward(self, x, *args, **kwargs):
       
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)
    
    def apply_model(self, x, t, c, force_drop_future, force_uncond=False):
        return self.model(x, t, c,force_uncond=force_uncond,force_drop_future=force_drop_future)

    def get_input(self, batch, k):
        x = batch[k]
        if isinstance(x, torch.Tensor):
            x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        runoff = self.get_input(batch, self.runoff_key)
        geo = self.get_input(batch, self.geo_key)
        past_fast = self.get_input(batch, self.past_fast_key)
        
        future_weather = self.get_input(batch, self.future_weather_key)
        cond = [past_fast, future_weather, geo]
        weights = None
        
        loss, loss_dict = self(runoff, cond=cond,weights=weights)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,batch_size=256,
                      logger=True, on_step=True, on_epoch=True, sync_dist=True)

        self.log("global_step", self.global_step,batch_size=256,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True,batch_size=256)
        self.log_dict(loss_dict_ema, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True,batch_size=256)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)


    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt 
    



