import torch
import numpy as np 

class DDPMSampler:
    def __init__(self,generator:torch.Generator,num_training_steps=1000,beta_start:float=0.00085,beta_end:float=0.0120):#1000 numbers between start and end
        self.betas = torch.linspace(beta_start**0.5,beta_end**0.5,num_training_steps,dtype=torch.float32)**2
        self.alphas = 1.0-self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas,0) # alpha0 ,alpha0*alpha1,alpha0*alpha1*alpha2 ..... 
        self.one = torch.tensor(1.0)
        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0,num_training_steps)[::-1].copy())
    def set_inference_timesteps(self,num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange(0,num_inference_steps)*step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
    
    def _get_prev_t(self,t:int)->int:
        prev_t = t - (self.num_training_steps // self.num_inference_steps)
        return prev_t
    
    def _get_variance(self,t:int)->torch.Tensor:
        prev_t = self._get_prev_t(t)
        alpha_prod_t = self.alpha_cumprod[t]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance,min=1e-20)
        return variance
    
    def set_strength(self,strength=1):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_Step = start_step
        
        

    def step(self,timesteps:int,latents:torch.Tensor,model_output:torch.Tensor):
        t = timesteps
        prev_t = self._get_prev_t(t)
        alpha_prod_t = self.alpha_cumprod[t]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t  =  1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        pred_original_sample = (latents - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents
        variance = 0
        if t>0:
            device = model_output.device
            noise = torch.randn(model_output.shape,generator=self.generator,device=device,dtype=model_output.dtype)
            variance = (self._get_variance(t)**0.5)*noise
        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample
        
    
    def add_noise(self,original_samples:torch.FloatTensor,timesteps:torch.IntTensor)->torch.FloatTensor:
        alpha_cumprod = self.alpha_cumprod.to(device=original_samples.device,dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        sqrt_alpha_cumprod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.flatten()
        while len(sqrt_alpha_cumprod.shape) < len(original_samples.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
        sqrt_one_minus_alpha_cumprod = (1.0 - alpha_cumprod[timesteps]) ** 0.5 #S.D
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.flatten()
        while len(sqrt_one_minus_alpha_cumprod.shape)<len(original_samples.shape):
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)
        noise = torch.randn(original_samples.shape,generator=self.generator,device=original_samples.device,dtype=original_samples.dtype)
        noisy_samples = (sqrt_alpha_cumprod*original_samples)+(sqrt_one_minus_alpha_cumprod)*noise # reparametrisation trick : mu + sd*noise get the sample
        return noisy_samples



