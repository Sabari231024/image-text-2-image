import torch 
import numpy as np 
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH  = WIDTH//8
LATENTS_HEIGHT = HEIGHT//8
#cfg -> classifier free guidance
def generate(prompt:str,uncond_prompt:str, #uncond_prompt or negative prompt you instruct something you don't want in the image
            input_image=None,strength=0.8,#strength tells how much attention the model should give to input image in output. more strength the moder the noise we add
            do_cfg=True,cfg_scale=7.5,
            sampler_name="ddpm",n_inference_steps=50,
            models={},seed=None,device=None,idle_device=None,tokenizer=None):

            with torch.no_grad():
                if not (0<strength<=1):
                    raise ValueError("strength must be between 0 and 1")

                if idle_device:
                    to_idle = lambda x: x.to(idle_device)
                else:
                    to_idle = lambda x: x
                
                generator = torch.Generator(device=device)
                if seed is None:
                    generator.seed()
                else:
                    generator.manual_seed(seed)
                
                clip = models["clip"]
                clip.to(device)

                if do_cfg:
                    #convert prompt to tokens
                    cond_tokens = tokenizer.batch_encode_plus([prompt],padding="max_length",max_length=77).input_ids
                    # (batch,seq_len)
                    cond_tokens = torch.tensor(cond_tokens,dtype=torch.long,device=device)
                    #convert tokens to embeddings (batch,seq_len ,768)
                    cond_context = clip(cond_tokens)
                    uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt],padding="max_length",max_length=77).input_ids
                    uncond_tokens = torch.tensor(uncond_tokens,dtype=torch.long,device=device)
                    uncond_context = clip(uncond_tokens)
                    context = torch.cat([cond_context,uncond_context]) #(2,77,768)
                else:
                    tokens = tokenizer.batch_encode_plus([prompt],padding="max_length",max_length=77).input_ids
                    tokens = torch.tensor(tokens,dtype=torch.long,device=device)
                    context = clip(tokens) #(1,77,768)
                to_idle(clip)
                
                if sampler_name == "ddpm":
                    sampler = DDPMSampler(generator)
                    sampler.set_inference_timesteps(n_inference_steps)
                else:
                    raise ValueError("Invalid sampler name")

                latents_shape = (1,4,LATENTS_HEIGHT,LATENTS_WIDTH)
                if input_image:
                    encoder = models["encoder"]
                    encoder.to(device)
                    input_image_tensor = input_image.resize((WIDTH,HEIGHT))
                    input_image_tensor = np.array(input_image_tensor)
                    input_image_tensor = torch.tensor(input_image_tensor,dtype=torch.float32,device=device)
                    input_image_tensor = rescale(input_image_tensor,(0,255),(-1,1))
                    input_image_tensor = input_image_tensor.unsqueeze(0) # adds batch dimension
                    input_image_tensor = input_image_tensor.permute(0,3,1,2) # (batch,channels,height,width)
                    encoder_noise = torch.randn(latents_shape,generator=generator,device=device)
                    #let's pass it to the encoder
                    latents = encoder(input_image_tensor,encoder_noise)
                    sampler.set_strength(strength)
                    latents = sampler.add_noise(latents,sampler.timesteps[0])
                    to_idle(encoder)
                else: #text to image
                    latents = torch.randn(latents_shape,generator=generator,device=device)
                
                diffusion = models["diffusion"]
                diffusion.to(device)
                # so 1-1000 are time steps if 50 steps then 1000/50 = 20 so 1000->980->...->20->1
                timesteps = tqdm(sampler.timesteps)
                for i,timestep in enumerate(timesteps):
                   time_embedding = get_time_embedding(timestep).to(device) #(1,320)
                   model_input = latents
                   if do_cfg:
                       model_input = model_input.repeat(2,1,1,1) # making two copies of the same latent one used with prompt nad another without prompt
                   model_output = diffusion(model_input,context,time_embedding)
                   if do_cfg:
                       output_cond,output_uncond = model_output.chunk(2) #splitting the model output into two
                       model_output = output_uncond + (output_cond-output_uncond)*cfg_scale
                   latents = sampler.step(timestep,latents,model_output) #remove the predicted noise by the unet

                to_idle(diffusion)
                decoder = models["decoder"]
                decoder.to(device)
                images = decoder(latents)
                to_idle(decoder)
                images = rescale(images,(-1,1),(0,255),clamp=True)
                images = images.permute(0,2,3,1)
                images = images.to("cpu",torch.uint8).numpy()
                return images[0]

def rescale(x,old_range,new_range,clamp=False):
    old_min,old_max = old_range
    new_min,new_max = new_range
    x -= old_min
    x *= (new_max-new_min)/(old_max-old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min,new_max)
    return x

def get_time_embedding(timestep):
    freq = torch.pow(10000, -torch.arange(start=0,end=160,dtype=torch.float32)/160) #(160,)
    x = torch.tensor([timestep],dtype=torch.float32)[:,None]*freq[None] #(1,160)
    x = torch.cat([torch.cos(x),torch.sin(x)],dim=-1) #(1,320)
    return x






                


                    

