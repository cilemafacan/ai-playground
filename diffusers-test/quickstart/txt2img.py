#----------------------------------------------------------------------------------------#
# Add diffusers path to sys path
import sys
import os
from dotenv import load_dotenv
# Load the .env file
load_dotenv()
# Get Diffusers path from environment variable
diffusers_path = os.getenv('DIFFUSERS_PATH')
print(f"Diffusers path: {diffusers_path}")
if diffusers_path is None:
    raise ValueError("Please set DIFFUSERS_PATH environment variable to Diffusers path")
sys.path.append(diffusers_path+"/src")
#----------------------------------------------------------------------------------------#


from diffusers import StableDiffusionPipeline

import torch
models = [r'f:\sd_models\ghostmix_v2',
          r'f:\sd_models\aesteticmix',
          r'f:\sd_models\revAnimated_v122',]
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(models[2], safety_checker=None)
pipe = pipe.to(device, torch.float16)

image = pipe("(sharp, crisp, masterpiece, top quality, best quality, official art, beautiful and aesthetic:1.2), (1girl), extreme detailed,(fractal art:1.3),colorful,highest detailed:1.3"
                        ,num_inference_steps = 30
                        ,height = 768
                        ,width = 512
                        , guidance_scale = 6
                        , negative_prompt = "(worst quality, low quality:2), monochrome, zombie,overexposure, watermark,text,bad anatomy,bad hand,extra hands,extra fingers,too many fingers,fused fingers,bad arm,distorted arm,extra arms,fused arms,extra legs,missing leg,disembodied leg,extra nipples, detached arm, liquid hand,inverted hand,disembodied limb, small breasts, loli, oversized head,extra body,completely nude, extra navel,easynegative,(hair between eyes),sketch, duplicate, ugly, huge eyes, text, logo, worst face, (bad and mutated hands:1.3),  (blurry:2.0), horror, geometry, bad_prompt, (bad hands), (missing fingers), multiple limbs, bad anatomy, (interlocked fingers:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), ((2girl)), (deformed fingers:1.2), (long fingers:1.2),(bad-artist-anime), bad-artist, bad hand, extra legs ,(ng_deepnegative_v1_75t)"
                        ).images[0]
# save the image
image.save('out.png')