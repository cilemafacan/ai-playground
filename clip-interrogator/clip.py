import os
import csv
import torch
import dotenv
from PIL import Image
from clip_interrogator import Interrogator, Config
dotenv.load_dotenv()

sf_models_path = os.getenv("SF_MODELS_PATH")


"""
Bu fonksiyon CLIP modelini yükler ve bir Interrogator nesnesi oluşturur.

Returns:
    Interrogator: CLIP modelini yüklenmiş ve hazır bir Interrogator nesnesi.

"""
def load_interrogator():
    caption_model_name = "blip-base"
    clip_model_name = "ViT-H-14/laion2b_s32b_b79k"
    config = Config()
    config.cache_path = sf_models_path
    print("Downloading models to {}".format(config.cache_path))
    config.clip_model_name = clip_model_name
    config.caption_model_name = caption_model_name
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    ci = Interrogator(config)
    return ci


"""
Bu fonksiyon bir görüntüyü alır ve bir prompt oluşturur.

Args:
    image (PIL.Image): Prompt oluşturulacak görüntü.
    mode (str): Prompt oluşturma modu. ['best', 'classic', 'fast', 'negative']

Returns:
    str: Prompt.

"""
def image_to_prompt(image : Image, mode : str):
    ci.config.chunk_size = 1024
    ci.config.flavor_intermediate_count = 1024
    image = image.convert('RGB')
    if mode == 'best':
        return ci.interrogate(image)
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    elif mode == 'fast':
        return ci.interrogate_fast(image)
    elif mode == 'negative':
        return ci.interrogate_negative(image)

"""
Bu fonksiyon birden fazla görüntüyü bir klasörden alır ve her biri için 
bir prompt oluşturur ve bir csv dosyasına kaydeder.

Args:
    dir (str): Görüntülerin bulunduğu klasörün yolu.

Returns:
    list: Prompt'lar ve görüntü yolları.

"""
def images_to_prompt(dir : str):
    folder_path = dir 
    prompt_mode = 'best' 
    csv_path = 'desc.csv' 

    supported_formats = ('jpg', 'jpeg', 'png', 'bmp', 'tiff' )
    files = [f for f in os.listdir(folder_path) if f.endswith(supported_formats)] if os.path.exists(folder_path) else []
    prompts = []

    for file in files:
        img_path = os.path.join(folder_path, file)
        image = Image.open(img_path)
        prompt = image_to_prompt(image, prompt_mode)
        prompts.append({"path": img_path, "prompt": prompt})
    
    if len(prompts):
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for prompt in prompts:
                for filename, prompt in prompt.items():
                    try:
                        writer.writerow([filename, prompt])
                    except Exception as e:
                        print(e)
                        pass

    return prompts

ci = load_interrogator()
