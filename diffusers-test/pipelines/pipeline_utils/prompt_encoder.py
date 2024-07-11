import torch

from typing import Union, List, Optional
from transformers import CLIPTokenizer, CLIPTextModel
from fake_triton_modules import TritonCLIPTokenizer, TritonCLIPTextModel

def encode_prompt(tokenizer : Union[CLIPTokenizer, TritonCLIPTokenizer],
                  text_encoder : Union[CLIPTextModel, TritonCLIPTextModel],
                  prompt : Union[str,list], 
                  negative_prompt : Union[str,list], 
                  num_images_per_prompt : Optional[int] = 1, 
                  cfg: Optional[bool] = False):
    
    if tokenizer is not None and isinstance(tokenizer, CLIPTokenizer):
        tokenizer = tokenizer
    elif tokenizer is not None and isinstance(tokenizer, TritonCLIPTokenizer):
        pass

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
   
    # Prompt tokenize
    tokens = tokenizer(prompt,
                       padding="max_length",
                       max_length=tokenizer.model_max_length,
                       truncation=True,
                       return_tensors="pt")
    
    tokens_ids = tokens.input_ids

    if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        attention_mask = tokens.attention_mask
    else:
        attention_mask = None
  
    # Prompt embeds
    prompt_embeds = text_encoder(tokens_ids, attention_mask=attention_mask) # text encoder encode edilmiş promptu döner
    prompt_embeds = prompt_embeds[0]

    # duplicate prompt embeds for each image
    bs_embed, seq_len, _ = prompt_embeds.shape 
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
 

    # negative prompt
    if negative_prompt is None:
        uncond_tokens = [""] * batch_size

    elif isinstance(negative_prompt, str):
        uncond_tokens = [negative_prompt]
    else:
        uncond_tokens = negative_prompt
    
    # negative prompt tokenize
    max_length = prompt_embeds.shape[1]
    uncond_tokens = tokenizer(uncond_tokens,
                             padding="max_length",
                             max_length=max_length,
                             truncation=True,
                             return_tensors="pt")

    if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        attention_mask = uncond_tokens.attention_mask
    else:
        attention_mask = None

    uncond_tokens_ids = uncond_tokens.input_ids
    negative_prompt_embeds = text_encoder(uncond_tokens_ids, attention_mask=attention_mask)
    negative_prompt_embeds = negative_prompt_embeds[0]

    if cfg:
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    return prompt_embeds