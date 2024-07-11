import numpy as np
import PIL
import torch

from typing import Union, List, Optional

def normalize(image : Optional[torch.FloatTensor]):
    """
    Normalize an image to be between -1 and 1
    """
    return 2.0 * image - 1.0

def denormalize(image: Optional[torch.FloatTensor]):
    """
    Denormalize an image from -1 and 1 to 0 and 1
    """
    return (image / 2 + 0.5).clamp(0, 1)

def resize(image: PIL.Image.Image, height: Optional[int] = None, width: Optional[int] = None) -> PIL.Image.Image:
    """
    Resize an image
    """
    if image is not None and not isinstance(image, PIL.Image.Image):
        raise TypeError(f"Input must be a PIL Image. But got {type(image)}")

    if height is None:
        height = image.height
    if width is None:
        width = image.width
    
    width, height = (x - x % 8 for x in (width, height)) # Make sure width and height are divisible by 8
    
    resized_image = image.resize((width, height), PIL.Image.BICUBIC)

    return resized_image

def image_preprocess(image: Union[torch.FloatTensor, PIL.Image.Image, np.ndarray],
                    height: Optional[int] = None,
                    width: Optional[int] = None,
                    do_resize: Optional[bool] = True,
                    do_normalize: Optional[bool] = True) -> torch.Tensor:
    """
    Preprocess an image
    """
    if isinstance(image, torch.Tensor):
        if do_resize:
            np_image = np.array(image)
            pil_image = PIL.Image.fromarray(np_image)
            resized_image = resize(pil_image, height, width)
            np_image = np.array(resized_image)
        else:
            np_image = image
        if len(np_image.shape) == 2:  # Single-channel image, add channel dimension
            np_image = np.expand_dims(np_image, axis=2)
        
        tensor_image = torch.from_numpy(np_image)
        tensor_image = torch.cat([tensor_image], dim=0)
        
    elif isinstance(image, PIL.Image.Image):
        if do_resize:
            image = resize(image, height, width)
        np_image = np.array(image)
        if len(np_image.shape) == 2:  # Single-channel image, add channel dimension
            np_image = np.expand_dims(np_image, axis=2)
        tensor_image = torch.from_numpy(np_image)

    elif isinstance(image, np.ndarray):
        if do_resize:
            pil_image = PIL.Image.fromarray(image)
            resized_image = resize(pil_image, height, width)
            np_image = np.array(resized_image)
        else:
            np_image = image

        if len(np_image.shape) == 2:  # Single-channel image, add channel dimension
            np_image = np.expand_dims(np_image, axis=2)
        np_image = np.concatenate([np_image], axis=0)
        tensor_image = torch.from_numpy(np_image)
       
    else:
        raise TypeError(f"Input must be a torch tensor, PIL Image or numpy array. But got {type(image)}")
    
    if tensor_image.max() > 1.0:
        tensor_image = tensor_image / 255.0

    if do_normalize:
        tensor_image = normalize(tensor_image)

    
    tensor_image = tensor_image.unsqueeze(0).permute(0, 3, 1, 2)
    return tensor_image

def image_postprocess(tensor_image: Optional[torch.FloatTensor], 
                      do_denormalize: Optional[bool] = True) -> PIL.Image.Image:
        
    if not isinstance(tensor_image, torch.Tensor):
        raise TypeError(f"Input must be a torch tensor. But got {type(tensor_image)}")
        
    tensor_image = torch.cat([denormalize(tensor_image) if do_denormalize else tensor_image], dim=0)
    if tensor_image.max() <= 1.0:
        tensor_image = tensor_image * 255.0

    pil_image = PIL.Image.fromarray(tensor_image.squeeze().permute(1, 2, 0).numpy().astype(np.uint8))
    return pil_image