import os
from aitsisui.elements import Col, Row, Col, Input, Image, Button, Row, Col, Check, Label
from aitsisui.core import Component
from aitsisui import app
from aitools_frontend.components import InputFile, Output, Slider,Checkbox, Prompt
from PIL import Image as PILImage

from diffusers import AutoPipelineForImage2Image
import numpy as np
import torch
import uuid

default_prompts = {
    "history": {
        "prompt-1": "a painting of colorful flowers on a blue background, inspired by Max Slevogt, fabrics textiles, trending on artstattion, seamless pattern, drawing technique Oil Painting, smooth, colorful, vibrant colors, elegant, beautiful, natural lighting, beautiful lighting, masterpiece, high resolution, 16k, vivid colors, painting, bright colors, bloom, flowers, vintage",
        "prompt-2": "a pattern of orange and blue flowers with leaves on a white background, luxury brand, Tom Lowell style, dense brushstrokes, drawing technique Tempera,  high quality, smooth, colorful, vibrant colors, high contrast, beautiful, dynamic lighting, 16k, high resolution, painting, digital illustration, white background, trending, bloom, wallpaper, colourful,  texture",
        "prompt-3": "a bunch of blue and yellow flowers on a white background, rococo and baroque styles, highly inventive pattern cutting, drawing technique Marbling Paint,soft,soft lighting,  a flemish Baroque by Annabel Kidston, pixiv, rococo, baroque wallpaper, flowery wallpaper, floral wallpaper,Baroque,Classical Realism, muted brown, green leaves",
        "prompt-4": "a pattern of soft blue!!!! and  soft color flowers on a white background, green leaves, , drawing technique Pastel, a flemish Baroque inspired by Annabel Kidston, rococo, roses background, baroque wallpaper, muted brown, mystical colors texture, romantic, brush stroke, smooth gradients, vivid color, flowery wallpaper, floral wallpaper,Baroqu",
        "prompt-5": "a painting of a bunch of yellow flowers, an ultrafine detailed painting by Louise Abbéma, behance contest winner, modern european ink painting, gold flaked flowers, dark flower pattern wallpaper, chaotic gold leaf flowers, drawing technique Fresco, smooth, colorful, vibrant colors, beautiful, natural lighting, 16k, Mannerism",
        "prompt-6": "a painting of colorful ethnical on a beige background, embroidery, embroideredi inspired by Max Slevogt, fabrics textiles, trending on artstattion, seamless pattern, drawing technique Sketch,colorful, vibrant colors, elegant, beautiful, natural lighting, beautiful lighting, masterpiece, high resolution, 16k, vivid colors, painting, bright colors, vintage,Sketch",
        "prompt-7": "a black and beige ethnical floral pattern on a vivid red background, rococo, a flemish Baroque inspired by William Morris,damask pattern, wallpaper,insanely complex details, high quality fabrics textiles, Abstract Illusionism, Baroque, Renaissance, Ink, highly detailed, hyper detailed, insanely detailed, high quality, fine details, sharp focus, colorful, vibrant colors, high contrast, elegant, beautiful, masterpiece, 16k, vivid colors",
        "prompt-8": "a close up of a colorful pattern on a black background, inspired by Peter Alexander Hay, shutterstock, brush stroke, in pastel colors, abstract background, spotted, colorful palette illustration, smooth, abstract,Abstract Illusionism,Expressionism, shuffled color, vivid colors, volumetric lightning, colorful, glowing, wallpaper",
        "prompt-9": "a painting of colorful flowers, red and white on a blue background, inspired by Max Slevogt, fabrics textiles, trending on artstattion, seamless pattern, drawing technique Oil Painting, smooth, colorful, vibrant colors, elegant, beautiful, natural lighting, beautiful lighting, masterpiece, high resolution, 16k, vivid colors, painting, bright colors, bloom, flowers, vintage, ",
        "prompt-10": "a close up of a multicolored area rug, a digital rendering, inspired by Hermenegildo Anglada Camarasa, bargello pattern, magma cascades, volcanic, vertical wallpaper, press shot, maroon, fabrics, textile fabrics,Abstract Art,Abstract Illusionism, op art, textured, blue stripe and yellow stripe, fractal art, 3d render",
        "prompt-11": "a blue, orange, and purple abstract pattern, a digital rendering, multicolored, brushstroke, highend, lined up horizontally, abstract patern, various colors, tempest, high res, rigorous,  weaving, displayed, thin stroke, ethnic, experiment, diagonal stripe,Abstract Illusionism,Expressionism, fractal art, Tempera",
        "prompt-12": "a close up of a red abstract print fabric, a digital rendering, inspired by Mirko Rački, stylized bold outline, cavewoman, color scheme, luxury fashion illustration, fineartamerica, complex pattern, chrome art, seamless pattern design, abstract smokey roses, safari, an illustration, colorful, vivid color, Marbling Paint style",
        "prompt-13": "a blue, orange, and purple abstract pattern, a digital rendering, multicolored, brushstroke, highend, lined up horizontally, abstract patern, various colors, tempest, high res, rigorous,  weaving, displayed, thin stroke, ethnic, experiment, diagonal stripe,Abstract Illusionism,Expressionism, fractal art, Tempera",
        "prompt-14": " a multicolored diagonal striped pattern in green, black, and blue, a digital rendering, bargello pattern, light boho carpet, second colours - green, pagoda, afar, colombian, sienna, punching, indigenous, maintenance, performance, 3 colours, at home, barlow, near the sea, op art, drawing technique Oil Painting, diagonal"
    }
}


pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipe.to("cuda", torch.float16)

class main(Component):
    def __init__(self, id=None, autoBind=True, **kwargs):
        super().__init__(id=id, autoBind=autoBind, **kwargs)
        self.style("height", "100vh")
        with self:
            with Col():     
                with Row().cls("wrapper").style("height", "10vh !important").style("width", "100%").cls("larger"):           
                    self.prompt = Prompt(history=default_prompts["history"]).style("width", "90%").style("height", "100%")
                    self.prompt.on("change", self.set_prompt)

                    with Col().style("width", "10%").style("height", "100%").style("justify-content", "center").style("align-items", "center"):
                        Button(value="Generate").on("click", self.generate).style("border", "1px white solid")
                        Button(value="Clear").on("click", self.clear).style("border", "1px white solid")

                with Row().cls("wrapper").style("height", "10% !important").style("justify-content", "flex-start"):
                    self.strength = Slider(value=0.5, min=0, max=1, step=0.01, label="Strength").style("width", "10%").style("gap", "0")
                    self.guidance_scale = Slider(value=0.0, min=0, max=15, step=0.01, label="Guidance Scale").style("width", "10%").style("gap", "0")
                    self.steps = Slider(value=4, min=1, max=50, step=1, label="Steps").style("width", "10%").style("gap", "0")
                    self.seed = Checkbox(label="Seed tut.",value=False).style("width", "10%").style("gap", "0")
                    self.seed_text = Input(value=-1)
                    self.seed_text.on("change", self.set_seed_text)
                    self.seed.on("click", self.set_seed)
                    self.label = Label(value="SDXL").style("width", "10%").style("gap", "0")
                
                with Row().cls("wrapper").style("gap","0").style("height", "80% !important"):
                    with Col().cls("wrapper").style("gap","0"):
                        fileDir = os.path.dirname(os.path.realpath('__file__'))
                        self.img = InputFile(save_path=os.path.join(fileDir, 'images'))
                        self.img.style("height", "1024px")
                        
                    with Col().cls("wrapper").style("gap","0"):
                        self.output = (
                            Output(
                                value="AIT_AI_LOGO.png",
                                tool='imagine'
                            )
                        )
                        self.output.style("height", "1024px")
                
                
    
    def set_seed(self, id, value):
        if self.seed_text.value == -1:
            self.seed_text.value = self.seed_value
        else:
            self.seed_text.value = -1
    
    def set_seed_text(self, id, value):
        self.seed_text.value = value

    def set_prompt(self, id, value):
        self.prompt.value = value
        self.prompt.textArea.value = value
    
    def clear(self, id, value):
        self.prompt.value = ""
        self.prompt.textArea.value = ""

    def generate(self,id,value):
        torch.cuda.empty_cache()
        uuid_str = str(uuid.uuid4())
        #uuid_str = "output"
        output_path = f"output/{uuid_str}.png"

        image = PILImage.open(self.img.value).convert("RGB")
        print(image.width, image.height)
        self.seed_value = int(self.seed_text.value)

        if self.seed_value == -1:
            self.seed_value =np.random.randint(0, np.iinfo(np.int32).max)

        generator = torch.Generator().manual_seed(self.seed_value)

        self.output.loading()
        print("Generating...")
        print(f"Seed: {self.seed_value}")
        print(f"Strength: {self.strength.value}")
        print(f"Guidance Scale: {self.guidance_scale.value}")
        print(f"Steps: {self.steps.value}")
        print(f"Prompt: {self.prompt.textArea.value}")

        output = pipe(self.prompt.value, 
                      image=image, 
                      generator=generator,
                      num_inference_steps=int(self.steps.value), 
                      strength=self.strength.value, 
                      guidance_scale=self.guidance_scale.value, 
                      width=image.width, 
                      height=image.height).images[0]
        output.save(output_path)
        
        self.output.loading()
        self.output.set_image(img_URL=output_path)

if __name__ == '__main__':    
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    app.add_static_route('images', os.path.join(fileDir, 'images'))
    app.add_static_route('output', os.path.join(fileDir, 'output'))
    app.add_static_route(
        "comp_static",
        osDirPath=(
            os
            .path
            .abspath(os.path.join(os.path.dirname(__file__), "aitools_frontend/components/static"))
        )
    )
    app.run(ui = main, debug=True, port=3000)