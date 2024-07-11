import os
from aitsisui.elements import Col, Row, Text, Col, Input, Image, Button, Row, Col, Dropzone
from aitsisui.core import Component
from aitsisui import app
import os
from aitsisui.core import Component, index_gen
from aitsisui.elements import Text, File, Label, Col, ImageViewer, Button, ImageCropper, Image

from aitools_frontend.components import InputFile
from aitools_frontend.components import AssestUI
from clip import image_to_prompt

from PIL import Image as PILImage



class main(Component):
    def __init__(self, id=None, autoBind=True, **kwargs):
        super().__init__(id=id, autoBind=autoBind, **kwargs)
        with self:
            self.cont = Col().cls("wrapper")
            with self.cont:                
                
                    self.prompt = Input(placeholder="Enter your prompt here").style("width", "100%").style("height", "10vh !important").cls("larger")
                    self.prompt.on("change", self.set_prompt)
                    Button(value="Generate").on("click", self.generate)
                
                    fileDir = os.path.dirname(os.path.realpath('__file__'))
                    self.img = InputFile(save_path=os.path.join(fileDir, 'images'))
                    self.img.style("width", "1024px").style("height", "1024px")

    def set_prompt(self, id, value):
        self.prompt.value = value
    def generate(self,id,value):
        image = PILImage.open(self.img.value)
        generated_prompt = image_to_prompt(image, "best")

        self.prompt.value = generated_prompt

if __name__ == '__main__':    
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    app.add_static_route('images', os.path.join(fileDir, 'images'))
    app.run(ui = main, debug=True, port=4000)