

from aitsisui import app
from aitsisui.core import Session, index_gen, Element, Component
from aitsisui.elements import Text, File, Label, Col, ImageViewer, Button, ImageCropper, Image
index_gen.add_css("inputfile-css", """                                  
.dropzone-label{
    width: 100%;
    height: 100%;
    border-radius: 3px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    color: gray;
    font-size: 20px;
    font-weight: 600;
    cursor: pointer;
}
                  
.file-input{
    display: block !important;
    width: 100%;
    height: 100%;
    opacity: 0;
    cursor: pointer;
    position: absolute;
    z-index: 1;

}

                  """)

class Comp_InputFile(Component):
    def __init__(
            self,
            id=None,
            save_path=None,
            viewer="seadragon",
            autoBind=True,
            afterUploadFn=None,
            useAPI=False,
            **kwargs
        ):
        super().__init__(id=id, autoBind=autoBind, **kwargs)
        self.save_path = save_path
        self.afterUploadFn = afterUploadFn
        self.useAPI = useAPI
        self.style("height","100%").style("width","100%").style("position","relative")

        with self:
            with (
                Button()
                    .style("z-index", "4")
                    .style("background-color", "rgba(0,0,0,0.5)")
                    .style("position", "absolute")
                    .style("top", "6px")
                    .style("left", "6px")
                    .style("min-width", "auto !important")
                    .on("click", self.resetImage)
                ):
                Image(value="comp_static/deleteButton.svg")
            self.file = File(
                save_path=self.save_path,
                on_upload_done=self.on_upload_done,
                useAPI=self.useAPI,
            ).cls("file-input")
            with Label(usefor=self.file.id).cls("dropzone-label"):          
                with (
                    Col()
                        .cls("wall hall")
                        .style("gap", "10px")
                        .style("justify-content", "center")
                        .style("align-items", "center")
                        .style("position", "absolute") as dropzone_inside
                ):
                    self.dropzone_inside = dropzone_inside
                    Text(value="Click to upload")
                if viewer == "seadragon":
                    self.dropzone_image = ImageViewer(
                        value=self.value,
                        hasButtons=False,
                        ableToZoom=True
                    ).cls("wall hall").style("display", "none").style("z-index", "3")

                else:
                    self.dropzone_image = ImageCropper(
                        value=self.value
                    ).cls("wall hall").style("display", "none").style("z-index", "3")

    def on_upload_done(self,file):
        try:
            if isinstance(file, dict) and self.useAPI:
                self.dropzone_inside.set_style("display", "none")
                self.dropzone_image.set_style("display", "flex")
                self.dropzone_image.value = file['data']['url']
                self.value = file['data']['_id']
                if self.afterUploadFn is not None:
                    self.afterUploadFn()
            elif isinstance(file, str) and not self.useAPI:
                file_name = os.path.basename(file)
                file_url = f"{upload_URI_prefix}/{file_name}"
                self.dropzone_inside.set_style("display", "none")
                self.dropzone_image.set_style("display", "flex")
                self.dropzone_image.value = file_url
                self.value = file
                if self.afterUploadFn is not None:
                    self.afterUploadFn()
            else:
                pass
        except Exception as e:
            pass

    def setImage(self, options):
        if not self.useAPI:
          url = options.get("url", None)
          path = options.get("path", None)
          if url and path is not None:
              self.dropzone_inside.set_style("display", "none")
              self.dropzone_image.set_style("display", "flex")
              self.dropzone_image.value = url
              self.value = path
        elif self.useAPI:
          url = options.get("url", None)
          imageId = options.get("_id", None) or options.get("id", None)
          if url and imageId is not None:
              self.dropzone_inside.set_style("display", "none")
              self.dropzone_image.set_style("display", "flex")
              self.dropzone_image.value = url
              self.value = imageId
        else:
            self.send(self.id, 'No image to set', 'alert')

    def resetImage(self, id, value):
        if self.value and self.dropzone_image.value:                
            self.dropzone_inside.set_style("display", "flex")
            self.dropzone_image.set_style("display", "none")
            # Both Classes have closeImage method
            self.dropzone_image.closeImage()
            self.dropzone_image.value = None
            self.file.value = None
            self.value = None
        else:
            self.send(self.id, 'No image to reset', 'alert')

if __name__ == "__main__":
    app.run(port=4000,debug=True, ui=Comp_InputFile)