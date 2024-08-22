import sys
import tifffile
import napari
from napari_adaptive_painting import LabelPropagatorWidget

if __name__ == "__main__":
    _, image_file = sys.argv
    image = tifffile.imread(image_file)
    if len(image.shape) != 3:
        print(f"Please provide a 3D or 2D+time image (shape found in your image: {image.shape})")
        sys.exit()
    
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(LabelPropagatorWidget(viewer))
    viewer.add_image(image)
    napari.run()