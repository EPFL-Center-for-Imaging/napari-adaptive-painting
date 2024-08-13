import napari
import skimage.io
import skimage.data

viewer = napari.Viewer()

image = skimage.io.imread("/home/wittwer/data/mouse.tif")
image = image / 255.0
image = image.astype(float)

viewer.add_image(image)

annotation = skimage.io.imread("/home/wittwer/data/mouse_annotation.tif")
viewer.add_labels(annotation)
print(annotation.shape)

from src.napari_adaptive_painting import LabelPropagatorWidget

widget = LabelPropagatorWidget(viewer)
viewer.window.add_dock_widget(widget)

if __name__ == "__main__":
    napari.run()
