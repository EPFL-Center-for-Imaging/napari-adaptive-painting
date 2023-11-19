
import napari
import skimage.data

viewer = napari.Viewer()

viewer.add_image(skimage.data.coins())

from src.napari_snakes import SnakesWidget

widget = SnakesWidget(viewer)

viewer.window.add_dock_widget(widget)

if __name__ == '__main__':
    napari.run()