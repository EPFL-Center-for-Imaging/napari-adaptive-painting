![EPFL Center for Imaging logo](https://imaging.epfl.ch/resources/logo-for-gitlab.svg)
# 🖌️ napari-adaptive-painting

Propagate label annotations in Napari.

<p align="center">
    <img src="https://github.com/EPFL-Center-for-Imaging/napari-adaptive-painting/blob/main/assets/screenshot.gif" height="400">
</p>


## Installation

You can install `napari-adaptive-painting` via [pip](https://pypi.org/project/pip/):

    pip install napari-adaptive-painting

## Usage

- Select the plugin from the `Plugins` menu of Napari.
- Open an image to annotate (2D+t or 3D).
- Open, load, or create a `Labels` layer for annotating objects.
- Select the instance label to adaptively paint from the layer controls (tip: use the `Pick mode` to quickly select labels) or draw a new one (in the visible 2D plane).
- Click on *Start*.
- Move along the Z axis (which can also represent time). Whenever the Z plane changes, the label mask is adapted to match the new Z plane.

**Known limitations**

- The plugin won't work if the layers are transposed. Stick to the original layer orientation.

## Contributing

Contributions are welcome. Please get in touch if you'd like to be involved in improving or extending the package.

## License

Distributed under the terms of the [BSD-3](http://opensource.org/licenses/BSD-3-Clause) license, "napari-adaptive-painting" is free and open source software.

----------------------------------

This [napari](https://github.com/napari/napari) plugin is an output of a collaborative project between the [EPFL Center for Imaging](https://imaging.epfl.ch/) and the [De Palma Lab](https://www.epfl.ch/labs/depalma-lab/) in 2024.
