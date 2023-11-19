# 🐍 napari-snakes

Propagate object annotations in Napari using active contours (*snakes*). This tool is based on the [`morphological_geodesic_active_contour`](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.morphological_geodesic_active_contour) function of Scikit-image.

<p align="center">
    <img src="https://github.com/MalloryWittwer/napari-snakes/blob/main/assets/screenshot.gif" height="400">
</p>

## Installation

You can install `napari-snakes` via [pip]:

    pip install napari-snakes

## Usage

- Select the plugin from the `Plugins` menu of Napari.
- Open an image to annotate (2D, RGB, 2D+t, or 3D).
- Click on the button "Start active contour tracking" or press `S`. A new `Labels` layer *Snakes (current edit)* should appear.

**Options and parameters**

## Contributing

Contributions are very welcome. Please get in touch if you'd like to be involved in improving or extending the package.

## License

Distributed under the terms of the [BSD-3] license,
"napari-snakes" is free and open source software

## Issues

If you encounter any problems, please file an issue along with a detailed description.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
