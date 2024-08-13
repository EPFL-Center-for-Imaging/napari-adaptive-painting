import napari
import napari.layers
from napari.utils.notifications import show_info
from PyQt5.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QComboBox,
    QSizePolicy,
    QLabel,
    QGridLayout,
    QPushButton,
)

import numpy as np
import scipy.ndimage as ndi
from skimage.filters import gaussian


def keep_biggest_object(lab_int: np.ndarray) -> np.ndarray:
    """Selects only the biggest object of a labels image."""
    labels = ndi.label(lab_int)[0]  # label from scipy
    counts = np.unique(labels, return_counts=1)
    biggestLabel = np.argmax(counts[1][1:]) + 1
    return labels == biggestLabel


class LabelPropagatorWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer
        self.viewer.text_overlay.text = "Adaptive painting is active."

        self.is_active = False

        # Layout
        grid_layout = QGridLayout()
        grid_layout.setAlignment(Qt.AlignTop)
        self.setLayout(grid_layout)

        # Image
        self.cb_image = QComboBox()
        self.cb_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(QLabel("Image", self), 0, 0)
        grid_layout.addWidget(self.cb_image, 0, 1)

        # Labels
        self.cb_labels = QComboBox()
        self.cb_labels.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(QLabel("Labels", self), 1, 0)
        grid_layout.addWidget(self.cb_labels, 1, 1)

        self.cb_labels.currentTextChanged.connect(self._handle_labels_changed)

        # Selected label info
        self.cd_label = QLabel(f"Selected label: {self.selected_label}", self)
        grid_layout.addWidget(self.cd_label, 2, 0, 1, 2)

        # Push button
        self.btn = QPushButton(self)
        self._set_button_text()
        self.btn.clicked.connect(self._on_button_push)
        grid_layout.addWidget(self.btn, 4, 0, 1, 2)

        # Setup layer callbacks
        self.viewer.layers.events.inserted.connect(
            lambda e: e.value.events.name.connect(self._on_layer_change)
        )
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)
        self._on_layer_change(None)

        # Viewer events
        self.viewer.dims.events.order.connect(self._handle_inactive)
        self.viewer.dims.events.ndisplay.connect(self._handle_inactive)

    @property
    def image_layer(self) -> napari.layers.Image:
        if self.cb_image.currentText() != "":
            return self.viewer.layers[self.cb_image.currentText()]

    @property
    def labels_layer(self) -> napari.layers.Image:
        if self.cb_labels.currentText() != "":
            return self.viewer.layers[self.cb_labels.currentText()]

    @property
    def image_data(self):
        """The image data, adjusted to handle the RGB case."""
        if self.image_layer is None:
            return

        if self.image_layer.data is None:
            return

        return self.image_layer.data

    @property
    def is_in_3d_view(self):
        return self.viewer.dims.ndisplay == 3

    @property
    def current_step(self):
        """Current step, adjusted based on the layer transpose state."""
        return self.viewer.dims.current_step[0]

    @property
    def selected_label(self):
        if self.labels_layer is None:
            return

        return self.labels_layer.selected_label

    @property
    def image_data_slice(self):
        """The currently visible 2D slice if the image is 3D, otherwise the image itself (if 2D)."""
        if self.image_data is None:
            return

        return self.image_data[self.current_step]

    @property
    def labels_data_slice(self):
        """The currently visible 2D slice if the image is 3D, otherwise the image itself (if 2D)."""
        if self.labels_layer is None:
            return

        return self.labels_layer.data[self.current_step]

    @property
    def previous_labels_data_slice(self):
        if self.labels_layer is None:
            return

        if self.current_step > 0:
            return self.labels_layer.data[self.current_step - 1]
        else:
            return self.labels_data_slice

    @property
    def next_labels_data_slice(self):
        if self.labels_layer is None:
            return

        max_step = len(self.labels_layer.data) - 1
        if self.current_step < max_step:
            return self.labels_layer.data[self.current_step + 1]
        else:
            return self.labels_data_slice

    def _on_layer_change(self, e):
        self.cb_image.clear()
        for x in self.viewer.layers:
            if isinstance(x, napari.layers.Image):
                if x.data.ndim == 3:
                    self.cb_image.addItem(x.name, x.data)

        self.cb_labels.clear()
        for x in self.viewer.layers:
            if isinstance(x, napari.layers.Labels):
                self.cb_labels.addItem(x.name, x.data)

    def _set_button_text(self):
        text = "Stop" if self.is_active else "Start"
        self.btn.setText(text)

    def _on_button_push(self):
        if self.image_layer is None:
            show_info("Select a valid image!")
            return

        if self.labels_layer is None:
            show_info("Select valid labels!")
            return

        if self.is_in_3d_view:
            show_info("Be in 2D view mode!")
            return

        self.is_active = not self.is_active

        if self.is_active:
            self._handle_active()
        else:
            self._handle_inactive()

    def _handle_active(self):
        self.is_active = True
        self._set_button_text()
        self.viewer.dims.events.current_step.connect(
            self._on_current_step_changed
        )
        self.viewer.text_overlay.visible = True

    def _handle_inactive(self, e=None):
        self.is_active = False
        self._set_button_text()
        self.viewer.dims.events.current_step.disconnect(
            self._on_current_step_changed
        )
        self.viewer.text_overlay.visible = False

    def _handle_labels_changed(self, selected_labels_name):
        for x in self.viewer.layers:
            if isinstance(x, napari.layers.Labels):
                x.events.selected_label.disconnect(
                    self._update_label_display_value
                )
        if self.labels_layer is not None:
            self.labels_layer.events.selected_label.connect(
                self._update_label_display_value
            )
            self._update_label_display_value(None)

    def _update_label_display_value(self, e):
        self.cd_label.setText(f"Selected label: {self.selected_label}")

    def _on_current_step_changed(self, event):
        init_mask = np.max(
            np.stack(
                (
                    self.previous_labels_data_slice,
                    self.labels_data_slice,
                    self.next_labels_data_slice,
                )
            ),
            axis=0,
        )

        if init_mask.sum() == 0:
            # print('Labels slices are empty.')
            return

        if (
            self.labels_layer.data[self.current_step] == self.selected_label
        ).sum():
            # Do not overwrite previous slices
            return

        # These parameters seem good - but they could be tweaked
        gaussian_sigma = 2.0
        n_iterations_dilation = 1
        intensity_quantile = 0.1

        mask = init_mask == self.selected_label
        max_intensity = np.quantile(
            self.image_data_slice[mask], q=1 - intensity_quantile
        )
        min_intensity = np.quantile(
            self.image_data_slice[mask], q=intensity_quantile
        )

        smoothed_image_data_slice = gaussian(
            self.image_data_slice, sigma=gaussian_sigma, preserve_range=True
        )
        new_mask = (
            ndi.binary_dilation(
                mask,
                structure=ndi.generate_binary_structure(2, 1),
                iterations=n_iterations_dilation,
            )
            & (smoothed_image_data_slice <= max_intensity)
            & (smoothed_image_data_slice >= min_intensity)
        )
        if new_mask.sum():
            new_mask = ndi.binary_fill_holes(new_mask)
            new_mask = keep_biggest_object(new_mask)

        self.labels_layer.data[self.current_step][
            new_mask
        ] = self.selected_label
        self.labels_layer.refresh()


if __name__ == "__main__":
    import tifffile

    viewer = napari.Viewer()
    viewer.window.add_dock_widget(LabelPropagatorWidget(viewer))
    viewer.add_image(tifffile.imread("/home/wittwer/data/amaia/1493.tif"))
    viewer.add_labels(
        tifffile.imread("/home/wittwer/data/amaia/1493-init-labels.tif")
    )
    napari.run()
