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
    # QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
)
# from superqt import QLabeledDoubleRangeSlider

import numpy as np
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
from skimage.exposure import rescale_intensity

from functools import partial

mgac = partial(
    morphological_geodesic_active_contour, 
    num_iter=20,  # higher = more accurate?
    smoothing=2, 
    balloon=0.0,  # positive = inflate, negative = shrink 
    threshold='auto'
)

def snakify(image_data_slice, init_mask, sigma: float=1.0):
    gradient_img = rescale_intensity(image_data_slice, out_range=(0, 1))
    gradient_img = inverse_gaussian_gradient(gradient_img, sigma=sigma)

    labels = init_mask.copy()

    for lab_idx in np.unique(labels).delete(0):
        old_label = (labels == lab_idx).astype(np.uint8)
        labels[old_label] = 0
        fresh_label = mgac(gradient_img, old_label)
        labels[fresh_label] = lab_idx
    
    return labels


class SnakesWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer
        self.viewer.text_overlay.text = 'Snakes tracking is active.'

        # self.image_layer = None
        self.labels_layer = None
        self.result_layer = None
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

        # Result
        self.cb_result = QComboBox()
        self.cb_result.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout.addWidget(QLabel("Labels (result)", self), 1, 0)
        grid_layout.addWidget(self.cb_result, 1, 1)

        # Flood fill parameters (group)
        algo_params_group = QGroupBox()
        algo_params_group.setTitle("Active contour parameters")
        algo_params_layout = QGridLayout()
        algo_params_group.setLayout(algo_params_layout)
        algo_params_group.layout().setContentsMargins(10, 10, 10, 10)
        grid_layout.addWidget(algo_params_group, 2, 0, 1, 2)

        # # Intensity range
        # self.thresholdSlider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        # self.thresholdSlider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # self.thresholdSlider.setRange(0, 1)
        # self.thresholdSlider.setValue((0.0, 1.0))
        # self.thresholdSlider.setEdgeLabelMode(QLabeledDoubleRangeSlider.LabelPosition.NoLabel)
        # self.thresholdSlider.setHandleLabelPosition(QLabeledDoubleRangeSlider.LabelPosition.NoLabel)
        # ff_layout.addWidget(QLabel("Intensity range", self), 0, 0)
        # ff_layout.addWidget(self.thresholdSlider, 0, 1)

        # Gaussian Gradient Sigma
        self.sigma = QDoubleSpinBox()
        self.sigma.setMinimum(0.0)
        self.sigma.setMaximum(10.0)
        self.sigma.setSingleStep(0.1)
        self.sigma.setValue(1.0)
        algo_params_layout.addWidget(QLabel("Sigma", self), 0, 0)
        algo_params_layout.addWidget(self.sigma, 0, 1)

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

        # Key bindings
        self.viewer.bind_key('s', lambda _: self._on_button_push())
        self.viewer.bind_key('Escape', self._handle_inactive)

        # Viewer events
        self.viewer.dims.events.order.connect(self._handle_inactive)
        self.viewer.dims.events.ndisplay.connect(self._handle_inactive)

    @property
    def image_layer(self) -> napari.layers.Image:
        if self.cb_image.currentText() != '':
            return self.viewer.layers[self.cb_image.currentText()]

    @property
    def image_data(self):
        """The image data, adjusted to handle the RGB case."""
        if self.image_layer is None:
            return
        
        if self.image_layer.data is None:
            return
        
        if self.image_layer.data.ndim == 2:
            return self.image_layer.data
        
        elif self.image_layer.data.ndim == 3:
            if self.image_layer.rgb is True:
                return np.mean(self.image_layer.data, axis=2)
            else:
                return self.image_layer.data
    
    @property
    def is_in_3d_view(self):
        return self.viewer.dims.ndisplay == 3

    @property
    def dims_displayed(self):
        return list(self.viewer.dims.displayed)
    
    @property
    def ndim(self):        
        if self.image_data is None:
            return
        
        if self.image_layer.rgb is True:
            return 2
        else:
            return self.image_layer.data.ndim
    
    @property
    def axes(self):
        if self.is_in_3d_view:
            return
        
        axes = self.dims_displayed
        if self.ndim == 3:
            axes.insert(0, list(set([0, 1, 2]) - set(self.dims_displayed))[0])
        
        return axes
    
    @property
    def current_step(self):
        """Current step, adjusted based on the layer transpose state."""
        return np.array(self.viewer.dims.current_step)[self.axes][0]
    
    @property
    def selected_label(self):
        if self.labels_layer is None:
            return
        
        return self.labels_layer.selected_label
    
    @selected_label.setter
    def selected_label(self, selected_label):
        if self.labels_layer is None:
            return
        
        self.labels_layer.selected_label = selected_label
    
    @property
    def image_data_slice(self):
        """The currently visible 2D slice if the image is 3D, otherwise the image itself (if 2D)."""      
        if self.image_data is None:
            return
        
        if self.ndim == 2:
            return self.image_data
        
        elif self.ndim == 3:
            return self.image_data.transpose(self.axes)[self.current_step]
    
    @property
    def results_data_slice(self):
        """The currently visible 2D slice if the image is 3D, otherwise the image itself (if 2D)."""
        if self.result_layer is None:
            return
        
        if self.ndim == 2:
            return self.result_layer.data
        
        elif self.ndim == 3:
            return self.result_layer.data.transpose(self.axes)[self.current_step]
    
    @property
    def labels_data_slice(self):
        """The currently visible 2D slice if the image is 3D, otherwise the image itself (if 2D)."""
        if self.labels_layer is None:
            return
        
        if self.ndim == 2:
            return self.labels_layer.data

        elif self.ndim == 3:
            return self.labels_layer.data.transpose(self.axes)[self.current_step]
    
    @property
    def intensity_range(self):
        return self.image_data_slice.max() - self.image_data_slice.min()
    
    def _on_layer_change(self, e):
        self.cb_image.clear()
        for x in self.viewer.layers:
            if isinstance(x, napari.layers.Image):
                if x.data.ndim in [2, 3]:
                    self.cb_image.addItem(x.name, x.data)

        self.cb_result.clear()
        for x in self.viewer.layers:
            if isinstance(x, napari.layers.Labels):
                self.cb_result.addItem(x.name, x.data)

    def _set_button_text(self):
        text = 'Stop' if self.is_active else 'Start'
        self.btn.setText(text)

    def _on_button_push(self):
        if self.image_layer is None:
            show_info('Select a valid image!')
            return
        
        if self.is_in_3d_view:
            show_info('Be in 2D view mode!')
            return
        
        if self.cb_result.currentText() == '':
            self.result_layer = self.viewer.add_labels(np.zeros_like(self.image_data, dtype=np.int_), name='Annotation')
        else:
            self.result_layer = self.viewer.layers[self.cb_result.currentText()]
        
        self.is_active = not self.is_active

        if self.is_active:
            self._handle_active()
        else:
            self._handle_inactive()
        
    def _handle_active(self):
        self.is_active = True
        self._set_button_text()

        # Create a Labels layer
        self.labels_layer = self.viewer.add_labels(np.zeros_like(self.image_data, dtype=np.int_), name='Snakes (current edit)')

        self.viewer.text_overlay.visible = True
    
    def _handle_inactive(self, e=None):
        self.is_active = False
        self._set_button_text()

        # Remove the Labels layer
        for idx, layer in enumerate(self.viewer.layers):
            if layer.name == 'Snakes (current edit)':
                self.viewer.layers.pop(idx)

        self.viewer.text_overlay.visible = False
    
    def _on_mouse_click(self, source_layer, e):
        if self.is_in_3d_view:
            return
        
        if self.labels_layer is None:
            return
        
        if self.labels_layer.data.sum() == 0:
            return
        
        self.result_layer.data[self.labels_layer.data == self.selected_label] = self.selected_label
        self.result_layer.refresh()

        if self.check_label_increment.isChecked():
            self.selected_label += 1

    def _on_cursor_move(self, e):
        if self.is_in_3d_view:
            return
        
        if self.ndim == 2:
            xpos, ypos = np.array(e.value).astype(int)
        elif self.ndim == 3:
            _, xpos, ypos = np.array(e.value).astype(int)[self.axes]
        
        # Check that the cursor is inside the image.
        rx, ry = self.image_data_slice.shape
        if not (0 < xpos < rx) & (0 < ypos < ry):
            return
        
        # Flood fill
        start_pixel_intensity = self.image_data_slice[xpos, ypos]
        min_intensity, max_intensity = self.thresholdSlider.value()
        if not (min_intensity < (start_pixel_intensity / self.intensity_range) < max_intensity):
            return
        
        if self.results_data_slice[xpos, ypos] != 0:
            return
        
        seg_data = self.labels_data_slice.copy()    
        seg_data[seg_data == self.selected_label] = 0

        ff_tolerance = self.ff_tolerance.value() * self.intensity_range

        mask = np.logical_and(
            skimage.segmentation.flood(self.image_data_slice, (xpos, ypos), tolerance=ff_tolerance),
            seg_data == 0
        )

        seg_data[mask] = self.selected_label

        if self.ndim == 2:
            self.labels_layer.data = seg_data
        elif self.ndim == 3:
            self.labels_layer.data.transpose(self.axes)[self.current_step] = seg_data
        
        self.labels_layer.refresh()