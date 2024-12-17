import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QPushButton,
    QHBoxLayout,
    QLabel,
)
from PyQt5.QtCore import Qt


class SpectraViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Spectra Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        # Matplotlib Figure
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(NavigationToolbar(self.canvas, self))
        self.layout.addWidget(self.canvas)

        # Buttons
        self.button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Spectrum")
        self.load_button.clicked.connect(self.load_spectrum)
        self.button_layout.addWidget(self.load_button)

        self.load_mask_button = QPushButton("Load Mask")
        self.load_mask_button.clicked.connect(self.load_mask)
        self.button_layout.addWidget(self.load_mask_button)

        self.edit_mask_button = QPushButton("Edit Mask")
        self.edit_mask_button.clicked.connect(self.activate_mask_editing)
        self.button_layout.addWidget(self.edit_mask_button)

        self.delete_mask_button = QPushButton("Delete Mask")
        self.delete_mask_button.clicked.connect(self.activate_mask_deletion)
        self.button_layout.addWidget(self.delete_mask_button)

        self.save_mask_button = QPushButton("Save Mask")
        self.save_mask_button.clicked.connect(self.save_mask)
        self.button_layout.addWidget(self.save_mask_button)

        self.layout.addLayout(self.button_layout)

        self.mask_data = None
        self.editing_mode = False
        self.deletion_mode = False
        self.selected_line = None
        self.editing_stage = None

    def load_spectrum(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Spectrum File",
            "",
            "Text Files (*.txt);;All Files (*)",
            options=options,
        )

        if file_path:
            try:
                # Load the spectrum data
                data = np.loadtxt(file_path)
                self.wavelengths = data[:, 0]
                self.intensities = data[:, 1]

                # Plot the spectrum
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()
                self.ax.clear()
                self.ax.plot(self.wavelengths, self.intensities, label="Spectrum")
                self.ax.set_xlabel("Wavelength")
                self.ax.set_ylabel("Intensity")
                self.ax.set_title("Spectrum Viewer")
                self.ax.legend().remove()
                self.ax.set_xlim(
                    xlim
                    if xlim != (0.0, 1.0)
                    else (self.wavelengths[0], self.wavelengths[-1])
                )
                self.ax.set_ylim(
                    ylim
                    if ylim != (0.0, 1.0)
                    else (self.intensities.min(), self.intensities.max())
                )

                self.canvas.draw()
            except Exception as e:
                print(f"Error loading file: {e}")

    def load_mask(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Mask File",
            "",
            "Text Files (*.txt);;All Files (*)",
            options=options,
        )

        if file_path:
            try:
                # Load the mask data
                self.mask_data = np.loadtxt(file_path)
                centers = self.mask_data[:, 0]
                left_edges = self.mask_data[:, 1]
                right_edges = self.mask_data[:, 2]

                # Plot the mask
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()
                self.ax.clear()
                self.ax.plot(self.wavelengths, self.intensities, label="Spectrum")

                for center, left, right in zip(centers, left_edges, right_edges):
                    self.ax.axvline(center, color="red", linestyle="--")
                    self.ax.axvspan(left, right, color="blue", alpha=0.3)

                self.ax.set_xlim(xlim)
                self.ax.set_ylim(ylim)
                self.canvas.draw()
            except Exception as e:
                print(f"Error loading mask file: {e}")

    def activate_mask_editing(self):
        if self.mask_data is None:
            print("No mask data loaded.")
            return

        self.editing_mode = True
        self.deletion_mode = False
        self.selected_line = None
        self.editing_stage = "center"
        self.canvas.mpl_connect("button_press_event", self.on_click)
        print("Editing mode activated. Click near a line to edit it.")

    def activate_mask_deletion(self):
        if self.mask_data is None:
            print("No mask data loaded.")
            return

        self.deletion_mode = True
        self.editing_mode = False
        self.selected_line = None
        self.canvas.mpl_connect("button_press_event", self.on_click_delete)
        print("Deletion mode activated. Click near a line to delete it.")

    def on_click(self, event):
        if not self.editing_mode or self.mask_data is None:
            return

        if self.selected_line is None:
            # Find the closest line to the click position
            centers = self.mask_data[:, 0]
            distances = np.abs(centers - event.xdata)
            closest_index = np.argmin(distances)

            if (
                distances[closest_index]
                < (self.wavelengths[-1] - self.wavelengths[0]) * 0.01
            ):  # Threshold: 1% of the spectrum range
                self.selected_line = closest_index
                print(f"Selected line index: {closest_index}")
                print("Now click to set the new center for the line.")
        else:
            self.edit_line(event)

    def on_click_delete(self, event):
        if not self.deletion_mode or self.mask_data is None:
            return

        # Find the closest line to the click position
        centers = self.mask_data[:, 0]
        distances = np.abs(centers - event.xdata)
        closest_index = np.argmin(distances)

        if (
            distances[closest_index]
            < (self.wavelengths[-1] - self.wavelengths[0]) * 0.01
        ):  # Threshold: 1% of the spectrum range
            # Delete the selected line
            self.mask_data = np.delete(self.mask_data, closest_index, axis=0)
            print(f"Deleted line index: {closest_index}")
            self.deletion_mode = False

            # Replot the mask
            self.replot_mask()

    def edit_line(self, event):
        if self.selected_line is None or event.xdata is None:
            return

        line = self.selected_line

        if self.editing_stage == "center":
            self.mask_data[line, 0] = event.xdata
            print(f"New center for line {line}: {event.xdata}")
            self.editing_stage = "left"
            print("Click to set the left edge.")
        elif self.editing_stage == "left":
            self.mask_data[line, 1] = event.xdata
            print(f"New left edge for line {line}: {event.xdata}")
            self.editing_stage = "right"
            print("Click to set the right edge.")
        elif self.editing_stage == "right":
            self.mask_data[line, 2] = event.xdata
            print(f"New right edge for line {line}: {event.xdata}")

            # Mark the line as edited by adding an additional column if not already present
            if self.mask_data.shape[1] < 4:
                self.mask_data = np.hstack(
                    (self.mask_data, np.zeros((self.mask_data.shape[0], 1)))
                )
            self.mask_data[line, 3] = 1  # Mark as edited

            # Finish editing
            self.editing_mode = False
            self.selected_line = None
            self.editing_stage = None
            print("Editing complete. Replotting mask.")
            self.replot_mask()

    def replot_mask(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self.ax.clear()
        self.ax.plot(self.wavelengths, self.intensities, label="Spectrum")

        centers = self.mask_data[:, 0]
        left_edges = self.mask_data[:, 1]
        right_edges = self.mask_data[:, 2]

        for i, (center, left, right) in enumerate(
            zip(centers, left_edges, right_edges)
        ):
            color = (
                "green"
                if self.mask_data.shape[1] > 3 and self.mask_data[i, 3] == 1
                else "blue"
            )
            self.ax.axvline(center, color="red", linestyle="--")
            self.ax.axvspan(left, right, color=color, alpha=0.3)

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.canvas.draw()

    def save_mask(self):
        if self.mask_data is None:
            print("No mask data to save.")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mask File",
            "",
            "Text Files (*.txt);;All Files (*)",
            options=options,
        )

        if file_path:
            try:
                np.savetxt(file_path, self.mask_data[:, :3], fmt="%f")
                print(f"Mask data saved to {file_path}")
            except Exception as e:
                print(f"Error saving mask file: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = SpectraViewer()
    viewer.show()
    sys.exit(app.exec_())
