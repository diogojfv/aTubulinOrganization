import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QMenuBar, QAction, QFileDialog, QTabWidget, QVBoxLayout, QWidget, QGridLayout, QLabel, QSlider, QPushButton, QCheckBox, QRadioButton, QGroupBox, QHBoxLayout, QComboBox, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
import pandas as pd

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Processing GUI")
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { border: 1px solid black; }")
        self.image_label.setFixedSize(400, 400)

        self.tab_widget = QTabWidget()
        self.preprocess_tab = QWidget()
        self.processing_tab = QWidget()
        self.results_tab = QWidget()

        self.init_menu()
        self.init_preprocess_tab()
        self.init_processing_tab()
        self.init_results_tab()

        self.tab_widget.addTab(self.preprocess_tab, "Preprocessing")
        self.tab_widget.addTab(self.processing_tab, "Processing")
        self.tab_widget.addTab(self.results_tab, "Results")

        self.setCentralWidget(self.tab_widget)

    def init_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open Images from Folder", self)
        open_action.triggered.connect(self.open_images)
        file_menu.addAction(open_action)

        save_action = QAction("Save Image", self)
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)

    def init_preprocess_tab(self):
        layout = QVBoxLayout()
        self.preprocess_tab.setLayout(layout)

        filter_combo = QComboBox()
        filter_combo.addItems(["Gaussian", "Hessian", "Sato", "Skeletonization"])
        layout.addWidget(filter_combo)

        filter_slider = QSlider(Qt.Horizontal)
        layout.addWidget(filter_slider)

        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_filter)
        layout.addWidget(apply_button)

        self.preprocess_images_layout = QHBoxLayout()
        layout.addLayout(self.preprocess_images_layout)

    def init_processing_tab(self):
        layout = QVBoxLayout()
        self.processing_tab.setLayout(layout)

        self.tab_widget_processing = QTabWidget()
        layout.addWidget(self.tab_widget_processing)

        dcf_tab = QWidget()
        lsf_tab = QWidget()
        cnf_tab = QWidget()

        self.tab_widget_processing.addTab(dcf_tab, "DCF")
        self.tab_widget_processing.addTab(lsf_tab, "LSF")
        self.tab_widget_processing.addTab(cnf_tab, "CNF")

        dcf_layout = QVBoxLayout()
        dcf_tab.setLayout(dcf_layout)

        lsf_layout = QVBoxLayout()
        lsf_tab.setLayout(lsf_layout)

        cnf_layout = QVBoxLayout()
        cnf_tab.setLayout(cnf_layout)

        dcf_checkboxes = QGroupBox("DCF Options")
        dcf_checkboxes_layout = QVBoxLayout()
        dcf_checkboxes.setLayout(dcf_checkboxes_layout)

        for i in range(6):
            checkbox = QCheckBox(f"Option {i+1}")
            dcf_checkboxes_layout.addWidget(checkbox)

        dcf_layout.addWidget(dcf_checkboxes)

        lsf_checkboxes = QGroupBox("LSF Options")
        lsf_checkboxes_layout = QVBoxLayout()
        lsf_checkboxes.setLayout(lsf_checkboxes_layout)

        for i in range(6):
            checkbox = QCheckBox(f"Option {i+1}")
            lsf_checkboxes_layout.addWidget(checkbox)

        lsf_sliders_layout = QHBoxLayout()
        lsf_layout.addLayout(lsf_sliders_layout)

        for i in range(3):
            slider = QSlider(Qt.Horizontal)
            lsf_sliders_layout.addWidget(slider)

        lsf_layout.addWidget(lsf_checkboxes)

        cnf_checkboxes = QGroupBox("CNF Options")
        cnf_checkboxes_layout = QVBoxLayout()
        cnf_checkboxes.setLayout(cnf_checkboxes_layout)

        for i in range(6):
            checkbox = QCheckBox(f"Option {i+1}")
            cnf_checkboxes_layout.addWidget(checkbox)

        cnf_layout.addWidget(cnf_checkboxes)

    def init_results_tab(self):
        layout = QVBoxLayout()
        self.results_tab.setLayout(layout)

        add_button = QPushButton("Calculate")
        add_button.clicked.connect(self.add_to_results)
        layout.addWidget(add_button)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["ROI", "Value"])
        layout.addWidget(self.results_table)

        plot_layout = QHBoxLayout()
        layout.addLayout(plot_layout)

        self.plot_radiobutton1 = QRadioButton("Plot 1")
        plot_layout.addWidget(self.plot_radiobutton1)

        self.plot_radiobutton2 = QRadioButton("Plot 2")
        plot_layout.addWidget(self.plot_radiobutton2)

        self.plot_radiobutton3 = QRadioButton("Plot 3")
        plot_layout.addWidget(self.plot_radiobutton3)

    def open_images(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.Directory)
        folder_path = file_dialog.getExistingDirectory(self, "Select Folder")
        # Code to open .tif and .png images from the folder_path
        # Display the image on the self.image_label using QPixmap

    def save_image(self):
        file_dialog = QFileDialog()
        file_dialog.setDefaultSuffix(".png")
        file_path, _ = file_dialog.getSaveFileName(self, "Save Image", "", "PNG Image (*.png)")
        # Code to save the current image displayed on self.image_label to file_path

    def apply_filter(self):
        filter_type = self.preprocess_tab.findChild(QComboBox).currentText()
        filter_value = self.preprocess_tab.findChild(QSlider).value()
        # Code to apply the selected filter to the image and obtain the filtered image

        # Create a miniature square box with the filtered image
        miniature_label = QLabel()
        miniature_label.setAlignment(Qt.AlignCenter)
        miniature_label.setFixedSize(100, 100)
        miniature_label.setPixmap(QPixmap.fromImage(filtered_image).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio))
        self.preprocess_images_layout.addWidget(miniature_label)

    def add_to_results(self):
        # Code to add a new row to the results_table with ROI and value
        row_count = self.results_table.rowCount()
        self.results_table.insertRow(row_count)
        self.results_table.setItem(row_count, 0, QTableWidgetItem("ROI"))
        self.results_table.setItem(row_count, 1, QTableWidgetItem("Value"))

    def plot_results(self):
        # Code to plot the selected plot based on the radio button selection
        pass

    def show_global_results(self):
        # Code to collapse the main image rectangle window and show barplots of every feature of every checkbox that was checked on the processing tab
        pass

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
