import os
import numpy as np
import nibabel as nib
from nibabel import processing
import PyTorchUtils
torch = PyTorchUtils.PyTorchUtilsLogic().torch
import glob

import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleWidget, ScriptedLoadableModuleLogic
import qt
from tqdm import tqdm as std_tqdm
from functools import partial
from utils.cropping import cropping
from utils.hemisphere import hemisphere
from utils.load_model import load_model
from utils.make_csv import make_csv
from utils.parcellation import parcellation
from utils.postprocessing import postprocessing
from utils.preprocessing import preprocessing
from utils.stripping import stripping

import argparse

from utils.update_segment_name import update_segment_names

# This wraps tqdm for progress bar
tqdm = partial(std_tqdm, dynamic_ncols=True)

def create_parser(args_list):
    parser = argparse.ArgumentParser(
        description="Use this to run inference with OpenMAP-T1."
    )
    parser.add_argument(
        "-i",
        required=True,
        help="Input folder. Specifies the folder containing the input brain MRI images.",
    )
    parser.add_argument(
        "-o",
        required=True,
        help="Output folder. Difines the output folder where the results will be saved. If the specified folder does not exist, it will be automatically created.",
    )
    parser.add_argument(
        "-m",
        required=True,
        help="Folder of pretrained models. Indicates the location of the pretrained models to be used for processing.",
    )
    return parser.parse_args(args_list)

class BrainSegmentation(ScriptedLoadableModule):
    """Define the module metadata."""
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Brain Segmentation"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Your Name (Your Institution)"]
        self.parent.helpText = """This module performs brain segmentation."""
        self.parent.acknowledgementText = """This module was developed with the support of XYZ."""

class BrainSegmentationWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.layout = qt.QFormLayout()

        self.inputSelectorButton = qt.QPushButton("Select Input Folder")
        self.inputSelectorButton.toolTip = "Select the input folder containing NIfTI files"
        self.inputSelectorButton.clicked.connect(self.onSelectInputFolder)
        self.layout.addRow(self.inputSelectorButton)
        self.inputFileLabel = qt.QLabel("No folder selected")
        self.layout.addRow(self.inputFileLabel)

        self.outputSelectorButton = qt.QPushButton("Select Output Folder")
        self.outputSelectorButton.toolTip = "Select the folder for saving the output"
        self.outputSelectorButton.clicked.connect(self.onSelectOutputFolder)
        self.layout.addRow(self.outputSelectorButton)
        self.outputFolderLabel = qt.QLabel("No folder selected")
        self.layout.addRow(self.outputFolderLabel)

        self.applyButton = qt.QPushButton("Apply Segmentation")
        self.applyButton.toolTip = "Apply brain segmentation"
        self.applyButton.clicked.connect(self.onApplySegmentation)
        self.layout.addRow(self.applyButton)

        self.subfolderComboBox = qt.QComboBox()
        self.subfolderComboBox.setToolTip("Select a subfolder from the output directory")
        self.subfolderComboBox.currentIndexChanged.connect(self.onSubfolderSelected)
        self.layout.addRow(qt.QLabel("Select Subfolder:"))
        self.layout.addRow(self.subfolderComboBox)

        self.niiFileComboBox = qt.QComboBox()
        self.niiFileComboBox.setToolTip("Select a .nii file")
        self.niiFileComboBox.currentIndexChanged.connect(self.onNiiFileSelected)
        self.layout.addRow(qt.QLabel("Select .nii file:"))
        self.layout.addRow(self.niiFileComboBox)

        self.layout.addRow(qt.QLabel(" "))

        self.inputFile = None
        self.outputFolder = None

        self.currentVolumeNode = None
        self.currentSegmentationNode = None

        self.parent.layout().addLayout(self.layout)

    def onSelectInputFolder(self):
        folderDialog = qt.QFileDialog()
        folderDialog.setFileMode(qt.QFileDialog.Directory)
        folderDialog.setOption(qt.QFileDialog.ShowDirsOnly, True)

        if folderDialog.exec_():
            selectedFolder = folderDialog.selectedFiles()
            if selectedFolder:
                self.inputFile = selectedFolder[0]
                print(f"Selected folder: {self.inputFile}")
            else:
                print("No folder selected.")
        else:
            print("Folder dialog was cancelled.")

        if self.inputFile and os.path.isdir(self.inputFile):
            self.inputFileLabel.setText(f"Selected input folder: {os.path.basename(self.inputFile)}")
        else:
            slicer.util.errorDisplay("Invalid folder selected or folder does not exist.")

    def onNiiFileSelected(self):
        selectedNiiFile = self.niiFileComboBox.currentText
        if selectedNiiFile:
            subfolderPath = self.subfolderComboBox.currentText
            fullPath = os.path.join(self.outputFolder, subfolderPath, selectedNiiFile)
            self.loadVolume(fullPath)


    def loadVolume(self, inputFile):
        try:
            self.clearPreviousNodes()

            if self.isSegmentation(inputFile):
                segmentationNode = slicer.util.loadSegmentation(inputFile)
                if segmentationNode:
                    self.currentSegmentationNode = segmentationNode
                    self.setupDisplayNodeFor3D(segmentationNode, "Segmentation successfully displayed in the 3D view.")
                else:
                    print("Segmentation yüklenemedi.")

            else:
                volumeNode = slicer.util.loadVolume(inputFile)
                if volumeNode:
                    self.currentVolumeNode = volumeNode
                    slicer.util.setSliceViewerLayers(background=volumeNode, foreground=None)
                    self.setupDisplayNodeFor3D(volumeNode, "Volume successfully displayed in the 3D view.")
                else:
                    print("Volume yüklenemedi.")
        except Exception as e:
            print(f"Error loading the file: {e}")

    def setupDisplayNodeFor3D(self, node, successMessage):
        displayNode = node.GetDisplayNode()
        if displayNode and slicer.app.layoutManager().threeDViewCount > 0:
            displayNode.SetVisibility3D(True)
            layoutManager = slicer.app.layoutManager()
            threeDWidget = layoutManager.threeDWidget(0)
            threeDView = threeDWidget.threeDView()
            threeDView.resetFocalPoint()

            print(successMessage)
        else:
            print("DisplayNode not found or 3D view not available.")

    def clearPreviousNodes(self):
        slicer.mrmlScene.Clear(0)
        self.currentVolumeNode = None
        self.currentSegmentationNode = None

    def isSegmentation(self, inputFile):
        return any(x in inputFile.lower() for x in ["_280", ".seg.nrrd"])


    def onSelectOutputFolder(self):
        folderDialog = qt.QFileDialog()
        self.outputFolder = folderDialog.getExistingDirectory(None, "Select Output Folder")
        if self.outputFolder:
            self.outputFolderLabel.setText(f"Selected output folder: {self.outputFolder}")
            self.subfolderComboBox.clear()
            subfolders = [f for f in os.listdir(self.outputFolder) if os.path.isdir(os.path.join(self.outputFolder, f))]

            if subfolders:
                self.subfolderComboBox.addItems(subfolders)
            else:
                self.subfolderComboBox.addItem("No subfolders found")
        else:
            slicer.util.errorDisplay("No folder selected.")

    def onSubfolderSelected(self):
        self.niiFileComboBox.clear()
        selectedSubfolder = self.subfolderComboBox.currentText
        if not any(os.path.isdir(os.path.join(self.outputFolder, d)) for d in os.listdir(self.outputFolder)):
                print(f"No subfolders found in the output folder: {self.outputFolder}")
                return

        if selectedSubfolder:
            subfolderPath = os.path.join(self.outputFolder, selectedSubfolder)

            niiFiles = [f for f in os.listdir(subfolderPath) if f.endswith('.nii') or f.endswith('.seg.nrrd')]

            if niiFiles:
                self.niiFileComboBox.addItems(niiFiles)
            else:
                self.niiFileComboBox.addItem("No .nii files found")

    def updateComboBoxes(self):
        if not self.outputFolder:
            return

        subfolders = [f for f in os.listdir(self.outputFolder) if os.path.isdir(os.path.join(self.outputFolder, f))]
        self.subfolderComboBox.clear()
        self.subfolderComboBox.addItems(subfolders)
        self.niiFileComboBox.clear()
        selectedSubfolder = self.subfolderComboBox.currentText

        if not any(os.path.isdir(os.path.join(self.outputFolder, d)) for d in os.listdir(self.outputFolder)):
            print(f"No subfolders found in the output folder: {self.outputFolder}")
            return

        if selectedSubfolder:
            subfolderPath = os.path.join(self.outputFolder, selectedSubfolder)
            niiFiles = [f for f in os.listdir(subfolderPath) if f.endswith('.nii') or f.endswith(".seg.nrrd")]

            if niiFiles:
                self.niiFileComboBox.addItems(niiFiles)
            else:
                self.niiFileComboBox.addItem("No .nii files found")

    def onApplySegmentation(self):
        if not self.inputFile or not self.outputFolder:
            slicer.util.errorDisplay("Please select an input folder and output folder.")
            return

        logic = BrainSegmentationLogic()
        logic.run(self.inputFile, self.outputFolder)

        update_segment_names(self.outputFolder)
        self.updateComboBoxes()
        slicer.util.infoDisplay("Segmentation completed successfully!")

class BrainSegmentationLogic:
    def run(self, inputFile, outputFolder):
        print(f"Running segmentation with input: {inputFile} and output: {outputFolder}")


        print(f"Input file: {inputFile}")
        print(f"Output folder: {outputFolder}")

        script_dir = os.path.dirname(os.path.realpath(__file__))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_folder= os.path.join(script_dir, "MODEL_FOLDER")
        args_list = ['-i', inputFile, '-o', outputFolder, '-m', model_folder]
        opt = create_parser(args_list)
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        cnet, ssnet, pnet_c, pnet_s, pnet_a, hnet_c, hnet_a = load_model(opt, device)

        print("load complete !!")
        pathes = sorted(glob.glob(os.path.join(opt.i, "**/*.nii"), recursive=True))

        for path in tqdm(pathes):
            save = os.path.splitext(os.path.basename(path))[0]
            output_dir = f"{opt.o}/{save}"
            os.makedirs(output_dir, exist_ok=True)

            odata = nib.squeeze_image(nib.as_closest_canonical(nib.load(path)))
            nii = nib.Nifti1Image(odata.get_fdata().astype(np.float32), affine=odata.affine)
            nib.save(nii, os.path.join(output_dir, f"{save}.nii"))

            odata, data = preprocessing(path, save)
            cropped = cropping(data, cnet, device)
            stripped, shift = stripping(cropped, data, ssnet, device)
            parcellated = parcellation(stripped, pnet_c, pnet_s, pnet_a, device)
            separated = hemisphere(stripped, hnet_c, hnet_a, device)
            output = postprocessing(parcellated, separated, shift, device)

            df = make_csv(output, save)
            df.to_csv(os.path.join(output_dir, f"{save}_volume.csv"), index=False)

            nii = nib.Nifti1Image(output.astype(np.uint16), affine=data.affine)
            header = odata.header
            nii = processing.conform(
                nii,
                out_shape=(header["dim"][1], header["dim"][2], header["dim"][3]),
                voxel_size=(header["pixdim"][1], header["pixdim"][2], header["pixdim"][3]),
                order=0,
            )

            output_path = os.path.join(output_dir, f"{save}_280.nii")
            nib.save(nii, output_path)

            del odata, data
            os.remove(f"N4/{save}.nii")
