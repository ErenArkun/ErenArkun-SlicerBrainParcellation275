# SlicerBrainParcellation275

This repository contains the code for two Slicer modules that can be used to segment brain structures on T1-weighted MRIs.

Segmentations are performed using convolutional neural networks (CNNs), i.e., deep learning models. They take less than a minute on a graphics processing unit (GPU).

### Add directory in Slicer

In Slicer, go to `Edit -> Application Settings -> Modules` and add the cloned/downloaded folder to the `Additional module paths`. When prompted, restart Slicer.

## Model Setup

To set up the deep learning models required for brain segmentation, please follow these steps:

1. [Click here](https://drive.google.com/file/d/1rslTenl_TutBWi7uIlXJXrimZB3RgtxE/view?usp=sharing) to download the `MODEL_FOLDER.zip` file from Google Drive.
2. Extract the downloaded `MODEL_FOLDER.zip` file.
3. Place the extracted `MODEL_FOLDER` directory into the `BrainSegmentation` folder in this repository. This `MODEL_FOLDER` contains the deep learning models necessary for segmentation.

After completing these steps, the model files will be set up, and you’ll be ready to run the brain segmentation module.


## Requirements

This project uses deep learning models built with **PyTorch**. To perform computations on a GPU, please ensure you have the following dependencies installed:

- **CUDA 11.8** 
- **cuDNN 8.9.7**

These libraries are necessary to enable GPU acceleration, which significantly speeds up the segmentation process.

For CUDA and cuDNN installation instructions, refer to the official documentation:

- [CUDA 11.8 Installation Guide](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- [cuDNN 8.9.7 Installation Guide](https://developer.nvidia.com/rdp/cudnn-archive)

**Note**: When installing cuDNN, be sure to select the version compatible with **CUDA 11**.

Make sure your environment is configured to use these libraries to take full advantage of GPU support.



## PyTorch with CUDA Installation for 3D Slicer

To enable GPU acceleration using PyTorch in 3D Slicer, you need to install PyTorch with CUDA support. Follow these steps to install the required libraries:

### 1. Install CUDA and cuDNN

Before installing PyTorch, ensure that **CUDA 11.8** and **cuDNN 8.9.7** are properly installed on your system. These libraries are required for GPU acceleration in PyTorch.

- **CUDA 11.8** Installation:  
  Follow the official installation guide here:  
  [CUDA 11.8 Installation Guide](https://developer.nvidia.com/cuda-11-8-0-download-archive)

- **cuDNN 8.9.7** Installation:  
  For **CUDA 11.8**, download the corresponding version of cuDNN here:  
  [cuDNN 8.9.7 Installation Guide](https://developer.nvidia.com/rdp/cudnn-archive)

Follow the instructions to set up these libraries on your system.

### 2. Install PyTorch with CUDA Support

After installing CUDA and cuDNN, you can install PyTorch with CUDA support in your 3D Slicer environment. Use the following steps:

1. **Open 3D Slicer** and go to the Python Interactor or use the command line.

2. **Install PyTorch with CUDA** by running the following command in the Python Interactor:

   ```python
   slicer.util.pip_install('torch torchvision torchaudio')
