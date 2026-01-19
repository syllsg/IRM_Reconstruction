## Overview
This repo aims to discover and explore IRM especially from parallel imaging. 
It shows you how to open kspace data, to produce images from k-space and understand coil sensitivity. 

It also demonstrates how to undersample a kspace and then explore reconstruction methods. It uses the SigPy library for performing SENSE (Sensitivity Encoding) 
reconstruction as well compressed sensing alongside a simple gradient descent method for comparison.

## Features
- Explore IRM data and Handles multi-coil MRI data
- Implements SENSE reconstruction using SigPy.
- Provides an example workflow for reconstructing undersampled k-space data.
- Compares the results with a simple gradient descent method.

## Description
### IRM data configuration
This script explore IRM data from .h5 file that contains 3 informations : ismrmrd_header (or metadata or informations about the configuration of the IRM acquisition as xml file); kspace data and reconstruction_rss (that contains reconstructed images output from the scanner). To reproduce the results, please download the raw [fastMRI](https://fastmri.med.nyu.edu/) data, the [h5 file](https://www.dropbox.com/scl/fi/9ag9tj6oeci1x6uxaq6s4/file_brain_AXT2_205_2050058.h5?rlkey=xmr8o0xlzffjxsg604yup7vvw&dl=0) and place the file_brain_AXT2_205_2050058.h5 file in the data folder. 

### Visualising and handling multi coil data
These notebooks explore parallel imaging data and visualise multicoil data that are combined into a single image per slice by computing the pixel-wise root-sum-of-squares (RSS) across the coil dimension.
Coil sensitivity maps are created with  [EspiritCalib](https://sigpy.readthedocs.io/en/latest/generated/sigpy.mri.app.EspiritCalib.html#sigpy.mri.app.EspiritCalib) for	ESPIRiT calibration from the SigPy library.
ESPIRiT [Paper](https://pubmed.ncbi.nlm.nih.gov/23649942/) and 
[Library](https://sigpy.readthedocs.io/en/latest/generated/sigpy.mri.app.EspiritCalib.html#sigpy.mri.app.EspiritCalib) links. 
ESPIRiT is a method that finds the subspace of multi-coil data from a calibration region in k-space using a series of eigen-value decompositions in k-space and image space. Here we also use the "soft" sense idea (Uecker et. al, "ESPIRiT Reconstruction using Soft-SENSE", Proceedings of the ISMRM 2013, pp-127) by using the eigen values to weight the eigen-vectors.

### Reconstruction methods
These notebooks use the SigPy library and in particular the [apps](https://sigpy.readthedocs.io/en/latest/mri_app.html) : EspiritCalib for	ESPIRiT calibration, SenseRecon for SENSE Reconstruction, L1WaveletRecon for L1 Wavelet regularized reconstruction.
"Two different lines of reconstruction algorithms were mostly used: Reconstruction algorithms based on explicit knowledge of the coil sensitivities such as SENSE [1, 2] and algorithms based on local kernels in k-space, which exploit the learned correlation between multiple channels in neighboring points in k-space, such as GRAPPA and SPIRiT". [link](https://pmc.ncbi.nlm.nih.gov/articles/PMC4142121/)


## Requirements and installation
Dependencies detailed in the pyproject.toml file : <br>
[Python](https://www.python.org/downloads/release/python-3923/) 3.9 <br>
[UV](https://docs.astral.sh/uv/) as package manager  <br>
Instal UV, clone the repo and then install dependencies and activate the virtual env with : 
```bash
uv sync
```


## References and Acknowledgments
These notebooks have been inspired from the 1st part of this [tutorial](https://github.com/tutorial-MRI-recon/tutorial/tree/main) that is accompanying the review paper: "A Tutorial on MRI Reconstruction: From Modern Methods to Clinical Implications".