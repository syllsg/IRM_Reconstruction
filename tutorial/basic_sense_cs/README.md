## Overview
This script demonstrates the use of the SigPy library for performing SENSE (Sensitivity Encoding) 
reconstruction as well compressed sensing alongside a simple gradient descent method for comparison.

## Features
- Implements SENSE reconstruction using SigPy.
- Handles multi-coil MRI data.
- Provides an example workflow for reconstructing undersampled k-space data.
- Includes compressed sensing reconstruction.
- Compares the results with a simple gradient descent method.

## Description
This script uses the SigPy library and in particular the [apps](https://sigpy.readthedocs.io/en/latest/mri_app.html) : EspiritCalib for	ESPIRiT calibration, SenseRecon for SENSE Reconstruction, L1WaveletRecon for L1 Wavelet regularized reconstruction.
"Two different lines of reconstruction algorithms were mostly used: Reconstruction algorithms based on explicit knowledge of the coil sensitivities such as SENSE [1, 2] and algorithms based on local kernels in k-space, which exploit the learned correlation between multiple channels in neighboring points in k-space, such as GRAPPA and SPIRiT". [link](https://pmc.ncbi.nlm.nih.gov/articles/PMC4142121/)
ESPIRiT [Paper](https://pubmed.ncbi.nlm.nih.gov/23649942/) and 
[Library](https://sigpy.readthedocs.io/en/latest/generated/sigpy.mri.app.EspiritCalib.html#sigpy.mri.app.EspiritCalib) links. 
ESPIRiT is a method that finds the subspace of multi-coil data from a calibration region in k-space using a series of eigen-value decompositions in k-space and image space. Here we also use the "soft" sense idea (Uecker et. al, "ESPIRiT Reconstruction using Soft-SENSE", Proceedings of the ISMRM 2013, pp-127) by using the eigen values to weight the eigen-vectors.

## Requirements
- detailed in the yaml file

## Usage
- Ensure you have the required libraries installed.
- Run the script in an environment with access to the required libraries
  -> having installed jupyter will allow for running each cell individually

