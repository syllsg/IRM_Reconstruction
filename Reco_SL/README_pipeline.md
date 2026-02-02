# MRI Reconstruction Pipeline

Automated pipeline for k-space undersampling, reconstruction, and evaluation with publication-quality figures.

## Features

- **Multiple Undersampling Patterns**: Cartesian, spiral, radial, random
- **Various Reconstruction Methods**: RSS, sensitivity-based, SENSE, gradient descent, regularization, deep learning
- **Automatic Metrics Calculation**: MSE, NRMSE, PSNR, SSIM
- **Publication-Quality Figures**: Automatically generated comparison figures
- **Efficient Data Storage**: HDF5 format with compression

## Installation

```bash
pip install numpy matplotlib h5py scikit-image
```

For advanced reconstructions, also install:
```bash
pip install sigpy  # For SENSE reconstructions
```

## Quick Start

### 1. Basic Usage

```python
from reconstruction_pipeline import ReconstructionPipeline, create_config_template
from figure_generator import ReconstructionFigureGenerator

# Initialize pipeline with save format option
# Options: 'numpy', 'hdf5', or 'both' (default)
pipeline = ReconstructionPipeline(output_dir="experiments", save_format="numpy")

# Load your k-space data
kspace = np.load('your_kspace.npy')  # Shape: [coils, kx, ky]
sensitivity_maps = np.load('your_smaps.npy')  # Optional

# Create configuration
config = create_config_template(
    acceleration_factor=4,
    mask_type='cartesian',
    reconstruction_methods=['rss', 'sensitivity_based', 'sense_sigpy']
)

# Run experiment
results = pipeline.run_experiment(
    kspace=kspace,
    config=config,
    experiment_name="my_experiment",
    sensitivity_maps=sensitivity_maps
)

# Generate figure
fig_gen = ReconstructionFigureGenerator()
fig_gen.create_comparison_figure(results, "output_figure.png")
```

### Save Format Options

The pipeline supports three save formats:

**Numpy arrays only (recommended for easy access):**
```python
pipeline = ReconstructionPipeline(output_dir="experiments", save_format="numpy")
```
- Saves all images as `.npy` files
- Complex images saved as both `_magnitude.npy` and `_complex.npy`
- Metrics saved as `metrics.json`
- Easy to load: `img = np.load('path/to/rss.npy')`

**HDF5 only (recommended for large datasets):**
```python
pipeline = ReconstructionPipeline(output_dir="experiments", save_format="hdf5")
```
- Single compressed file per experiment
- Efficient for large data
- Load with: `h5py.File('experiment.h5', 'r')`

**Both formats (default):**
```python
pipeline = ReconstructionPipeline(output_dir="experiments", save_format="both")
```
- Maximum flexibility
- Numpy for quick access, HDF5 for archival

### 2. Configuration Options

#### Undersampling Masks

**Cartesian:**
```python
config = create_config_template(
    acceleration_factor=4,
    mask_type='cartesian',
    center_fraction=0.08  # Fraction of center k-space to fully sample
)
```

**Spiral:**
```python
config = create_config_template(
    acceleration_factor=4,
    mask_type='spiral',
    num_spirals=3,
    spiral_type='archimedean'  # or 'fermat'
)
```

**Radial:**
```python
config = create_config_template(
    acceleration_factor=4,
    mask_type='radial'
)
```

**Random:**
```python
config = create_config_template(
    acceleration_factor=4,
    mask_type='random',
    center_fraction=0.08
)
```

#### Reconstruction Methods

Available methods (specify in `reconstruction_methods` list):
- `'rss'`: Root Sum of Squares
- `'sensitivity_based'`: Sensitivity-based coil combination
- `'sense_sigpy'`: SigPy SENSE reconstruction
- `'gradient_descent'`: Iterative gradient descent
- `'gd_l1_wavelet'`: Gradient descent with L1 wavelet regularization
- `'unet'`: U-Net deep learning reconstruction
- `'unrolled'`: Unrolled network reconstruction

### 3. Adding Custom Reconstruction Methods

```python
# Define your custom reconstruction
def my_reconstruction(masked_kspace, mask, sensitivity_maps):
    # Your implementation here
    img_coils = np.fft.ifft2(masked_kspace, axes=(-2, -1))
    # ... processing ...
    return reconstructed_image

# Register it
pipeline.reconstruction_methods['my_method'] = my_reconstruction

# Use it in config
config = create_config_template(
    reconstruction_methods=['rss', 'my_method']
)
```

### 4. Running Multiple Experiments

```python
# Compare different acceleration factors
for acceleration in [2, 4, 8]:
    config = create_config_template(
        acceleration_factor=acceleration,
        mask_type='cartesian'
    )
    
    results = pipeline.run_experiment(
        kspace=kspace,
        config=config,
        experiment_name=f"accel_{acceleration}x"
    )

# Create summary comparison
fig_gen = ReconstructionFigureGenerator()
fig_gen.create_summary_figure(
    'experiments/data',
    'experiments/figures/summary.png',
    metric='PSNR'
)
```

### 5. Loading and Visualizing Saved Results

**Loading HDF5 files:**
```python
from figure_generator import load_and_visualize_results

# Load experiment and create visualization
results = load_and_visualize_results(
    'experiments/data/my_experiment.h5',
    output_dir='experiments/figures'
)

# Access loaded data
reconstructions = results['reconstructions']
metrics = results['metrics']
ground_truth = results['ground_truth']
```

**Loading numpy arrays:**
```python
import numpy as np
import json

# Load reconstructions
rss_img = np.load('experiments/data/experiment1_numpy/reconstructions/rss.npy')
sens_img = np.load('experiments/data/experiment1_numpy/reconstructions/sensitivity_based.npy')

# For complex images
rss_complex = np.load('experiments/data/experiment1_numpy/reconstructions/rss_complex.npy')
rss_magnitude = np.load('experiments/data/experiment1_numpy/reconstructions/rss_magnitude.npy')

# Load ground truth
ground_truth = np.load('experiments/data/experiment1_numpy/ground_truth.npy')

# Load mask
mask = np.load('experiments/data/experiment1_numpy/mask.npy')

# Load metrics
with open('experiments/data/experiment1_numpy/metrics.json', 'r') as f:
    metrics = json.load(f)

# Load config
with open('experiments/data/experiment1_numpy/config.json', 'r') as f:
    config = json.load(f)
```

## Output Structure

### When using `save_format="numpy"` or `save_format="both"`:

```
experiments/
├── data/
│   ├── experiment1.h5                    # (if save_format includes 'hdf5')
│   ├── experiment1_numpy/                # Numpy arrays directory
│   │   ├── mask.npy
│   │   ├── ground_truth.npy
│   │   ├── kspace.npy
│   │   ├── masked_kspace.npy
│   │   ├── reconstructions/
│   │   │   ├── rss.npy                   # or rss_magnitude.npy + rss_complex.npy
│   │   │   ├── sensitivity_based.npy
│   │   │   └── ...
│   │   ├── metrics.json
│   │   └── config.json
│   └── ...
├── figures/
│   ├── experiment1_comparison.png
│   └── ...
└── configs/
    ├── experiment1_config.json
    └── ...
```

### When using `save_format="hdf5"`:

```
experiments/
├── data/                      # HDF5 files with all data
│   ├── experiment1.h5
│   ├── experiment2.h5
│   └── ...
├── figures/                   # Generated figures
│   ├── experiment1_comparison.png
│   ├── experiment2_comparison.png
│   └── summary.png
└── configs/                   # JSON configuration files
    ├── experiment1_config.json
    ├── experiment2_config.json
    └── ...
```

## HDF5 File Structure

Each experiment is saved in an HDF5 file with the following structure:

```
experiment.h5
├── kspace                     # Fully sampled k-space
├── masked_kspace              # Undersampled k-space
├── mask                       # Undersampling mask
├── ground_truth               # Ground truth image
├── reconstructions/           # Reconstructed images
│   ├── rss
│   ├── sensitivity_based
│   └── ...
└── metrics/                   # Metrics for each method
    ├── rss/
    │   ├── MSE (attribute)
    │   ├── NRMSE (attribute)
    │   ├── PSNR (attribute)
    │   └── SSIM (attribute)
    └── ...
```

## File Descriptions

- **`reconstruction_pipeline.py`**: Main pipeline orchestrator
- **`figure_generator.py`**: Figure generation module
- **`run_metrics.py`**: Metrics calculation utilities
- **`example_usage.py`**: Example scripts demonstrating usage
- **`README.md`**: This documentation file

## Metrics

The pipeline automatically calculates four metrics for each reconstruction:

1. **MSE** (Mean Squared Error): Lower is better
2. **NRMSE** (Normalized Root Mean Squared Error): Lower is better
3. **PSNR** (Peak Signal-to-Noise Ratio): Higher is better (dB)
4. **SSIM** (Structural Similarity Index): Higher is better (0-1)

## Tips

1. **Complex Images**: The pipeline automatically handles complex-valued images by converting to magnitude
2. **Memory**: For large datasets, consider processing slices individually
3. **Parallel Processing**: You can parallelize multiple experiments using multiprocessing
4. **Custom Metrics**: Extend the pipeline to add your own metrics in `_calculate_all_metrics()`

## Troubleshooting

**Issue: "NotImplementedError" for certain reconstruction methods**
- Solution: Implement the method in the pipeline class or use available methods

**Issue: Out of memory**
- Solution: Process fewer slices at once or reduce image size

**Issue: Figures look different from screenshot**
- Solution: Adjust `figsize`, `vmin`, `vmax` parameters in `create_comparison_figure()`

## Examples

See `example_usage.py` for complete working examples:
- Example 1: Basic single experiment
- Example 2: Multiple acceleration factors
- Example 3: Different mask types
- Example 4: Loading and visualizing saved results
- Example 5: Custom reconstruction workflow

## Contributing

To add new reconstruction methods:
1. Implement method following the signature: `(masked_kspace, mask, sensitivity_maps) -> reconstructed_image`
2. Add to `reconstruction_methods` dictionary in pipeline
3. Update method display names in `figure_generator.py`

## Citation

If you use this pipeline in your research, please cite appropriately.
