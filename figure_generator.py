"""
Figure Generator for MRI Reconstruction Results
Creates publication-quality figures matching the layout in the screenshot
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import patches
import h5py
from pathlib import Path
from typing import Dict, Optional, List


class ReconstructionFigureGenerator:
    """
    Generate figures for reconstruction results
    """
    
    def __init__(self, fontsize: int = 10):
        """
        Initialize figure generator
        
        Parameters:
        -----------
        fontsize : int
            Base font size for figures
        """
        self.fontsize = fontsize
        plt.rcParams['font.size'] = fontsize
        plt.rcParams['font.family'] = 'sans-serif'
        
    def create_comparison_figure(self,
                                 results: Dict,
                                 output_path: str,
                                 slice_idx: Optional[int] = None,
                                 vmin: Optional[float] = None,
                                 vmax: Optional[float] = None,
                                 figsize: tuple = (16, 10)):
        """
        Create a comparison figure like in the screenshot
        
        Parameters:
        -----------
        results : dict
            Results dictionary from pipeline.run_experiment()
        output_path : str
            Path to save the figure
        slice_idx : int, optional
            Slice index if dealing with 3D data
        vmin, vmax : float, optional
            Intensity range for display
        figsize : tuple
            Figure size in inches
        """
        
        reconstructions = results['reconstructions']
        metrics = results['metrics']
        mask = results['mask']
        ground_truth = results['ground_truth']
        
        # Extract 2D slice if needed
        if slice_idx is not None and len(ground_truth.shape) > 2:
            ground_truth = ground_truth[..., slice_idx]
            mask = mask[..., slice_idx] if len(mask.shape) > 2 else mask
            reconstructions = {k: v[..., slice_idx] if v is not None and len(v.shape) > 2 else v 
                             for k, v in reconstructions.items()}
        
        # Determine intensity range
        if vmin is None:
            vmin = np.min(ground_truth)
        if vmax is None:
            vmax = np.max(ground_truth)
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create grid layout
        n_methods = len([v for v in reconstructions.values() if v is not None])
        
        # Main grid: title row, image row, metrics rows
        gs_main = gridspec.GridSpec(6, n_methods + 2, figure=fig,
                                    height_ratios=[0.3, 3, 0.5, 0.5, 0.5, 0.5],
                                    width_ratios=[1.2] + [1] * (n_methods + 1),
                                    hspace=0.05, wspace=0.15,
                                    left=0.05, right=0.95, top=0.92, bottom=0.08)
        
        # Title
        fig.suptitle('Reconstruction Synthesis', fontsize=self.fontsize + 6, 
                    fontweight='bold', y=0.98)
        
        # Column headers
        col_idx = 0
        
        # Ground truth column
        ax_header = fig.add_subplot(gs_main[0, col_idx])
        ax_header.text(0.5, 0.5, 'Full sample\nmulticoil\ncombined image\n(RSS)', 
                      ha='center', va='center', fontsize=self.fontsize,
                      transform=ax_header.transAxes)
        ax_header.axis('off')
        col_idx += 1
        
        # Reconstruction method columns
        method_display_names = {
            'rss': 'RSS',
            'sensitivity_based': 'x conj(sensitivitymaps)\n \nSENSE style',
            'sense_sigpy': 'SENSE sigpy',
            'gradient_descent': 'Gradient descent',
            'gd_l1_wavelet': 'GD + L1 wavelet\nregularization',
            'unet': 'U-Net',
            'unrolled': 'Unrolled\nnetwork'
        }
        
        method_names = [k for k in reconstructions.keys() if reconstructions[k] is not None]
        
        for method_name in method_names:
            ax_header = fig.add_subplot(gs_main[0, col_idx])
            display_name = method_display_names.get(method_name, method_name)
            ax_header.text(0.5, 0.5, f'reco {results.get("acceleration", "4")}x undersampled Multicoil combined image\n \n{display_name}',
                          ha='center', va='center', fontsize=self.fontsize,
                          transform=ax_header.transAxes)
            ax_header.axis('off')
            col_idx += 1
        
        # Mask column
        ax_header = fig.add_subplot(gs_main[0, col_idx])
        acceleration = results.get('acceleration', '4')
        ax_header.text(0.5, 0.5, f'{acceleration}x sampling\nmask',
                      ha='center', va='center', fontsize=self.fontsize,
                      transform=ax_header.transAxes)
        ax_header.axis('off')
        
        # Image row
        col_idx = 0
        
        # Ground truth image
        ax_img = fig.add_subplot(gs_main[1, col_idx])
        im = ax_img.imshow(ground_truth, cmap='gray', vmin=vmin, vmax=vmax)
        ax_img.axis('off')
        col_idx += 1
        
        # Reconstruction images
        for method_name in method_names:
            ax_img = fig.add_subplot(gs_main[1, col_idx])
            recon = reconstructions[method_name]
            if np.iscomplexobj(recon):
                recon = np.abs(recon)
            im = ax_img.imshow(recon, cmap='gray', vmin=vmin, vmax=vmax)
            ax_img.axis('off')
            col_idx += 1
        
        # Mask image
        ax_mask = fig.add_subplot(gs_main[1, col_idx])
        ax_mask.imshow(mask, cmap='gray', vmin=0, vmax=1)
        ax_mask.axis('off')
        
        # Metrics rows
        metric_names = ['MSE', 'NRMSE', 'PSNR', 'SSIM']
        metric_colors = ['#4A90E2', '#B8D4F1', '#A8C8E8', '#C8DCF0']  # Blue shades
        
        for metric_idx, metric_name in enumerate(metric_names):
            row_idx = 2 + metric_idx
            
            # Metric label
            ax_label = fig.add_subplot(gs_main[row_idx, 0])
            ax_label.text(0.95, 0.5, metric_name, ha='right', va='center',
                         fontsize=self.fontsize + 1, fontweight='bold',
                         transform=ax_label.transAxes)
            ax_label.axis('off')
            
            # Metric values for each method
            for col_offset, method_name in enumerate(method_names):
                ax_metric = fig.add_subplot(gs_main[row_idx, col_offset + 1])
                
                # Get metric value
                if method_name in metrics and metric_name in metrics[method_name]:
                    value = metrics[method_name][metric_name]
                    if value is not None:
                        if metric_name in ['MSE', 'NRMSE']:
                            text = f'{value:.4f}'
                        else:
                            text = f'{value:.2f}'
                    else:
                        text = 'N/A'
                else:
                    text = 'N/A'
                
                # Color background
                ax_metric.add_patch(patches.Rectangle((0, 0), 1, 1,
                                                      facecolor=metric_colors[metric_idx],
                                                      transform=ax_metric.transAxes,
                                                      zorder=0))
                
                ax_metric.text(0.5, 0.5, text, ha='center', va='center',
                              fontsize=self.fontsize, transform=ax_metric.transAxes,
                              zorder=1)
                ax_metric.set_xlim(0, 1)
                ax_metric.set_ylim(0, 1)
                ax_metric.axis('off')
            
            # Empty cell for mask column
            ax_empty = fig.add_subplot(gs_main[row_idx, len(method_names) + 1])
            ax_empty.axis('off')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Figure saved to: {output_path}")
        
        return fig
    
    def create_summary_figure(self,
                            experiment_dir: str,
                            output_path: str,
                            metric: str = 'PSNR'):
        """
        Create a summary figure comparing multiple experiments
        
        Parameters:
        -----------
        experiment_dir : str
            Directory containing experiment HDF5 files
        output_path : str
            Path to save the summary figure
        metric : str
            Metric to compare ('MSE', 'NRMSE', 'PSNR', 'SSIM')
        """
        
        experiment_dir = Path(experiment_dir)
        h5_files = list(experiment_dir.glob('*.h5'))
        
        if not h5_files:
            print("No experiment files found!")
            return
        
        # Collect data
        all_data = {}
        method_names = set()
        
        for h5_file in h5_files:
            exp_name = h5_file.stem
            with h5py.File(h5_file, 'r') as f:
                if 'metrics' in f:
                    all_data[exp_name] = {}
                    for method_name in f['metrics'].keys():
                        method_names.add(method_name)
                        if metric in f['metrics'][method_name].attrs:
                            value = f['metrics'][method_name].attrs[metric]
                            all_data[exp_name][method_name] = value
        
        # Create bar plot
        method_names = sorted(list(method_names))
        exp_names = sorted(list(all_data.keys()))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(method_names))
        width = 0.8 / len(exp_names)
        
        for i, exp_name in enumerate(exp_names):
            values = [all_data[exp_name].get(method, 0) for method in method_names]
            ax.bar(x + i * width, values, width, label=exp_name)
        
        ax.set_xlabel('Reconstruction Method', fontsize=self.fontsize + 2)
        ax.set_ylabel(metric, fontsize=self.fontsize + 2)
        ax.set_title(f'{metric} Comparison Across Experiments', 
                    fontsize=self.fontsize + 4, fontweight='bold')
        ax.set_xticks(x + width * (len(exp_names) - 1) / 2)
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Summary figure saved to: {output_path}")
        
        return fig


def load_and_visualize_results(h5_file: str, output_dir: str = None):
    """
    Load results from HDF5 file and create visualization
    
    Parameters:
    -----------
    h5_file : str
        Path to HDF5 results file
    output_dir : str, optional
        Directory to save figures (defaults to same directory as h5_file)
    """
    
    h5_path = Path(h5_file)
    if output_dir is None:
        output_dir = h5_path.parent / 'figures'
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    with h5py.File(h5_path, 'r') as f:
        # Load reconstructions
        reconstructions = {}
        if 'reconstructions' in f:
            for method_name in f['reconstructions'].keys():
                reconstructions[method_name] = f['reconstructions'][method_name][:]
        
        # Load ground truth
        ground_truth = f['ground_truth'][:] if 'ground_truth' in f else None
        
        # Load mask
        mask = f['mask'][:] if 'mask' in f else None
        
        # Load metrics
        metrics = {}
        if 'metrics' in f:
            for method_name in f['metrics'].keys():
                metrics[method_name] = dict(f['metrics'][method_name].attrs)
        
        # Load config
        if 'config' in f.attrs:
            import json
            config = json.loads(f.attrs['config'])
            acceleration = config.get('undersampling', {}).get('acceleration_factor', 4)
        else:
            acceleration = 4
    
    # Create results dictionary
    results = {
        'reconstructions': reconstructions,
        'metrics': metrics,
        'ground_truth': ground_truth,
        'mask': mask,
        'acceleration': acceleration
    }
    
    # Generate figure
    fig_gen = ReconstructionFigureGenerator()
    output_path = output_dir / f"{h5_path.stem}_comparison.png"
    fig_gen.create_comparison_figure(results, str(output_path))
    
    return results


# Example usage
if __name__ == "__main__":
    # Example: Load and visualize results
    # load_and_visualize_results('experiments/data/test_experiment.h5')
    
    # Example: Create summary figure
    # fig_gen = ReconstructionFigureGenerator()
    # fig_gen.create_summary_figure('experiments/data', 'experiments/figures/summary.png')
    
    print("Figure generator module ready!")
