"""
QUICK INTEGRATION GUIDE
How to integrate the pipeline with your existing MRI reconstruction code
"""

import numpy as np
from reconstruction_pipeline import ReconstructionPipeline, create_config_template
from figure_generator import ReconstructionFigureGenerator

# ============================================================================
# STEP 1: CONNECT YOUR EXISTING RECONSTRUCTION FUNCTIONS
# ============================================================================

# Assuming you have functions like these in your existing code:
# - rss_reconstruction(kspace)
# - sensitivity_based_reconstruction(kspace, sensitivity_maps)
# - sigpy_sense_reconstruction(masked_kspace, mask, sensitivity_maps)
# etc.

class MyReconstructionPipeline(ReconstructionPipeline):
    """
    Extended pipeline with your existing reconstruction methods
    """
    
    def __init__(self, output_dir="experiments"):
        super().__init__(output_dir)
        
        # Import your existing functions here
        # from your_recon_module import (
        #     rss_recon, sensitivity_recon, sigpy_sense, 
        #     gradient_descent_recon, wavelet_recon
        # )
        
        # Override reconstruction methods with your implementations
        # self.reconstruction_methods['rss'] = self._wrap_rss
        # self.reconstruction_methods['sensitivity_based'] = self._wrap_sensitivity
        # etc.
    
    def _wrap_rss(self, masked_kspace, mask, sensitivity_maps):
        """Wrapper for your RSS reconstruction"""
        # Call your existing function
        # return rss_recon(masked_kspace)
        
        # Example implementation:
        img_coils = np.fft.ifftshift(
            np.fft.ifft2(np.fft.fftshift(masked_kspace, axes=(-2, -1)), axes=(-2, -1)),
            axes=(-2, -1)
        )
        return np.sqrt(np.sum(np.abs(img_coils)**2, axis=0))
    
    def _wrap_sensitivity(self, masked_kspace, mask, sensitivity_maps):
        """Wrapper for your sensitivity-based reconstruction"""
        if sensitivity_maps is None:
            raise ValueError("Sensitivity maps required")
        
        # Call your existing function
        # return sensitivity_recon(masked_kspace, sensitivity_maps)
        
        # Example implementation:
        img_coils = np.fft.ifftshift(
            np.fft.ifft2(np.fft.fftshift(masked_kspace, axes=(-2, -1)), axes=(-2, -1)),
            axes=(-2, -1)
        )
        return np.abs(np.sum(img_coils * np.conj(sensitivity_maps), axis=0))


# ============================================================================
# STEP 2: SIMPLE USAGE EXAMPLE
# ============================================================================

def simple_example():
    """
    Simple example matching your workflow from the screenshot
    """
    
    # Initialize pipeline
    pipeline = MyReconstructionPipeline(output_dir="mri_experiments")
    
    # ---- Load your data ----
    # Replace these with your actual data loading
    print("Loading data...")
    
    # Option A: Load from files
    # kspace = np.load('/path/to/your/kspace.npy')
    # sensitivity_maps = np.load('/path/to/your/smaps.npy')
    
    # Option B: Use data you already have in memory
    # kspace = your_existing_kspace_variable
    # sensitivity_maps = your_existing_sensitivity_maps
    
    # For demonstration:
    n_coils = 8
    img_size = 256
    kspace = np.random.randn(n_coils, img_size, img_size) + \
             1j * np.random.randn(n_coils, img_size, img_size)
    sensitivity_maps = np.random.randn(n_coils, img_size, img_size) + \
                      1j * np.random.randn(n_coils, img_size, img_size)
    
    # ---- Configure experiment ----
    print("Setting up experiment configuration...")
    
    config = create_config_template(
        acceleration_factor=4,
        mask_type='cartesian',
        center_fraction=0.08,
        reconstruction_methods=[
            'rss',
            'sensitivity_based',
            # Add more methods as you implement them:
            # 'sense_sigpy',
            # 'gradient_descent',
            # 'gd_l1_wavelet'
        ]
    )
    
    # ---- Run experiment ----
    print("Running reconstruction experiment...")
    
    results = pipeline.run_experiment(
        kspace=kspace,
        config=config,
        experiment_name="cartesian_4x_multicoil",
        sensitivity_maps=sensitivity_maps
    )
    
    # ---- Generate figure (like your screenshot) ----
    print("Generating comparison figure...")
    
    fig_gen = ReconstructionFigureGenerator(fontsize=10)
    output_path = "mri_experiments/figures/reconstruction_synthesis.png"
    
    fig_gen.create_comparison_figure(
        results,
        output_path,
        figsize=(16, 10)
    )
    
    print(f"\n✓ Complete!")
    print(f"  Data saved: {results['output_file']}")
    print(f"  Figure saved: {output_path}")
    
    # ---- Access results ----
    print("\nResults summary:")
    for method, metrics in results['metrics'].items():
        print(f"\n{method}:")
        for metric_name, value in metrics.items():
            if value is not None:
                print(f"  {metric_name}: {value:.4f}")


# ============================================================================
# STEP 3: BATCH PROCESSING MULTIPLE EXPERIMENTS
# ============================================================================

def batch_processing_example():
    """
    Run experiments with different parameters
    """
    
    pipeline = MyReconstructionPipeline(output_dir="mri_experiments")
    
    # Load your data once
    kspace = ...  # Your k-space data
    sensitivity_maps = ...  # Your sensitivity maps
    
    # Define experiments to run
    experiments = [
        {
            'name': 'cartesian_2x',
            'acceleration': 2,
            'mask_type': 'cartesian'
        },
        {
            'name': 'cartesian_4x',
            'acceleration': 4,
            'mask_type': 'cartesian'
        },
        {
            'name': 'cartesian_8x',
            'acceleration': 8,
            'mask_type': 'cartesian'
        },
        {
            'name': 'spiral_4x',
            'acceleration': 4,
            'mask_type': 'spiral',
            'num_spirals': 3
        }
    ]
    
    # Run all experiments
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Running: {exp['name']}")
        print(f"{'='*60}")
        
        config = create_config_template(
            acceleration_factor=exp['acceleration'],
            mask_type=exp['mask_type'],
            reconstruction_methods=['rss', 'sensitivity_based'],
            **{k: v for k, v in exp.items() 
               if k not in ['name', 'acceleration', 'mask_type']}
        )
        
        results = pipeline.run_experiment(
            kspace=kspace,
            config=config,
            experiment_name=exp['name'],
            sensitivity_maps=sensitivity_maps
        )
        
        # Generate figure for each experiment
        fig_gen = ReconstructionFigureGenerator()
        fig_path = f"mri_experiments/figures/{exp['name']}_comparison.png"
        fig_gen.create_comparison_figure(results, fig_path)
    
    # Create summary comparison across all experiments
    print("\nGenerating summary comparison...")
    fig_gen = ReconstructionFigureGenerator()
    fig_gen.create_summary_figure(
        'mri_experiments/data',
        'mri_experiments/figures/all_experiments_summary.png',
        metric='PSNR'
    )
    
    print("\n✓ All experiments complete!")


# ============================================================================
# STEP 4: WORKING WITH YOUR EXISTING WORKFLOW
# ============================================================================

def integrate_with_existing_code():
    """
    Example showing how to use the pipeline within your existing workflow
    """
    
    # Your existing code for loading/preparing data
    print("Step 1: Load data (your existing code)...")
    # kspace_data = your_load_kspace_function()
    # sensitivity_maps = your_compute_sensitivity_maps(kspace_data)
    
    # Your existing preprocessing
    # kspace_data = your_preprocessing_function(kspace_data)
    
    # NEW: Use pipeline for reconstruction and evaluation
    print("Step 2: Run pipeline for reconstruction...")
    pipeline = MyReconstructionPipeline(output_dir="results")
    
    config = create_config_template(
        acceleration_factor=4,
        mask_type='cartesian',
        reconstruction_methods=['rss', 'sensitivity_based']
    )
    
    results = pipeline.run_experiment(
        kspace=kspace_data,  # Your data
        config=config,
        experiment_name="my_scan",
        sensitivity_maps=sensitivity_maps  # Your sensitivity maps
    )
    
    # NEW: Automatic figure generation
    print("Step 3: Generate figure...")
    fig_gen = ReconstructionFigureGenerator()
    fig_gen.create_comparison_figure(results, "my_results.png")
    
    # Continue with your existing analysis/processing
    print("Step 4: Continue with your analysis...")
    # reconstructed_image = results['reconstructions']['rss']
    # your_post_processing_function(reconstructed_image)
    # your_analysis_function(results['metrics'])


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("MRI Reconstruction Pipeline - Integration Guide")
    print("="*60)
    
    # Run the simple example
    simple_example()
    
    # To run batch processing:
    # batch_processing_example()
    
    # To see integration with existing code:
    # integrate_with_existing_code()
