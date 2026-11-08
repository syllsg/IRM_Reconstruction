"""
Example Usage of MRI Reconstruction Pipeline
Demonstrates how to run complete experiments
"""

import numpy as np
from reconstruction_pipeline import ReconstructionPipeline, create_config_template
from figure_generator import ReconstructionFigureGenerator, load_and_visualize_results


def example_1_basic_experiment():
    """
    Example 1: Basic experiment with Cartesian undersampling
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Cartesian Undersampling Experiment")
    print("="*60)
    
    # Initialize pipeline
    pipeline = ReconstructionPipeline(output_dir="experiments")
    
    # Load your k-space data
    kspace = np.load('data/img_8st_slice.npy')
    sensitivity_maps = np.load('data/sensitivity_maps_8st_slice.npy')
    
    # For demonstration, create dummy data
    #print("\nCreating dummy k-space data (replace with your actual data)...")
    #n_coils = 8
    #img_size = 256
    #kspace = np.random.randn(n_coils, img_size, img_size) + \
     #        1j * np.random.randn(n_coils, img_size, img_size)
    #sensitivity_maps = np.random.randn(n_coils, img_size, img_size) + \
     #                 1j * np.random.randn(n_coils, img_size, img_size)
    
    # Create configuration for 4x Cartesian undersampling
    config = create_config_template(
        acceleration_factor=4,
        mask_type='cartesian',
        center_fraction=0.08,
        reconstruction_methods=['rss', 'sensitivity_based']
    )
    
    # Run experiment
    results = pipeline.run_experiment(
        kspace=kspace,
        config=config,
        experiment_name="cartesian_4x",
        sensitivity_maps=sensitivity_maps
    )
    
    # Generate figure
    print("\nGenerating comparison figure...")
    fig_gen = ReconstructionFigureGenerator()
    output_path = "experiments/figures/cartesian_4x_comparison.png"
    fig_gen.create_comparison_figure(results, output_path)
    
    print(f"\n✓ Experiment complete!")
    print(f"  - Data saved to: {results['output_file']}")
    print(f"  - Figure saved to: {output_path}")
    
    return results


def example_2_multiple_accelerations():
    """
    Example 2: Compare multiple acceleration factors
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Multiple Acceleration Factors")
    print("="*60)
    
    # Initialize pipeline
    pipeline = ReconstructionPipeline(output_dir="experiments")
    
    # Create dummy data
    n_coils = 8
    img_size = 256
    kspace = np.random.randn(n_coils, img_size, img_size) + \
             1j * np.random.randn(n_coils, img_size, img_size)
    sensitivity_maps = np.random.randn(n_coils, img_size, img_size) + \
                      1j * np.random.randn(n_coils, img_size, img_size)
    
    # Test different acceleration factors
    acceleration_factors = [2, 4, 8]
    
    for accel in acceleration_factors:
        print(f"\n--- Running experiment with {accel}x acceleration ---")
        
        config = create_config_template(
            acceleration_factor=accel,
            mask_type='cartesian',
            reconstruction_methods=['rss', 'sensitivity_based']
        )
        
        results = pipeline.run_experiment(
            kspace=kspace,
            config=config,
            experiment_name=f"cartesian_{accel}x",
            sensitivity_maps=sensitivity_maps
        )
        
        # Generate figure
        fig_gen = ReconstructionFigureGenerator()
        output_path = f"experiments/figures/cartesian_{accel}x_comparison.png"
        fig_gen.create_comparison_figure(results, output_path)
    
    # Create summary comparison
    print("\nCreating summary figure...")
    fig_gen = ReconstructionFigureGenerator()
    fig_gen.create_summary_figure(
        'experiments/data',
        'experiments/figures/acceleration_comparison.png',
        metric='PSNR'
    )
    
    print("\n✓ All experiments complete!")


def example_3_different_mask_types():
    """
    Example 3: Compare different undersampling patterns
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Different Undersampling Patterns")
    print("="*60)
    
    # Initialize pipeline
    pipeline = ReconstructionPipeline(output_dir="experiments")
    
    # Create dummy data
    n_coils = 8
    img_size = 256
    kspace = np.random.randn(n_coils, img_size, img_size) + \
             1j * np.random.randn(n_coils, img_size, img_size)
    sensitivity_maps = np.random.randn(n_coils, img_size, img_size) + \
                      1j * np.random.randn(n_coils, img_size, img_size)
    
    # Test different mask types
    mask_configs = [
        {'type': 'cartesian', 'name': 'Cartesian'},
        {'type': 'radial', 'name': 'Radial'},
        {'type': 'spiral', 'name': 'Spiral', 'num_spirals': 3, 'spiral_type': 'archimedean'},
        {'type': 'random', 'name': 'Random'}
    ]
    
    for mask_config in mask_configs:
        mask_type = mask_config['type']
        print(f"\n--- Running experiment with {mask_config['name']} mask ---")
        
        config = create_config_template(
            acceleration_factor=4,
            mask_type=mask_type,
            reconstruction_methods=['rss', 'sensitivity_based'],
            **{k: v for k, v in mask_config.items() if k not in ['type', 'name']}
        )
        
        results = pipeline.run_experiment(
            kspace=kspace,
            config=config,
            experiment_name=f"{mask_type}_4x",
            sensitivity_maps=sensitivity_maps
        )
        
        # Generate figure
        fig_gen = ReconstructionFigureGenerator()
        output_path = f"experiments/figures/{mask_type}_4x_comparison.png"
        fig_gen.create_comparison_figure(results, output_path)
    
    print("\n✓ All mask type experiments complete!")


def example_4_load_and_visualize():
    """
    Example 4: Load existing results and create new visualizations
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Load and Visualize Existing Results")
    print("="*60)
    
    # Assuming you have run some experiments
    h5_file = "experiments/data/cartesian_4x.h5"
    
    print(f"\nLoading results from: {h5_file}")
    
    try:
        results = load_and_visualize_results(h5_file, output_dir="experiments/figures")
        print("\n✓ Visualization created successfully!")
        
        # You can also access the loaded data
        print("\nAvailable reconstructions:")
        for method_name in results['reconstructions'].keys():
            print(f"  - {method_name}")
        
        print("\nMetrics:")
        for method_name, metrics in results['metrics'].items():
            print(f"  {method_name}:")
            for metric_name, value in metrics.items():
                print(f"    {metric_name}: {value}")
    
    except FileNotFoundError:
        print(f"File not found. Run example_1_basic_experiment() first!")


def example_5_custom_workflow():
    """
    Example 5: Custom workflow with your own reconstruction methods
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Custom Workflow")
    print("="*60)
    
    # Initialize pipeline
    pipeline = ReconstructionPipeline(output_dir="experiments")
    
    # You can add your own reconstruction methods
    def my_custom_reconstruction(masked_kspace, mask, sensitivity_maps):
        """Your custom reconstruction implementation"""
        # Implement your reconstruction here
        img_coils = np.fft.ifftshift(
            np.fft.ifft2(np.fft.fftshift(masked_kspace, axes=(-2, -1)), axes=(-2, -1)),
            axes=(-2, -1)
        )
        # Your processing...
        result = np.sqrt(np.sum(np.abs(img_coils)**2, axis=0))
        return result
    
    # Register your custom method
    pipeline.reconstruction_methods['my_custom_method'] = my_custom_reconstruction
    
    # Load data
    n_coils = 8
    img_size = 256
    kspace = np.random.randn(n_coils, img_size, img_size) + \
             1j * np.random.randn(n_coils, img_size, img_size)
    
    # Create config with your custom method
    config = create_config_template(
        acceleration_factor=4,
        mask_type='cartesian',
        reconstruction_methods=['rss', 'my_custom_method']
    )
    
    # Run experiment
    results = pipeline.run_experiment(
        kspace=kspace,
        config=config,
        experiment_name="custom_method_test"
    )
    
    print("\n✓ Custom workflow complete!")


if __name__ == "__main__":
    # Run examples
    print("\nMRI Reconstruction Pipeline - Examples")
    print("="*60)
    
    # Uncomment the example you want to run:
    
    # Example 1: Basic single experiment
    example_1_basic_experiment()
    
    # Example 2: Multiple accelerations
    # example_2_multiple_accelerations()
    
    # Example 3: Different mask types
    # example_3_different_mask_types()
    
    # Example 4: Load and visualize existing results
    # example_4_load_and_visualize()
    
    # Example 5: Custom workflow
    # example_5_custom_workflow()
