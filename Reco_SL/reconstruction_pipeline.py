"""
MRI Reconstruction Pipeline
Automated pipeline for k-space undersampling, reconstruction, and evaluation
"""

import numpy as np
import h5py
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings


def create_phantom_kspace(n_coils: int = 8, img_size: int = 256, 
                          add_noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create realistic phantom k-space data for testing
    
    Parameters:
    -----------
    n_coils : int
        Number of receiver coils
    img_size : int
        Image size (assumes square image)
    add_noise : bool
        Whether to add noise to k-space
        
    Returns:
    --------
    kspace : np.ndarray
        Multi-coil k-space data [n_coils, img_size, img_size]
    sensitivity_maps : np.ndarray
        Coil sensitivity maps [n_coils, img_size, img_size]
    """
    
    # Create a simple brain-like phantom
    y, x = np.ogrid[-img_size//2:img_size//2, -img_size//2:img_size//2]
    
    # Outer skull
    phantom = ((x**2 + y**2) < (img_size//2 - 20)**2).astype(float)
    # Brain tissue
    phantom += 0.5 * ((x**2 + y**2) < (img_size//2 - 40)**2).astype(float)
    # Inner structures (ventricles)
    phantom -= 0.3 * ((x**2 + (y-20)**2) < 400).astype(float)
    phantom -= 0.3 * ((x**2 + (y+20)**2) < 400).astype(float)
    # Gray/white matter contrast
    phantom += 0.3 * (((x-30)**2 + y**2) < 2000).astype(float)
    phantom += 0.3 * (((x+30)**2 + y**2) < 2000).astype(float)
    phantom = np.clip(phantom, 0, 1)
    
    # Create sensitivity maps (smooth Gaussian-like patterns for each coil)
    sensitivity_maps = np.zeros((n_coils, img_size, img_size), dtype=complex)
    for coil in range(n_coils):
        angle = 2 * np.pi * coil / n_coils
        # Position sensitivity maximum at different angles
        cx = img_size//2 + int(60 * np.cos(angle))
        cy = img_size//2 + int(60 * np.sin(angle))
        y_coil, x_coil = np.ogrid[:img_size, :img_size]
        sensitivity = np.exp(-((x_coil - cx)**2 + (y_coil - cy)**2) / (2 * 80**2))
        # Add phase variation
        phase = np.exp(1j * angle)
        sensitivity_maps[coil] = sensitivity * phase
    
    # Normalize sensitivity maps
    sensitivity_maps = sensitivity_maps / (np.sqrt(np.sum(np.abs(sensitivity_maps)**2, axis=0, keepdims=True)) + 1e-10)
    
    # Create multi-coil images
    img_coils = phantom[np.newaxis, :, :] * sensitivity_maps
    
    # Generate k-space by Fourier transform
    kspace = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(img_coils, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1)
    )
    
    # Add noise if requested
    if add_noise:
        noise_level = 0.01 * np.max(np.abs(kspace))
        kspace += noise_level * (np.random.randn(*kspace.shape) + 1j * np.random.randn(*kspace.shape))
    
    return kspace, sensitivity_maps

class ReconstructionPipeline:
    """
    Main pipeline for MRI reconstruction experiments
    """
    
    def __init__(self, output_dir: str = "experiments", save_format: str = "both"):
        """
        Initialize the pipeline
        
        Parameters:
        -----------
        output_dir : str
            Base directory for saving all outputs
        save_format : str
            Format for saving data: 'hdf5', 'numpy', or 'both' (default: 'both')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.save_format = save_format.lower()
        
        if self.save_format not in ['hdf5', 'numpy', 'both']:
            raise ValueError(f"save_format must be 'hdf5', 'numpy', or 'both', got: {save_format}")
        
        # Create subdirectories
        self.data_dir = self.output_dir / "data"
        self.figures_dir = self.output_dir / "figures"
        self.configs_dir = self.output_dir / "configs"
        
        for dir_path in [self.data_dir, self.figures_dir, self.configs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Available reconstruction methods
        self.reconstruction_methods = {
            'rss': self._rss_reconstruction,
            'sensitivity_based': self._sensitivity_based_reconstruction,
            'sense_sigpy': self._sigpy_sense_reconstruction,
            'gradient_descent': self._gradient_descent_reconstruction,
            'gd_l1_wavelet': self._gd_l1_wavelet_reconstruction,
            'unet': self._unet_reconstruction,
            'unrolled': self._unrolled_network_reconstruction,
        }
        
    def run_experiment(self, 
                       kspace: np.ndarray,
                       config: Dict[str, Any],
                       experiment_name: Optional[str] = None,
                       sensitivity_maps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run a complete reconstruction experiment
        
        Parameters:
        -----------
        kspace : np.ndarray
            Fully sampled k-space data [coils, kx, ky] or [coils, kx, ky, slices]
        config : dict
            Experiment configuration (see create_config_template)
        experiment_name : str, optional
            Name for this experiment (auto-generated if None)
        sensitivity_maps : np.ndarray, optional
            Pre-computed sensitivity maps
            
        Returns:
        --------
        dict : Results containing reconstructions and metrics
        """
        
        if experiment_name is None:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*60}")
        print(f"Starting Experiment: {experiment_name}")
        print(f"{'='*60}\n")
        
        # Validate k-space data
        print("Validating input data...")
        print(f"  K-space shape: {kspace.shape}")
        print(f"  K-space dtype: {kspace.dtype}")
        print(f"  K-space is complex: {np.iscomplexobj(kspace)}")
        
        if not np.iscomplexobj(kspace):
            raise ValueError(
                "K-space data must be complex! "
                f"Current dtype: {kspace.dtype}. "
                "Please ensure your k-space has complex dtype (complex64 or complex128). "
                "If you loaded from .npy file, it might have lost complex dtype. "
                "Try: kspace = kspace.astype(np.complex64) or reload with allow_pickle=True"
            )
        
        if sensitivity_maps is not None:
            print(f"  Sensitivity maps shape: {sensitivity_maps.shape}")
            print(f"  Sensitivity maps dtype: {sensitivity_maps.dtype}")
        
        print("  ✓ Data validation passed")
        
        # Save configuration
        config_path = self.configs_dir / f"{experiment_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(self._convert_config_to_serializable(config), f, indent=4)
        print(f"Configuration saved to: {config_path}")
        
        # Step 1: Apply undersampling mask
        print("\n[1/4] Applying undersampling mask...")
        masked_kspace, mask = self._apply_mask(kspace, config['undersampling'])
        
        # Step 2: Run all reconstruction methods
        print("\n[2/4] Running reconstruction methods...")
        reconstructions = self._run_reconstructions(
            masked_kspace, 
            mask,
            kspace,
            config['reconstruction_methods'],
            sensitivity_maps
        )
        
        # Step 3: Calculate metrics
        print("\n[3/4] Calculating metrics...")
        ground_truth = self._get_ground_truth(kspace, sensitivity_maps)
        metrics = self._calculate_all_metrics(reconstructions, ground_truth)
        
        # Step 4: Save results
        print("\n[4/4] Saving results...")
        output_file = self.data_dir / f"{experiment_name}.h5"
        self._save_results(
            output_file,
            kspace=kspace,
            masked_kspace=masked_kspace,
            mask=mask,
            reconstructions=reconstructions,
            metrics=metrics,
            ground_truth=ground_truth,
            config=config
        )
        
        print(f"\nResults saved to: {output_file}")
        print(f"\n{'='*60}")
        print("Experiment Complete!")
        print(f"{'='*60}\n")
        
        return {
            'reconstructions': reconstructions,
            'metrics': metrics,
            'ground_truth': ground_truth,
            'mask': mask,
            'experiment_name': experiment_name,
            'output_file': str(output_file)
        }
    
    def _convert_config_to_serializable(self, config):
        """Convert numpy arrays in config to lists for JSON serialization"""
        serializable_config = {}
        for key, value in config.items():
            if isinstance(value, np.ndarray):
                serializable_config[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_config[key] = self._convert_config_to_serializable(value)
            else:
                serializable_config[key] = value
        return serializable_config
    
    def _apply_mask(self, kspace: np.ndarray, mask_config: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply undersampling mask to k-space
        
        Parameters:
        -----------
        kspace : np.ndarray
            Fully sampled k-space
        mask_config : dict
            Mask configuration with keys: type, acceleration_factor, etc.
            
        Returns:
        --------
        masked_kspace, mask
        """
        mask_type = mask_config['type']
        acceleration = mask_config['acceleration_factor']
        
        print(f"  Mask type: {mask_type}")
        print(f"  Acceleration factor: {acceleration}x")
        
        if mask_type == 'cartesian':
            mask = self._create_cartesian_mask(kspace.shape, acceleration, mask_config)
        elif mask_type == 'spiral':
            mask = self._create_spiral_mask(kspace.shape, acceleration, mask_config)
        elif mask_type == 'radial':
            mask = self._create_radial_mask(kspace.shape, acceleration, mask_config)
        elif mask_type == 'random':
            mask = self._create_random_mask(kspace.shape, acceleration, mask_config)
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
        
        # Apply mask
        masked_kspace = kspace * mask[np.newaxis, ...]  # Broadcast over coils
        
        sampling_percentage = (np.sum(mask > 0) / mask.size) * 100
        print(f"  Sampling: {sampling_percentage:.2f}%")
        
        return masked_kspace, mask
    
    def _create_cartesian_mask(self, shape, acceleration, config):
        """Create Cartesian undersampling mask"""
        if len(shape) == 3:  # [coils, kx, ky]
            _, kx, ky = shape
        else:
            _, kx, ky, _ = shape
        
        mask = np.zeros((kx, ky))
        
        # Center lines (always sampled)
        center_fraction = config.get('center_fraction', 0.08)
        num_center_lines = int(ky * center_fraction)
        center_start = ky // 2 - num_center_lines // 2
        mask[:, center_start:center_start + num_center_lines] = 1
        
        # Outer lines (undersampled)
        remaining_lines = int((ky - num_center_lines) / acceleration)
        outer_indices = np.random.choice(
            [i for i in range(ky) if i < center_start or i >= center_start + num_center_lines],
            remaining_lines,
            replace=False
        )
        mask[:, outer_indices] = 1
        
        return mask
    
    def _create_spiral_mask(self, shape, acceleration, config):
        """Create spiral undersampling mask"""
        if len(shape) == 3:
            _, kx, ky = shape
        else:
            _, kx, ky, _ = shape
        
        num_spirals = config.get('num_spirals', 1)
        spiral_type = config.get('spiral_type', 'archimedean')
        
        mask = np.zeros((kx, ky))
        center_x, center_y = kx // 2, ky // 2
        
        # Create spiral trajectory
        max_radius = min(kx, ky) // 2
        num_points = int(np.pi * max_radius**2 / acceleration)
        
        for spiral_idx in range(num_spirals):
            theta_offset = 2 * np.pi * spiral_idx / num_spirals
            
            for i in range(num_points):
                theta = 2 * np.pi * i / (num_points / (max_radius / 10)) + theta_offset
                
                if spiral_type == 'archimedean':
                    r = max_radius * i / num_points
                elif spiral_type == 'fermat':
                    r = max_radius * np.sqrt(i / num_points)
                else:
                    r = max_radius * i / num_points
                
                x = int(center_x + r * np.cos(theta))
                y = int(center_y + r * np.sin(theta))
                
                if 0 <= x < kx and 0 <= y < ky:
                    mask[x, y] = 1
        
        return mask
    
    def _create_radial_mask(self, shape, acceleration, config):
        """Create radial undersampling mask"""
        if len(shape) == 3:
            _, kx, ky = shape
        else:
            _, kx, ky, _ = shape
        
        mask = np.zeros((kx, ky))
        center_x, center_y = kx // 2, ky // 2
        
        num_spokes = int(180 / acceleration)
        angles = np.linspace(0, np.pi, num_spokes, endpoint=False)
        
        max_radius = min(kx, ky) // 2
        
        for angle in angles:
            for r in range(max_radius):
                x = int(center_x + r * np.cos(angle))
                y = int(center_y + r * np.sin(angle))
                if 0 <= x < kx and 0 <= y < ky:
                    mask[x, y] = 1
                
                x = int(center_x - r * np.cos(angle))
                y = int(center_y - r * np.sin(angle))
                if 0 <= x < kx and 0 <= y < ky:
                    mask[x, y] = 1
        
        return mask
    
    def _create_random_mask(self, shape, acceleration, config):
        """Create random undersampling mask"""
        if len(shape) == 3:
            _, kx, ky = shape
        else:
            _, kx, ky, _ = shape
        
        sampling_rate = 1.0 / acceleration
        mask = np.random.rand(kx, ky) < sampling_rate
        
        # Always sample center
        center_fraction = config.get('center_fraction', 0.08)
        num_center = int(min(kx, ky) * center_fraction)
        center_x_start = kx // 2 - num_center // 2
        center_y_start = ky // 2 - num_center // 2
        mask[center_x_start:center_x_start+num_center, 
             center_y_start:center_y_start+num_center] = True
        
        return mask.astype(float)
    
    def _run_reconstructions(self, masked_kspace, mask, full_kspace, 
                            method_names, sensitivity_maps):
        """Run all specified reconstruction methods"""
        reconstructions = {}
        
        for method_name in method_names:
            if method_name not in self.reconstruction_methods:
                warnings.warn(f"Unknown reconstruction method: {method_name}")
                continue
            
            print(f"  Running {method_name}...")
            try:
                recon_func = self.reconstruction_methods[method_name]
                recon_img = recon_func(masked_kspace, mask, sensitivity_maps)
                reconstructions[method_name] = recon_img
                print(f"    ✓ Complete")
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                reconstructions[method_name] = None
        
        return reconstructions
    
    def _get_ground_truth(self, kspace, sensitivity_maps=None):
        """Get ground truth image from fully sampled k-space"""
        
        # Check if k-space is actually k-space (should be complex)
        if not np.iscomplexobj(kspace):
            warnings.warn(
                "WARNING: K-space data is not complex! This will produce incorrect results. "
                "Please ensure your k-space data has complex dtype (e.g., complex64 or complex128). "
                "Current dtype: {}".format(kspace.dtype)
            )
            # Assume it's already in image space if not complex
            img_coils = kspace
        else:
            # Apply inverse FFT: ifftshift -> ifft2 -> fftshift (with ortho normalization)
            img_coils = np.fft.fftshift(
                np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), axes=(-2, -1), norm='ortho'),
                axes=(-2, -1)
            )
        
        # Combine coils
        if sensitivity_maps is not None:
            # Sensitivity-based combination
            ground_truth = np.sum(img_coils * np.conj(sensitivity_maps), axis=0)
        else:
            # RSS combination
            ground_truth = np.sqrt(np.sum(np.abs(img_coils)**2, axis=0))
        
        return np.abs(ground_truth)
    
    def _calculate_all_metrics(self, reconstructions, ground_truth):
        """Calculate metrics for all reconstructions"""
        from run_metrics import run_all_metrics  # Import your metrics function
        
        metrics = {}
        for method_name, recon_img in reconstructions.items():
            if recon_img is None:
                continue
            
            print(f"  Calculating metrics for {method_name}...")
            
            # Convert to magnitude if complex
            if np.iscomplexobj(recon_img):
                recon_img = np.abs(recon_img)
            
            try:
                # Calculate metrics without saving to file
                results = {}
                results['MSE'] = self._mse(ground_truth, recon_img)
                results['NRMSE'] = self._nrmse(ground_truth, recon_img)
                results['PSNR'] = self._psnr(ground_truth, recon_img)
                results['SSIM'] = self._ssim(ground_truth, recon_img)
                
                metrics[method_name] = results
                
            except Exception as e:
                print(f"    Warning: Could not calculate metrics: {e}")
                metrics[method_name] = {'MSE': None, 'NRMSE': None, 'PSNR': None, 'SSIM': None}
        
        return metrics
    
    def _mse(self, img_true, img_pred):
        """Calculate MSE"""
        return float(np.mean((img_true - img_pred)**2))
    
    def _nrmse(self, img_true, img_pred):
        """Calculate NRMSE"""
        mse_val = self._mse(img_true, img_pred)
        return float(np.sqrt(mse_val) / (np.max(img_true) - np.min(img_true)))
    
    def _psnr(self, img_true, img_pred):
        """Calculate PSNR"""
        mse_val = self._mse(img_true, img_pred)
        if mse_val == 0:
            return float('inf')
        max_val = np.max(img_true)
        return float(20 * np.log10(max_val / np.sqrt(mse_val)))
    
    def _ssim(self, img_true, img_pred):
        """Calculate SSIM"""
        try:
            from skimage.metrics import structural_similarity
            return float(structural_similarity(img_true, img_pred, data_range=img_true.max() - img_true.min()))
        except:
            return None
    
    def _save_results(self, output_file, **data):
        """Save all results according to save_format setting"""
        
        output_path = Path(output_file)
        
        # Save to HDF5 if requested
        if self.save_format in ['hdf5', 'both']:
            print(f"\nSaving to HDF5: {output_file}")
            with h5py.File(output_file, 'w') as f:
                # Save k-space data
                if 'kspace' in data:
                    f.create_dataset('kspace', data=data['kspace'], compression='gzip')
                if 'masked_kspace' in data:
                    f.create_dataset('masked_kspace', data=data['masked_kspace'], compression='gzip')
                if 'mask' in data:
                    f.create_dataset('mask', data=data['mask'], compression='gzip')
                if 'ground_truth' in data:
                    f.create_dataset('ground_truth', data=data['ground_truth'], compression='gzip')
                
                # Save reconstructions
                if 'reconstructions' in data:
                    recon_group = f.create_group('reconstructions')
                    for method_name, recon_img in data['reconstructions'].items():
                        if recon_img is not None:
                            recon_group.create_dataset(method_name, data=recon_img, compression='gzip')
                
                # Save metrics
                if 'metrics' in data:
                    metrics_group = f.create_group('metrics')
                    for method_name, method_metrics in data['metrics'].items():
                        method_group = metrics_group.create_group(method_name)
                        for metric_name, value in method_metrics.items():
                            if value is not None:
                                method_group.attrs[metric_name] = value
                
                # Save config as attributes
                if 'config' in data:
                    config_str = json.dumps(self._convert_config_to_serializable(data['config']))
                    f.attrs['config'] = config_str
        
        # Save numpy arrays if requested
        if self.save_format in ['numpy', 'both']:
            numpy_dir = output_path.parent / f"{output_path.stem}_numpy"
            numpy_dir.mkdir(exist_ok=True, parents=True)
            
            print(f"\nSaving numpy arrays to: {numpy_dir}")
            
            # Save mask
            if 'mask' in data:
                np.save(numpy_dir / 'mask.npy', data['mask'])
                print(f"  ✓ mask.npy")
            
            # Save ground truth
            if 'ground_truth' in data:
                np.save(numpy_dir / 'ground_truth.npy', data['ground_truth'])
                print(f"  ✓ ground_truth.npy")
            
            # Save k-space (optional, can be large)
            if 'kspace' in data:
                np.save(numpy_dir / 'kspace.npy', data['kspace'])
                print(f"  ✓ kspace.npy")
            
            if 'masked_kspace' in data:
                np.save(numpy_dir / 'masked_kspace.npy', data['masked_kspace'])
                print(f"  ✓ masked_kspace.npy")
            
            # Save reconstructions
            if 'reconstructions' in data:
                recon_dir = numpy_dir / 'reconstructions'
                recon_dir.mkdir(exist_ok=True)
                
                for method_name, recon_img in data['reconstructions'].items():
                    if recon_img is not None:
                        # Convert to magnitude if complex
                        if np.iscomplexobj(recon_img):
                            # Save both magnitude and complex
                            np.save(recon_dir / f'{method_name}_magnitude.npy', np.abs(recon_img))
                            np.save(recon_dir / f'{method_name}_complex.npy', recon_img)
                            print(f"  ✓ {method_name}_magnitude.npy and {method_name}_complex.npy")
                        else:
                            np.save(recon_dir / f'{method_name}.npy', recon_img)
                            print(f"  ✓ {method_name}.npy")
            
            # Save metrics as JSON
            if 'metrics' in data:
                metrics_file = numpy_dir / 'metrics.json'
                with open(metrics_file, 'w') as f:
                    json.dump(data['metrics'], f, indent=4)
                print(f"  ✓ metrics.json")
            
            # Save config
            if 'config' in data:
                config_file = numpy_dir / 'config.json'
                with open(config_file, 'w') as f:
                    json.dump(self._convert_config_to_serializable(data['config']), f, indent=4)
                print(f"  ✓ config.json")
            
            print(f"\nNumpy arrays saved to: {numpy_dir}")
    
    # Placeholder reconstruction methods - replace with your actual implementations
    def _rss_reconstruction(self, masked_kspace, mask, sensitivity_maps):
        """RSS reconstruction"""
        print(f"    [DEBUG] RSS input k-space shape: {masked_kspace.shape}, dtype: {masked_kspace.dtype}")
        
        # Apply inverse FFT using the correct convention:
        # ifftshift -> ifft2 -> fftshift (with ortho normalization)
        img_coils = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(masked_kspace, axes=(-2, -1)), axes=(-2, -1), norm='ortho'),
            axes=(-2, -1)
        )
        
        print(f"    [DEBUG] After inverse FFT shape: {img_coils.shape}, dtype: {img_coils.dtype}")
        
        # RSS combination
        rss_img = np.sqrt(np.sum(np.abs(img_coils)**2, axis=0))
        print(f"    [DEBUG] RSS output shape: {rss_img.shape}, dtype: {rss_img.dtype}")
        print(f"    [DEBUG] RSS output range: [{rss_img.min():.3e}, {rss_img.max():.3e}]")
        
        # Check if result looks like k-space or image
        center = rss_img[rss_img.shape[0]//2-10:rss_img.shape[0]//2+10,
                        rss_img.shape[1]//2-10:rss_img.shape[1]//2+10]
        center_mean = np.mean(center)
        total_mean = np.mean(rss_img)
        ratio = center_mean / (total_mean + 1e-10)
        print(f"    [DEBUG] Center/Mean intensity ratio: {ratio:.2f}")
        if ratio > 5.0:
            print(f"    [DEBUG] ⚠️  WARNING: Result looks like K-SPACE (bright center)!")
        else:
            print(f"    [DEBUG] ✓ Result looks like IMAGE (no bright center)")
        
        return rss_img
    
    def _sensitivity_based_reconstruction(self, masked_kspace, mask, sensitivity_maps):
        """Sensitivity-based reconstruction"""
        if sensitivity_maps is None:
            raise ValueError("Sensitivity maps required for this method")
        
        print(f"    [DEBUG] Sens input k-space shape: {masked_kspace.shape}")
        
        # Apply inverse FFT: ifftshift -> ifft2 -> fftshift (with ortho normalization)
        img_coils = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(masked_kspace, axes=(-2, -1)), axes=(-2, -1), norm='ortho'),
            axes=(-2, -1)
        )
        
        print(f"    [DEBUG] After inverse FFT shape: {img_coils.shape}")
        
        # Sensitivity-based combination
        sens_img = np.sum(img_coils * np.conj(sensitivity_maps), axis=0)
        result = np.abs(sens_img)
        
        print(f"    [DEBUG] Sens output range: [{result.min():.3e}, {result.max():.3e}]")
        
        return result
    
    def _sigpy_sense_reconstruction(self, masked_kspace, mask, sensitivity_maps):
        """SigPy SENSE reconstruction"""
        # Your SigPy implementation here
        raise NotImplementedError("Implement SigPy SENSE reconstruction")
    
    def _gradient_descent_reconstruction(self, masked_kspace, mask, sensitivity_maps):
        """Gradient descent reconstruction"""
        # Your gradient descent implementation here
        raise NotImplementedError("Implement gradient descent reconstruction")
    
    def _gd_l1_wavelet_reconstruction(self, masked_kspace, mask, sensitivity_maps):
        """Gradient descent with L1 wavelet regularization"""
        # Your regularized gradient descent implementation here
        raise NotImplementedError("Implement GD + L1 wavelet reconstruction")
    
    def _unet_reconstruction(self, masked_kspace, mask, sensitivity_maps):
        """U-Net reconstruction"""
        # Your U-Net implementation here
        raise NotImplementedError("Implement U-Net reconstruction")
    
    def _unrolled_network_reconstruction(self, masked_kspace, mask, sensitivity_maps):
        """Unrolled network reconstruction"""
        # Your unrolled network implementation here
        raise NotImplementedError("Implement unrolled network reconstruction")


def create_config_template(
    acceleration_factor: int = 4,
    mask_type: str = 'cartesian',
    reconstruction_methods: List[str] = None,
    **kwargs
) -> Dict:
    """
    Create a configuration template for experiments
    
    Parameters:
    -----------
    acceleration_factor : int
        Undersampling acceleration factor (e.g., 2, 4, 8)
    mask_type : str
        Type of undersampling mask: 'cartesian', 'spiral', 'radial', 'random'
    reconstruction_methods : list of str
        List of reconstruction methods to run
    **kwargs : additional parameters
        
    Returns:
    --------
    dict : Configuration dictionary
    """
    
    if reconstruction_methods is None:
        reconstruction_methods = [
            'rss',
            'sensitivity_based',
            'sense_sigpy',
            'gradient_descent',
            'gd_l1_wavelet'
        ]
    
    config = {
        'undersampling': {
            'type': mask_type,
            'acceleration_factor': acceleration_factor,
            'center_fraction': kwargs.get('center_fraction', 0.08),
        },
        'reconstruction_methods': reconstruction_methods,
    }
    
    # Add mask-specific parameters
    if mask_type == 'spiral':
        config['undersampling']['num_spirals'] = kwargs.get('num_spirals', 1)
        config['undersampling']['spiral_type'] = kwargs.get('spiral_type', 'archimedean')
    
    return config


# Example usage
if __name__ == "__main__":
    # Initialize pipeline with save format option
    # Options: 'hdf5', 'numpy', or 'both' (default)
    
    # Option 1: Save only numpy arrays (no HDF5)
    pipeline = ReconstructionPipeline(output_dir="mri_experiments", save_format="numpy")
    
    # Option 2: Save only HDF5
    # pipeline = ReconstructionPipeline(output_dir="mri_experiments", save_format="hdf5")
    
    # Option 3: Save both (default)
    # pipeline = ReconstructionPipeline(output_dir="mri_experiments", save_format="both")
    
    # Create configuration
    config = create_config_template(
        acceleration_factor=4,
        mask_type='cartesian',
        reconstruction_methods=['rss', 'sensitivity_based']
    )
    
    # Load your k-space data
    # kspace = np.load('your_kspace.npy')
    # sensitivity_maps = np.load('your_sensitivity_maps.npy')  # Optional
    
    # Create realistic dummy data from a phantom image
    print("\nCreating phantom k-space data for demonstration...")
    
    # Create a simple brain-like phantom
    n_coils = 8
    img_size = 256
    
    # Create phantom image (circles representing brain structure)
    y, x = np.ogrid[-img_size//2:img_size//2, -img_size//2:img_size//2]
    
    # Outer skull
    phantom = ((x**2 + y**2) < (img_size//2 - 20)**2).astype(float)
    # Brain tissue
    phantom += 0.5 * ((x**2 + y**2) < (img_size//2 - 40)**2).astype(float)
    # Inner structures (ventricles)
    phantom -= 0.3 * ((x**2 + (y-20)**2) < 400).astype(float)
    phantom -= 0.3 * ((x**2 + (y+20)**2) < 400).astype(float)
    # Gray/white matter contrast
    phantom += 0.3 * (((x-30)**2 + y**2) < 2000).astype(float)
    phantom += 0.3 * (((x+30)**2 + y**2) < 2000).astype(float)
    phantom = np.clip(phantom, 0, 1)
    
    # Create sensitivity maps (smooth Gaussian-like patterns for each coil)
    sensitivity_maps = np.zeros((n_coils, img_size, img_size), dtype=complex)
    for coil in range(n_coils):
        angle = 2 * np.pi * coil / n_coils
        # Position sensitivity maximum at different angles
        cx = img_size//2 + int(60 * np.cos(angle))
        cy = img_size//2 + int(60 * np.sin(angle))
        y_coil, x_coil = np.ogrid[:img_size, :img_size]
        sensitivity = np.exp(-((x_coil - cx)**2 + (y_coil - cy)**2) / (2 * 80**2))
        # Add phase variation
        phase = np.exp(1j * angle)
        sensitivity_maps[coil] = sensitivity * phase
    
    # Normalize sensitivity maps
    sensitivity_maps = sensitivity_maps / (np.sqrt(np.sum(np.abs(sensitivity_maps)**2, axis=0, keepdims=True)) + 1e-10)
    
    # Create multi-coil images
    img_coils = phantom[np.newaxis, :, :] * sensitivity_maps
    
    # Generate k-space by Fourier transform
    kspace = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(img_coils, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1)
    )
    
    # Add small amount of noise
    noise_level = 0.01 * np.max(np.abs(kspace))
    kspace += noise_level * (np.random.randn(*kspace.shape) + 1j * np.random.randn(*kspace.shape))
    
    print(f"✓ Created phantom k-space: shape {kspace.shape}, complex: {np.iscomplexobj(kspace)}")
    
    # Run experiment
    results = pipeline.run_experiment(
        kspace=kspace,
        config=config,
        experiment_name="test_experiment",
        sensitivity_maps=sensitivity_maps
    )
    
    print("\nExperiment completed!")
    print(f"Output file: {results['output_file']}")
    
    # If saving as numpy, your reconstructions will be in:
    # mri_experiments/data/test_experiment_numpy/reconstructions/
    # - rss.npy (or rss_magnitude.npy if complex)
    # - sensitivity_based.npy
    # etc.
