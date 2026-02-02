import matplotlib.pyplot as plt
import numpy as np




#=============== Spiral mask creation
def create_spiral_mask(kspace_shape, num_spirals=1, undersampling_factor=4):
    """
    Create a spiral sampling mask for k-space
    
    Parameters:
    - kspace_shape: tuple (ny, nx) - shape of k-space
    - num_spirals: number of interleaved spirals
    - undersampling_factor: acceleration factor (higher = more undersampling)
    """
    ny, nx = kspace_shape
    cy, cx = ny // 2, nx // 2  # center of k-space
    
    # Create empty mask
    mask = np.zeros((ny, nx), dtype=bool)
    
    # Maximum radius (from center to corner)
    max_radius = np.sqrt(cy**2 + cx**2)
    
    # Number of points along spiral
    num_points = int((ny * nx) / undersampling_factor)
    
    for spiral_idx in range(num_spirals):
        # Archimedean spiral: r = a * theta
        theta = np.linspace(0, num_spirals * 10 * np.pi, num_points)
        r = (theta / theta.max()) * max_radius
        
        # Add rotation offset for interleaved spirals
        theta_offset = spiral_idx * (2 * np.pi / num_spirals)
        
        # Convert polar to Cartesian coordinates
        y = cy + r * np.sin(theta + theta_offset)
        x = cx + r * np.cos(theta + theta_offset)
        
        # Round to integer indices and clip to valid range
        y_idx = np.clip(np.round(y).astype(int), 0, ny - 1)
        x_idx = np.clip(np.round(x).astype(int), 0, nx - 1)
        
        # Set mask points
        mask[y_idx, x_idx] = True
    
    # Always sample center of k-space (important for image quality)
    center_size = 10
    mask[cy-center_size:cy+center_size, cx-center_size:cx+center_size] = True
    
    return mask


def create_golden_angle_spiral(kspace_shape, num_spirals=16, undersampling_factor=4):
    """
    Golden angle spiral - standard in clinical MRI
    """
    ny, nx = kspace_shape
    cy, cx = ny // 2, nx // 2
    
    mask = np.zeros((ny, nx), dtype=bool)
    max_radius = np.sqrt(cy**2 + cx**2)
    
    # Golden angle in radians
    golden_angle = np.pi * (3 - np.sqrt(5))  # ≈ 137.5 degrees
    
    total_points = (ny * nx) // undersampling_factor
    points_per_spiral = total_points // num_spirals
    
    for spiral_idx in range(num_spirals):
        # Uniform spacing in radius
        r = np.linspace(0, max_radius, points_per_spiral)
        
        # Golden angle rotation for each spiral
        theta_offset = spiral_idx * golden_angle
        
        # Archimedean spiral equation
        # Number of rotations increases with radius
        theta = (r / max_radius) * 15 * 2 * np.pi + theta_offset
        
        y = cy + r * np.sin(theta)
        x = cx + r * np.cos(theta)
        
        y_idx = np.clip(np.round(y).astype(int), 0, ny - 1)
        x_idx = np.clip(np.round(x).astype(int), 0, nx - 1)
        
        mask[y_idx, x_idx] = True
    
    # Always fully sample center
    center_radius = 15
    y_grid, x_grid = np.ogrid[:ny, :nx]
    center_mask = ((y_grid - cy)**2 + (x_grid - cx)**2) <= center_radius**2
    mask[center_mask] = True
    
    return mask
