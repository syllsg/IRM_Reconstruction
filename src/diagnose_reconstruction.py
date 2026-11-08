"""
Diagnostic: Check what's actually saved in the reconstruction files
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("DIAGNOSTIC: Checking saved reconstruction files")
print("="*70)

# Load the files
print("\n1. Loading files...")
ground_truth = np.load('experiments/data/cartesian_4x_numpy/ground_truth.npy')
rss_recon = np.load('experiments/data/cartesian_4x_numpy/reconstructions/rss.npy')
kspace = np.load('experiments/data/cartesian_4x_numpy/kspace.npy')
mask = np.load('experiments/data/cartesian_4x_numpy/mask.npy')

print(f"\nGround truth:")
print(f"  Shape: {ground_truth.shape}")
print(f"  Dtype: {ground_truth.dtype}")
print(f"  Is complex: {np.iscomplexobj(ground_truth)}")
print(f"  Range: [{ground_truth.min():.2e}, {ground_truth.max():.2e}]")

print(f"\nRSS reconstruction:")
print(f"  Shape: {rss_recon.shape}")
print(f"  Dtype: {rss_recon.dtype}")
print(f"  Is complex: {np.iscomplexobj(rss_recon)}")
print(f"  Range: [{rss_recon.min():.2e}, {rss_recon.max():.2e}]")

print(f"\nK-space:")
print(f"  Shape: {kspace.shape}")
print(f"  Dtype: {kspace.dtype}")
print(f"  Is complex: {np.iscomplexobj(kspace)}")

print(f"\nMask:")
print(f"  Shape: {mask.shape}")
print(f"  Sampling: {np.sum(mask > 0) / mask.size * 100:.1f}%")

# Check if data looks like k-space or image space
def is_kspace_like(data):
    """Check if data looks like k-space (bright center)"""
    if len(data.shape) == 2:
        center = data[data.shape[0]//2-10:data.shape[0]//2+10,
                     data.shape[1]//2-10:data.shape[1]//2+10]
    else:
        # Multi-coil, check first coil
        center = data[0, data.shape[1]//2-10:data.shape[1]//2+10,
                     data.shape[2]//2-10:data.shape[2]//2+10]
    
    center_mean = np.mean(np.abs(center))
    total_mean = np.mean(np.abs(data))
    
    ratio = center_mean / (total_mean + 1e-10)
    return ratio > 5.0, ratio

print("\n" + "="*70)
print("2. Analysis:")
print("="*70)

# Check ground truth
is_ks, ratio = is_kspace_like(ground_truth)
print(f"\nGround truth:")
print(f"  Center/Mean intensity ratio: {ratio:.2f}")
if is_ks:
    print(f"  ❌ LOOKS LIKE K-SPACE (bright center)")
else:
    print(f"  ✓ Looks like IMAGE (no bright center)")

# Check RSS reconstruction
is_ks, ratio = is_kspace_like(rss_recon)
print(f"\nRSS reconstruction:")
print(f"  Center/Mean intensity ratio: {ratio:.2f}")
if is_ks:
    print(f"  ❌ LOOKS LIKE K-SPACE (bright center)")
    print(f"  This means inverse FFT was NOT applied!")
else:
    print(f"  ✓ Looks like IMAGE (no bright center)")

# Visualize
print("\n" + "="*70)
print("3. Creating visualization...")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: K-space
if len(kspace.shape) == 3:
    axes[0, 0].imshow(np.log(1 + np.abs(kspace[0])), cmap='gray')
    axes[0, 0].set_title('K-space (log scale)\nShould have bright center')
else:
    axes[0, 0].imshow(np.log(1 + np.abs(kspace)), cmap='gray')
    axes[0, 0].set_title('K-space (log scale)\nShould have bright center')
axes[0, 0].axis('off')

axes[0, 1].imshow(ground_truth, cmap='gray')
axes[0, 1].set_title(f'Ground Truth\nRatio: {is_kspace_like(ground_truth)[1]:.1f}')
axes[0, 1].axis('off')

axes[0, 2].imshow(rss_recon, cmap='gray')
axes[0, 2].set_title(f'RSS Reconstruction\nRatio: {is_kspace_like(rss_recon)[1]:.1f}')
axes[0, 2].axis('off')

# Row 2: What it should look like
axes[1, 0].imshow(mask, cmap='gray')
axes[1, 0].set_title('Undersampling Mask')
axes[1, 0].axis('off')

# Manual RSS reconstruction to show what it SHOULD be
print("\nPerforming manual RSS reconstruction for comparison...")
if len(kspace.shape) == 3:
    img_coils = np.fft.ifftshift(
        np.fft.ifft2(np.fft.fftshift(kspace, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1)
    )
    manual_rss = np.sqrt(np.sum(np.abs(img_coils)**2, axis=0))
else:
    img = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(kspace)))
    manual_rss = np.abs(img)

axes[1, 1].imshow(manual_rss, cmap='gray')
axes[1, 1].set_title(f'Manual RSS (correct)\nRatio: {is_kspace_like(manual_rss)[1]:.1f}')
axes[1, 1].axis('off')

# Difference
if rss_recon.shape == manual_rss.shape:
    diff = np.abs(rss_recon - manual_rss)
    axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title(f'Difference\nMax: {diff.max():.2e}')
    axes[1, 2].axis('off')
    
    if diff.max() < 1e-6:
        print("  ✓ Saved RSS matches manual RSS - reconstruction is correct!")
    else:
        print(f"  ❌ Saved RSS differs from manual RSS!")
        print(f"     Max difference: {diff.max():.2e}")
else:
    axes[1, 2].text(0.5, 0.5, 'Shape mismatch', ha='center', va='center')
    axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('diagnostic_reconstruction.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved diagnostic figure to: diagnostic_reconstruction.png")
plt.close()

# Final diagnosis
print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

if is_kspace_like(rss_recon)[0]:
    print("\n❌ PROBLEM CONFIRMED:")
    print("   The RSS reconstruction file contains K-SPACE data, not IMAGE data!")
    print("\n   This means the inverse FFT is NOT being applied in the reconstruction.")
    print("\n   Possible causes:")
    print("   1. The _rss_reconstruction method is not being called")
    print("   2. The method is being called but returns k-space")
    print("   3. The wrong variable is being saved")
    print("\n   Check your _rss_reconstruction implementation!")
else:
    print("\n✓ RSS reconstruction looks correct (it's an image, not k-space)")

print("="*70 + "\n")
