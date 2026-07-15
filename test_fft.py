"""
Simple test to verify FFT operations are working correctly
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("TEST: Verify FFT operations")
print("="*70)

# Create test k-space data (20 coils, 320x320)
print("\n1. Creating test k-space...")
n_coils = 20
img_size = 320

# Create a simple phantom image
y, x = np.ogrid[-img_size//2:img_size//2, -img_size//2:img_size//2]
phantom = ((x**2 + y**2) < (img_size//2 - 20)**2).astype(float)
phantom += 0.5 * ((x**2 + y**2) < (img_size//2 - 40)**2).astype(float)

# Create multi-coil images (just replicate for simplicity)
img_coils = np.zeros((n_coils, img_size, img_size), dtype=complex)
for i in range(n_coils):
    img_coils[i] = phantom * (0.8 + 0.2 * np.random.rand())

# Forward FFT to create k-space
kspace = np.fft.fftshift(
    np.fft.fft2(np.fft.ifftshift(img_coils, axes=(-2, -1)), axes=(-2, -1)),
    axes=(-2, -1)
)

print(f"  K-space shape: {kspace.shape}")
print(f"  K-space dtype: {kspace.dtype}")
print(f"  K-space is complex: {np.iscomplexobj(kspace)}")

# Test inverse FFT (what _rss_reconstruction does)
print("\n2. Testing inverse FFT (RSS reconstruction)...")
img_coils_recon = np.fft.ifftshift(
    np.fft.ifft2(np.fft.fftshift(kspace, axes=(-2, -1)), axes=(-2, -1)),
    axes=(-2, -1)
)
rss_img = np.sqrt(np.sum(np.abs(img_coils_recon)**2, axis=0))

print(f"  Reconstructed image shape: {rss_img.shape}")
print(f"  Reconstructed image dtype: {rss_img.dtype}")
print(f"  Reconstructed image range: [{rss_img.min():.3f}, {rss_img.max():.3f}]")

# Check if it looks like k-space or image
def check_if_kspace(data):
    center = data[data.shape[0]//2-10:data.shape[0]//2+10,
                 data.shape[1]//2-10:data.shape[1]//2+10]
    center_mean = np.mean(np.abs(center))
    total_mean = np.mean(np.abs(data))
    ratio = center_mean / (total_mean + 1e-10)
    return ratio > 5.0, ratio

is_ks, ratio = check_if_kspace(rss_img)
print(f"  Center/Mean ratio: {ratio:.2f}")
if is_ks:
    print(f"  ❌ Result looks like K-SPACE!")
else:
    print(f"  ✓ Result looks like IMAGE")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original phantom
axes[0].imshow(phantom, cmap='gray')
axes[0].set_title('Original Phantom')
axes[0].axis('off')

# K-space
axes[1].imshow(np.log(1 + np.abs(kspace[0])), cmap='gray')
axes[1].set_title('K-space (log scale)\nShould have bright center')
axes[1].axis('off')

# Reconstructed
axes[2].imshow(rss_img, cmap='gray')
axes[2].set_title(f'RSS Reconstruction\nRatio: {ratio:.1f}')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('test_fft.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved test figure to: test_fft.png")
plt.close()

# Now test with YOUR data if available
print("\n" + "="*70)
print("3. Testing with your actual data (if available)...")
print("="*70)

try:
    your_kspace = np.load('experiments/data/cartesian_4x_numpy/kspace.npy')
    print(f"\n  Loaded your k-space:")
    print(f"    Shape: {your_kspace.shape}")
    print(f"    Dtype: {your_kspace.dtype}")
    print(f"    Is complex: {np.iscomplexobj(your_kspace)}")
    
    # Apply same reconstruction
    print(f"\n  Applying RSS reconstruction...")
    img_coils_test = np.fft.ifftshift(
        np.fft.ifft2(np.fft.fftshift(your_kspace, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1)
    )
    rss_test = np.sqrt(np.sum(np.abs(img_coils_test)**2, axis=0))
    
    is_ks, ratio = check_if_kspace(rss_test)
    print(f"    Result shape: {rss_test.shape}")
    print(f"    Result range: [{rss_test.min():.3e}, {rss_test.max():.3e}]")
    print(f"    Center/Mean ratio: {ratio:.2f}")
    
    if is_ks:
        print(f"    ❌ Result looks like K-SPACE!")
        print(f"\n    This suggests the FFT operations are NOT working correctly!")
        print(f"    Check:")
        print(f"      - Is your k-space properly fftshifted?")
        print(f"      - Are the axes correct?")
    else:
        print(f"    ✓ Result looks like IMAGE")
        print(f"\n    This means the FFT is working correctly!")
        print(f"    The problem must be elsewhere in the pipeline.")
    
    # Compare with saved RSS
    try:
        saved_rss = np.load('experiments/data/cartesian_4x_numpy/reconstructions/rss.npy')
        print(f"\n  Comparing with saved RSS:")
        print(f"    Saved RSS shape: {saved_rss.shape}")
        print(f"    Saved RSS range: [{saved_rss.min():.3e}, {saved_rss.max():.3e}]")
        
        if saved_rss.shape == rss_test.shape:
            diff = np.abs(saved_rss - rss_test)
            print(f"    Max difference: {diff.max():.3e}")
            
            if diff.max() < 1e-6:
                print(f"    ✓ Saved RSS matches our test - reconstruction is correct!")
            else:
                print(f"    ❌ Saved RSS differs from our test!")
                print(f"       Something is wrong in the saving process!")
        else:
            print(f"    ❌ Shape mismatch!")
            print(f"       Test: {rss_test.shape}")
            print(f"       Saved: {saved_rss.shape}")
    except FileNotFoundError:
        print(f"  (Saved RSS file not found - skipping comparison)")
        
except FileNotFoundError:
    print("\n  Your data files not found - skipping test")
    print("  Run the pipeline first, then run this diagnostic")

print("\n" + "="*70)
print("Summary:")
print("="*70)
print("""
If the FFT test above worked correctly (image, not k-space),
but your saved RSS file contains k-space, then the problem is in:

1. How the pipeline saves the data, OR
2. The reconstruction method returns k-space instead of image

Next step: Add print statements in _rss_reconstruction to see
what it's actually returning!
""")
print("="*70 + "\n")
