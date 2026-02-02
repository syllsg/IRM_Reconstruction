import numpy as np
import pandas as pd
import json
import os

# Assuming your metric functions are imported from your main file
# Adjust the import based on your actual file structure
# from your_file import mse, nrmse, psnr, ssim

def run_all_metrics(img_true, img_pred, save_path='metrics_results.csv'):
    """
    Run all 4 metric functions and save the results.
    
    Parameters:
    -----------
    img_true : np.ndarray
        Ground truth image(s) - will convert to magnitude if complex
    img_pred : np.ndarray
        Predicted/reconstructed image(s) - will convert to magnitude if complex
    save_path : str
        Path to save the results CSV file
    
    Returns:
    --------
    dict : Dictionary containing all metric scores
    """
    
    # Convert complex images to magnitude
    if np.iscomplexobj(img_true):
        print("Converting ground truth image to magnitude (absolute value)...")
        img_true = np.abs(img_true)
    
    if np.iscomplexobj(img_pred):
        print("Converting predicted image to magnitude (absolute value)...")
        img_pred = np.abs(img_pred)
    
    results = {}
    
    # Calculate MSE
    print("Calculating MSE...")
    results['MSE'] = mse(img_true, img_pred)
    
    # Calculate NRMSE
    print("Calculating NRMSE...")
    results['NRMSE'] = nrmse(img_true, img_pred)
    
    # Calculate PSNR
    print("Calculating PSNR...")
    results['PSNR'] = psnr(img_true, img_pred)
    
    # Calculate SSIM
    print("Calculating SSIM...")
    results['SSIM'] = ssim(img_true, img_pred)
    
    # Create DataFrame for better formatting
    df = pd.DataFrame([results])
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"\nResults saved to: {save_path}")
    
    # Also save as JSON for easy reading
    json_path = save_path.replace('.csv', '.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results also saved to: {json_path}")
    
    # Print results
    print("\n" + "="*50)
    print("METRICS RESULTS")
    print("="*50)
    for metric, value in results.items():
        print(f"{metric:15s}: {value:.6f}")
    print("="*50 + "\n")
    
    return results


def run_metrics_batch(true_images, pred_images, save_dir='metrics_output'):
    """
    Run metrics on multiple image pairs and save results.
    
    Parameters:
    -----------
    true_images : list of np.ndarray
        List of ground truth images
    pred_images : list of np.ndarray
        List of predicted images
    save_dir : str
        Directory to save results
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = []
    
    for idx, (true_img, pred_img) in enumerate(zip(true_images, pred_images)):
        print(f"\nProcessing image pair {idx + 1}/{len(true_images)}...")
        
        # Convert complex images to magnitude
        if np.iscomplexobj(true_img):
            true_img = np.abs(true_img)
        if np.iscomplexobj(pred_img):
            pred_img = np.abs(pred_img)
        
        results = {
            'image_id': idx,
            'MSE': mse(true_img, pred_img),
            'NRMSE': nrmse(true_img, pred_img),
            'PSNR': psnr(true_img, pred_img),
            'SSIM': ssim(true_img, pred_img)
        }
        
        all_results.append(results)
    
    # Create DataFrame with all results
    df = pd.DataFrame(all_results)
    
    # Save individual results
    csv_path = f"{save_dir}/all_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nAll results saved to: {csv_path}")
    
    # Calculate and save summary statistics
    summary = df.describe()
    summary_path = f"{save_dir}/metrics_summary.csv"
    summary.to_csv(summary_path)
    print(f"Summary statistics saved to: {summary_path}")
    
    return df


# Example usage:
if __name__ == "__main__":
    
    # Example 1: Single image pair
    # Load your actual data here
    # img_true = np.load('path/to/true_image.npy')
    # img_pred = np.load('path/to/predicted_image.npy')
    
    # For demonstration with dummy data:
    img_true = np.random.rand(256, 256)
    img_pred = img_true + np.random.randn(256, 256) * 0.1  # Add some noise
    
    # Run metrics on single pair
    results = run_all_metrics(img_true, img_pred, save_path='single_metrics.csv')
    
    
    # Example 2: Batch processing multiple images
    # true_images = [np.load(f'true_{i}.npy') for i in range(10)]
    # pred_images = [np.load(f'pred_{i}.npy') for i in range(10)]
    
    # For demonstration:
    # true_images = [np.random.rand(256, 256) for _ in range(5)]
    # pred_images = [img + np.random.randn(256, 256) * 0.1 for img in true_images]
    
    # df_results = run_metrics_batch(true_images, pred_images, save_dir='batch_metrics')
