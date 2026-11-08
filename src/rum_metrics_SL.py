import numpy as np
from skimage.metrics import mean_squared_error, normalized_root_mse ,peak_signal_noise_ratio, structural_similarity


#=============== Metrics
def mse(img_true: np.ndarray, img_pred: np.ndarray) -> float:
    """Compute Mean Squared Error (MSE), scikit-image metric"""
    #return np.mean((gt - pred) ** 2)
    # if input array contains complex values
    return mean_squared_error(img_true,img_pred)

def nrmse(img_true: np.ndarray, img_pred: np.ndarray) -> float:
    """Compute Normalized Root Mean Squared Error (NMSE). 
    Normalization default euclidean’ : normalize by the averaged Euclidean norm of im_true scikit-image metric"""
    #return np.array(np.linalg.norm(img_true - img_pred) ** 2 / np.linalg.norm(img_true) ** 2)
    return normalized_root_mse(img_true, img_pred)

def psnr(img_true: np.ndarray, img_pred: np.ndarray, maxval: float = None) -> float:
    """Compute Peak Signal to Noise Ratio metric (PSNR) scikit-image metric"""
    if maxval is None:
        maxval = img_true.max()
    if not img_true.ndim == img_pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")
    return peak_signal_noise_ratio(img_true, img_pred, data_range=maxval)

def ssim(img_true: np.ndarray, img_pred: np.ndarray, maxval: float = None ) -> float:
    """Compute Structural Similarity Index Metric (SSIM) scikit-image metric"""
    if not img_true.ndim == img_pred.ndim:
        raise ValueError("Ground truth dimensions does not match img_pred.")
    maxval = img_true.max() if maxval is None else maxval
    return structural_similarity(img_true, img_pred, data_range=maxval)

"""METRIC_FUNCS = dict(
    MSE=mse,
    NRMSE=nrmse,
    PSNR=psnr,
    SSIM=ssim,
)"""

#=============== Calculate Metrics
def run_all_metrics(img_true, img_pred):
    """
    Run all 4 metric functions and save the results.
    
    Parameters:
    -----------
    img_true : np.ndarray
        Ground truth image(s)
    img_pred : np.ndarray
        Predicted/reconstructed image(s)
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

    """    # Create DataFrame for better formatting
    df = pd.DataFrame([results])    """

    # Print results
    print("\n" + "="*50)
    print("METRICS RESULTS")
    print("="*50)
    for metric, value in results.items():
        print(f"{metric:15s}: {value:.4f}")
    print("="*50 + "\n")
    
    return results