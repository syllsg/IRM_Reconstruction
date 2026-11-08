# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#%% imagesc def
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def imagesc( img, fig_num, clim, title = '' ):
      
    fig = plt.figure(fig_num)
    fig.patch.set_facecolor('black')

    plt.imshow( img )
    plt.gray()
    plt.clim(clim)
    plt.suptitle(title, color='white', fontsize=48)


 
def mosaic(img, num_row, num_col, fig_num, clim, title = '', use_transpose = False, use_flipud = False):
    """
    Create and display a mosaic of image tiles using matplotlib.

    Parameters
    ----------
    img : ndarray
        If 2-D, the image is displayed directly. If 3-D, the expected shape
        is (H, W, N) where N == ``num_row * num_col``; the function arranges
        the N tiles into a single mosaic image in row-major order (tiles are
        taken from the third axis).
    num_row : int
        Number of rows in the mosaic.
    num_col : int
        Number of columns in the mosaic.
    fig_num : int
        Matplotlib figure number to create/use for the display.
    clim : tuple or list
        Color limits passed to ``plt.clim`` (e.g., ``(vmin, vmax)``).
    title : str, optional
        Figure title (default: empty string).
    use_transpose : bool, optional
        If True, each tile will be transposed before placing it in the
        mosaic. Note: when True the function modifies the input tiles in
        place for each slice (the transposition assignment is done
        in-place).
    use_flipud : bool, optional
        If True, the (assembled) image is flipped upside-down using
        ``np.flipud`` before displaying.

    Returns
    -------
    None
        The function displays the mosaic using matplotlib and does not
        return a value.

    Notes
    -----
    - If ``img.ndim < 3``, the image is displayed directly and the mosaic
      layout parameters are ignored.
    - If the number of tiles in the third axis does not equal
      ``num_row * num_col``, the function prints a warning message and
      does not construct the mosaic.

    Example
    -------
    >>> tiles = np.random.rand(64, 64, 9)
    >>> mosaic(tiles, num_row=3, num_col=3, fig_num=1, clim=(0, 1), title='Example')
    """

    fig = plt.figure(fig_num)
    fig.patch.set_facecolor('black')

    if img.ndim < 3:
        img_res = img
        plt.imshow(img_res)
        plt.gray()        
        plt.clim(clim)

    else:
        
        if img.shape[2] != (num_row * num_col):
            print('sizes do not match')    
        else:   
            
            if use_transpose:
                for slc in range(0, img.shape[2]):
                    img[:,:,slc] = np.transpose(img[:,:,slc])
            
            if use_flipud:
                img = np.flipud(img)
                
            
            img_res = np.zeros((img.shape[0]*num_row, img.shape[1]*num_col))
            
            idx = 0
            
            for r in range(0, num_row):
                for c in range(0, num_col):
                    img_res[r*img.shape[0] : (r+1)*img.shape[0], c*img.shape[1] : (c+1)*img.shape[1]] = img[:,:,idx]
                    idx = idx + 1
                
      
                
        plt.imshow(img_res)
        plt.gray()        
        plt.clim(clim)

        
    plt.suptitle(title, color='white', fontsize=48)
    
    
    

def imagesc3d( img, fig_num, scl, title = '' ):

    pos = np.divide( img.shape, 2. ).astype('int')
    
    plt.figure(fig_num)
    plt.subplot(1,3,1)
    imagesc(np.transpose(np.abs(img[:,:,pos[2]])), fig_num, scl)

    plt.subplot(1,3,2)
    imagesc(np.flip(np.transpose(np.abs(img[:,pos[1],:])), 0), fig_num, scl)

    plt.subplot(1,3,3)
    imagesc(np.flip(np.transpose(np.abs(img[pos[0],:,:])), 0), fig_num, scl, title)
    
    
    
    
def imagesc3d0( img, fig_num, scl, title = '' ):

    pos = np.divide( img.shape, 2. ).astype('int')
    
    plt.figure(fig_num)
    plt.subplot(1,3,1)
    imagesc(np.transpose(np.abs(img[:,:,pos[2]])), fig_num, scl)

    plt.subplot(1,3,2)
    imagesc(np.abs(img[:,pos[1],:]), fig_num, scl)

    plt.subplot(1,3,3)
    imagesc(np.abs(img[pos[0],:,:]), fig_num, scl, title)
    
    
    
def imagesc3d1( img, fig_num, scl, title = '', pos = [0,0,0] ):
  
    plt.figure(fig_num)
    plt.subplot(1,3,1)
    imagesc(np.transpose(np.abs(img[:,:,pos[2]])), fig_num, scl)

    plt.subplot(1,3,2)
    imagesc(np.abs(img[:,pos[1],:]), fig_num, scl)

    plt.subplot(1,3,3)
    imagesc(np.abs(img[pos[0],:,:]), fig_num, scl, title)
    
    
def imagesc3d2( img, fig_num, scl, title = '', pos = [0,0,0] ):
  
    plt.figure(fig_num)
    plt.subplot(1,3,1)
    imagesc(np.transpose(img[:,:,pos[2]]), fig_num, scl)

    plt.subplot(1,3,2)
    imagesc(img[:,pos[1],:], fig_num, scl)

    plt.subplot(1,3,3)
    imagesc(img[pos[0],:,:], fig_num, scl, title)
    
    

  
def imagesc3d3( img, fig_num, scl, title = '', pos = [0,0,0], rot=[0,0,0] ):
  
    import scipy.ndimage.interpolation as sni

    plt.figure(fig_num)
    plt.subplot(1,3,1)
    imagesc( sni.rotate(img[:,:,pos[2]], rot[0], axes=(0,1), reshape=True, order=5), fig_num, scl)

    plt.subplot(1,3,2)
    imagesc( sni.rotate(img[:,pos[1],:], rot[1], axes=(0,1), reshape=True, order=5), fig_num, scl)

    plt.subplot(1,3,3)
    imagesc( sni.rotate(img[pos[0],:,:], rot[1], axes=(0,1), reshape=True, order=5), fig_num, scl)    


    
    
def rsos(x, dim):
    """
    Compute the root-sum-of-squares (RSS) along a specified axis.

    Parameters
    ----------
    x : array_like
        Input array. Can be real or complex. RSS is computed over magnitudes.
    dim : int or tuple of ints
        Axis or axes along which to compute the RSS (passed to ``np.sum`` as
        the ``axis`` argument).

    Returns
    -------
    ndarray
        Array with the specified axis/axes removed. Each element is the
        square root of the sum of squared magnitudes along ``dim``.

    Notes
    -----
    This function is commonly used in MRI to combine multi-coil images into
    a single image by computing the pixel-wise root-sum-of-squares across the
    coil dimension.  For example, if x has shape (nx, ny, n_coils) and you pass dim=2, the result has shape (nx, ny) — one combined image.

    Examples
    --------
    >>> x = np.array([1+1j, 2+2j])
    >>> rsos(x, 0)
    3.1622776601683795
    """
    return np.sqrt(np.sum(np.square(np.abs(x)), dim))

    

def ifft2call(data, dim1, dim2):
    """perform 2-D inverse discrete Fourier Transform. Takes as arguments : kspace data as numpy array (? number of dimensions) and  two axes the transform is computed over   """
    shifted_rows = np.fft.ifftshift(data, axes=dim1)
    shifted_rows = np.fft.ifft(shifted_rows, axis=dim1)
    shifted_rows = np.fft.fftshift(shifted_rows, axes=dim1)


    shifted_rows = np.fft.ifftshift(shifted_rows, axes=dim2)
    shifted_rows = np.fft.ifft(shifted_rows, axis=dim2)
    shifted_rows = np.fft.fftshift(shifted_rows, axes=dim2)

    shifted_rows = shifted_rows * np.sqrt(shifted_rows.shape[dim1] * shifted_rows.shape[dim2])

    return shifted_rows


def inverse_fft2_shift(kspace):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2,-1)), norm='ortho'),axes=(-2,-1))


def fft2call(data, dim1, dim2):
    """
    Perform a 2-D discrete Fourier transform on `data` across two axes.

    This function applies an FFT along `dim1` and `dim2` with pre- and
    post-shifts so the input is treated as centered k-space. The result is
    normalized by dividing by sqrt(N1 * N2) where N1 and N2 are the sizes of
    the transformed axes. This normalization makes the forward/backward
    transforms unitary when paired with the corresponding normalization used
    in `ifft2call` (which multiplies by the same factor).

    Parameters
    ----------
    data : array_like
        Input array (e.g., image or k-space) to transform.
    dim1 : int
        First axis to transform (passed to numpy FFT as `axis`).
    dim2 : int
        Second axis to transform (passed to numpy FFT as `axis`).

    Returns
    -------
    ndarray
        The transformed array with the same shape as `data`.

    Notes
    -----
    - The function uses `np.fft.ifftshift/fft/fftshift` sequence for each
      axis so that the zero-frequency component is correctly handled for
      data that is centered.
    - The normalization by `sqrt(N1 * N2)` yields a unitary-style transform
      when combined with `ifft2call` which multiplies by the same factor.

    Example
    -------
    >>> img = np.zeros((4,4))
    >>> out = fft2call(img, 0, 1)

    """
    shifted_rows = np.fft.ifftshift(data, axes=dim1)
    shifted_rows = np.fft.fft(shifted_rows, axis=dim1)
    shifted_rows = np.fft.fftshift(shifted_rows, axes=dim1)


    shifted_rows = np.fft.ifftshift(shifted_rows, axes=dim2)
    shifted_rows = np.fft.fft(shifted_rows, axis=dim2)
    shifted_rows = np.fft.fftshift(shifted_rows, axes=dim2)

    shifted_rows = shifted_rows / np.sqrt(shifted_rows.shape[dim1] * shifted_rows.shape[dim2])

    return shifted_rows


def nrmse(img1, img2):
    """Compute the normalized root mean squared error (NRMSE) between two images.

    The function computes the root-mean-square error (RMSE) between ``img1``
    and ``img2`` using the magnitude for complex inputs, and normalizes the
    RMSE by the root-mean-square (RMS) of ``img2``. The returned value is
    expressed as a percentage::

        NRMSE = 100 * RMSE / RMS(img2)

    where::

        RMSE = sqrt(mean(|img1 - img2|^2))
        RMS(img2) = sqrt(mean(|img2|^2))

    Parameters
    ----------
    img1 : array_like
        Reconstructed or test image. May be real or complex. Must be
        broadcastable to ``img2``.
    img2 : array_like
        Reference or ground-truth image. May be real or complex.

    Returns
    -------
    float
        NRMSE value in percent. If the RMS of ``img2`` is zero the result
        will follow NumPy's division semantics (``inf`` or ``nan``).

    Notes
    -----
    - The function compares magnitudes when inputs are complex (uses
      ``np.abs``).
    - The output is scaled by 100 to present the error as a percentage.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([0., 1., 2.])
    >>> b = np.array([0., 1., 3.])
    >>> nrmse(a, b)
    18.257418583505537
    """
    # Compute the RMSE
    rmse = np.sqrt(np.mean(np.abs(img1 - img2) ** 2))

    # Compute the RMS of img2
    mean_img2 = np.sqrt(np.mean(np.abs(img2) ** 2))

    # Compute the NRMSE (percentage)
    nrmse_value = 100 * rmse / mean_img2

    return nrmse_value


def show_grid(data, coil_indices=None, cmap='viridis', nrows=4, ncols=5, fig_size=(12,9)):
    if coil_indices is None:
        coil_indices = list(range(data.shape[0]))
    fig, axes = plt.subplots(nrows, ncols, figsize=fig_size)
    axes = axes.flatten()
    # consistent color scaling across all coils
    vals = np.array([data[i] for i in coil_indices])
    vmin = vals.min()
    vmax = vals.max()
    im = None
    for ax, idx in zip(axes, coil_indices):
        im = ax.imshow(data[idx], cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(f'Coil {idx}', fontsize=9)
        ax.axis('off')
    # turn off any unused axes
    for ax in axes[len(coil_indices):]:
        ax.axis('off')
    #fig.tight_layout(rect=[0,0,0.92,0.96])
    fig.suptitle('Signal from the 20 coils on 1 middle slice', fontsize=14)