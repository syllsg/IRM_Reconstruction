import matplotlib.pyplot as plt
import numpy as np


#=============== Fourier Transform
def inverse_fft2_shift_1slice(kspace):
    """Function that takes kspace data from 1 slice as input and output the corresponding image.
    This function transforms the k-space array from the frequency domain back to the spatial domain, apply the 'ortho' normalization factor, and return the image.
    :param kspace: shape(number of coils, height, width)
    :return: Images corresponding to k-space with same shape as k-space (number of coils, height, width)
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2,-1)), norm='ortho'),axes=(-2,-1))


def inverse_fft2_shift_all_splices(mri_data):
    """
    Function that applys the inverse_fft2_shift_1slice function to each slice along the 0th dimension of the MRI data and return an array of images with the same shape.
    it prints the shape of the returned array.
    Each image in the resulting array is the result of inverse FFT and shift applied to the corresponding slice in the original 3D image data.
    
    :param mri_data: k-space with the following shape: (number of slices, number of coils, height, width)
    :return: Images corresponding to k-space with same shape as k-space (number of slices, number of coils, height, width)
    :rtype: Array
    """
    # Initialize an empty list to store the images
    images = []
    # Iterate over each slice along the 0th dimension
    for i in range(mri_data.shape[0]):
        # Apply the inverse FFT and shift functions to each slice
        reconstructed_image = inverse_fft2_shift_1slice(mri_data[i])
        # Append the reconstructed image to the list
        images.append(reconstructed_image)

    # Convert the list of images into a numpy array
    reconstructed_images = np.array(images)

    # Check the shape of the resulting array
    print(reconstructed_images.shape)  # Should print (16, 20, 640, 320)
    return reconstructed_images


#=============== Plot
def show_grid(data, coil_indices=None, cmap='viridis', set_vmax= None,nrows=4, ncols=5, type ="Coil" ,title = 'Signal from the 20 coils on 1 middle slice',fig_size=(12,9)):
    """
    This function plots multi-coil data for a single slice for each coil individually or different slices of MRI data. The default type is coil but you can change it to slice
    The first dimension of the data must contains the slices or coils that you want to plot separately. ex : (16, 320, 320). If not specify it with coil_indices
    Default values for the figure can be used to plot 20 sublplots. 
    If data is k-space, then you should apply: np.log(np.abs(k_space_1_slice) + 1e-9) to get and modify magnitude before plotting. Note that a small constant is added for numerical stability.
    If data is images, you should apply : np.abs(image_1_slice) to get the magnitude values before plotting.
    This function can be used also to plot the different slices of an exam.
    set_vmax allows to modify the calculated color scaling. If vmin or set_vmax is None, the image max will be used for color scaling.
    
    :param data: k-space or corresponding image from multi-coil. 3D array with coils or slices in 1st dimension. 
    :param coil_indices: Description
    :param cmap: Description
    :param nrows: Description
    :param ncols: Description
    :param fig_size: Description
    :return: Description
    :rtype: Any
    example : show_grid(np.abs(image_8th_slice), coil_indices=[i for i in range(0, 20)], cmap='gray', nrows=4, ncols=5, fig_size=(10,8))
    """
    if coil_indices is None:
        coil_indices = list(range(data.shape[0]))
    else:
        coil_indices = list(range(data.shape[coil_indices]))
    fig, axes = plt.subplots(nrows, ncols, figsize=fig_size)
    axes = axes.flatten()
    # consistent color scaling across all coils
    vals = np.array([data[i] for i in coil_indices])
    vmin = vals.min()
    vmax = vals.max()
    im = None
    for ax, idx in zip(axes, coil_indices):
        if set_vmax == None:
            im = ax.imshow(data[idx], cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        else:
            im = ax.imshow(data[idx], cmap=cmap, vmin=vmin, vmax=set_vmax, interpolation='nearest')
        ax.set_title(f'{type} {idx}', fontsize=9)
        ax.axis('off')
    # turn off any unused axes
    for ax in axes[len(coil_indices):]:
        ax.axis('off')
    fig.tight_layout(rect=[0,0,0.92,0.96])
    fig.suptitle(title, fontsize=14)

   
    

#=============== Multi-coil combination
def rsos(x, dim):
    """
    Compute the root-sum-of-squares (RSS) along the coil axis to combine multi coil signal.

    Parameters
    ----------
    x : array_like; Input array. Can be real or complex. RSS is computed over magnitudes.
    dim : int or tuple of ints; Axis or axes along which to compute the RSS (passed to ``np.sum`` as the ``axis`` argument). coils dimension. 
    Returns
    -------
    ndarray
        Array with the specified axis/axes removed. Each element is the square root of the sum of squared magnitudes along ``dim``.

    Notes
    -----
    This function is commonly used in MRI to combine multi-coil images into a single image by computing the pixel-wise root-sum-of-squares across the
    coil dimension.  For example, if x has shape (nx, ny, n_coils) and you pass dim=2, the result has shape (nx, ny) — one combined image.

    Examples
    --------
    with image_8th_slice.shape = (20, 640, 320)
    rsos(image_8th_slice, 0)
    """
    return np.sqrt(np.sum(np.square(np.abs(x)), dim))



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

