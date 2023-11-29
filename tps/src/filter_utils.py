from cv2 import getGaussianKernel
from numpy import outer, fft


def create_gauss_filter(height, width, k_size, sigma):
    # Create gaussian kernel
    gauss_kernel = getGaussianKernel(k_size, sigma)

    # Make kernel 2-D
    gauss_kernel = outer(gauss_kernel, gauss_kernel.T)

    # Normalization
    gauss_kernel /= gauss_kernel.sum()

    # Get fourier transformation of the kernel matching the img size
    gauss_fft = fft.fft2(gauss_kernel, s=(height, width))

    return gauss_kernel, gauss_fft
