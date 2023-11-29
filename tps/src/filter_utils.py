from cv2 import getGaussianKernel
from numpy import fft, outer, real


def create_gauss_filter(height, width, k_size, sigma):
    # Create gaussian kernel
    gauss_kernel = getGaussianKernel(k_size, sigma)

    # Make kernel 2-D
    gauss_kernel = outer(gauss_kernel, gauss_kernel.T)

    # Normalization
    gauss_kernel /= gauss_kernel.sum()

    # Get fourier transformation of the kernel matching the img size
    return gauss_kernel, img_to_fourier(gauss_kernel, height, width)


def img_to_fourier(img, height, width):
    # Get fourier transformation of the img
    img_fft = fft.fft2(img, s=(height, width))
    return fft.fftshift(img_fft)


def fourier_to_img(img):
    # Inverse functions
    f_ishift = fft.ifftshift(img)
    return real(fft.ifft2(f_ishift))
