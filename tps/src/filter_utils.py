from cv2 import getGaussianKernel
from numpy import abs, fft, histogram, max, mean, outer, real, sum


def create_gauss_filter(height, width, k_size, sigma):
    # Create gaussian kernel
    gauss_kernel = getGaussianKernel(k_size, sigma)

    # Make kernel 2-D
    gauss_kernel = outer(gauss_kernel, gauss_kernel.T)

    # Normalization
    gauss_kernel /= gauss_kernel.sum()

    # Get fourier transformation of the kernel matching the img size
    return gauss_kernel, img_to_shifted_fourier(gauss_kernel, height, width)


def img_to_fourier(img, height, width):
    # Get fourier transformation of the img
    return fft.fft2(img, s=(height, width))


def img_to_shifted_fourier(img, height, width):
    # Get fourier transformation of the img
    img_fft = fft.fft2(img, s=(height, width))
    return fft.fftshift(img_fft)


def shifted_fourier_to_img(img):
    # Inverse functions
    f_ishift = fft.ifftshift(img)
    return real(fft.ifft2(f_ishift))


def sharpness_method_quality_measure(img):
    height, width = img.shape

    img_fft_no_shift = img_to_fourier(img, height, width)
    img_fft = img_to_shifted_fourier(img, height, width)
    abs_img_fft = abs(img_fft)
    max_freq_val = max(abs_img_fft)

    threshold = max_freq_val / 1000

    pixel_count = sum(img_fft_no_shift > threshold)

    return pixel_count / (height * width)


def absolute_central_moment_quality_measure(img):
    # Get img histogram
    hist, bins = histogram(img.ravel(), 256, [0, 256])

    # Mu
    meanGray = mean(img)

    # ACMo
    ret = 0
    for i in range(len(hist)):
        ret += abs(bins[i] - meanGray) * hist[i]

    # Make the mean value
    nPix = sum(hist)

    return ret / nPix
