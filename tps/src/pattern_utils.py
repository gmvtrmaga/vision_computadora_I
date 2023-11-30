from cv2 import (
    COLOR_BGR2HSV,
    COLOR_BGR2RGB,
    COLOR_RGB2GRAY,
    IMREAD_GRAYSCALE,
    TM_CCOEFF_NORMED,
    Canny,
    GaussianBlur,
    cvtColor,
    imread,
    merge,
    split,
    resize,
    matchTemplate,
    rectangle,
    groupRectangles,
    minMaxLoc,
)
from numpy import float64, max, maximum, where


def tform_gauss_border_filter(img, kernel_size, th1, th2, contrast_factor=2.0):
    img_gray = cvtColor(img, COLOR_RGB2GRAY)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    img_gray = GaussianBlur(img_gray, (kernel_size, kernel_size), 0)
    img_canny = Canny(img_gray, threshold1=th1, threshold2=th2)
    return img_canny * contrast_factor


def detect_single_pattern(template, img, img_original, template_scale=1.0):
    # template_gray = cvtColor(template, COLOR_RGB2GRAY)
    # img_gray = cvtColor(img, COLOR_RGB2GRAY)
    template_gray = template
    img_gray = img

    # Template dimensions
    h, w = template_gray.shape

    # Rescale template if needed
    if template_scale != 1.0:
        h, w = int(h * template_scale), int(w * template_scale)
        template_gray = resize(template_gray, (h, w))

    # Apply template detection
    res = matchTemplate(img_gray, template_gray, TM_CCOEFF_NORMED)

    # Find the location of the best match
    _, _, _, max_loc = minMaxLoc(res)

    # Draw bounding box on the original image
    x, y = max_loc
    x2, y2 = x + w, y + h

    result_image = img_original.copy()
    result_image = rectangle(result_image, (x, y), (x2, y2), (0, 255, 0), 2)

    return result_image, [(x, y, x2, y2)]
