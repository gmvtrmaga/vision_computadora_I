from cv2 import (
    COLOR_BGR2HSV,
    COLOR_BGR2RGB,
    COLOR_RGB2GRAY,
    IMREAD_GRAYSCALE,
    TM_CCOEFF_NORMED,
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


def detect_single_pattern(template, img, template_scale=1.0):
    template_gray = cvtColor(template, COLOR_RGB2GRAY)
    img_gray = cvtColor(img, COLOR_RGB2GRAY)

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
    result_image = img.copy()
    result_image = rectangle(result_image, (x, y), (x2, y2), (0, 255, 0), 2)

    return result_image, [(x, y, x2, y2)]
