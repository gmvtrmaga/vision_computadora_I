from cv2 import (
    COLOR_BGR2HSV,
    COLOR_BGR2RGB,
    COLOR_RGB2GRAY,
    FONT_HERSHEY_SIMPLEX,
    IMREAD_GRAYSCALE,
    TM_CCOEFF_NORMED,
    Canny,
    GaussianBlur,
    cvtColor,
    getTextSize,
    imread,
    merge,
    putText,
    split,
    resize,
    matchTemplate,
    rectangle,
    groupRectangles,
    minMaxLoc,
)
from numpy import clip, float64, max, maximum, uint8, where


def tform_gauss_border_filter(img, kernel_size, th1, th2, contrast_factor=2.0):
    img_gray = cvtColor(img, COLOR_RGB2GRAY)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    img_gray = GaussianBlur(img_gray, (kernel_size, kernel_size), 0)
    img_canny = Canny(img_gray, threshold1=th1, threshold2=th2)

    return clip(img_canny * contrast_factor, 0, 255).astype(uint8)


def detect_single_pattern(template, img, template_scale=1.0):
    # Rescale template if needed
    if template_scale != 1.0:
        template = resize(template, None, fx=template_scale, fy=template_scale)

    # Template dimensions
    h, w = template.shape

    # Apply template detection
    res = matchTemplate(img, template, TM_CCOEFF_NORMED)

    # Find the location of the best match
    _, max_corr, _, max_loc = minMaxLoc(res)

    # Draw bounding box on the original image
    x, y = max_loc
    x2, y2 = x + w, y + h

    return [(x, y, x2, y2)], [max_corr]


def paint_detections(img, rectangles, corrs, border_size=2):
    TEXT_OFFSET = 3
    ret = img.copy()
    nItems = min(len(rectangles), len(corrs))
    for i in range(nItems):
        x, y, x2, y2 = rectangles[i]
        p_corr = f"{round(corrs[i] * 100, 1)}%"
        (label_width, label_height), _ = getTextSize(p_corr, FONT_HERSHEY_SIMPLEX, 1, 2)
        text_end = (x + label_width, y - label_height - 2 * border_size)

        ret = rectangle(ret, (x, y), (x2, y2), (0, 255, 0), border_size)
        ret = rectangle(ret, (x, y), text_end, (0, 255, 0), -1)
        ret = rectangle(ret, (x, y), text_end, (0, 255, 0), border_size)

        ret = putText(
            ret,
            p_corr,
            (x - TEXT_OFFSET, y - TEXT_OFFSET),
            FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )

    return ret
