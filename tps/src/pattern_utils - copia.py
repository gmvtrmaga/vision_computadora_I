from cv2 import (
    COLOR_BGR2HSV,
    COLOR_BGR2RGB,
    COLOR_RGB2GRAY,
    IMREAD_GRAYSCALE,
    TM_CCOEFF_NORMED,
    THRESH_BINARY,
    CV_64F,
    cvtColor,
    imread,
    merge,
    split,
    resize,
    matchTemplate,
    Canny,
    rectangle,
    groupRectangles,
    threshold,
    Sobel,
)
from numpy import float64, max, maximum, where, abs


# def border_filter(img):
#     img_gray = cvtColor(img, COLOR_RGB2GRAY)
#     return Canny(img_gray, threshold1=5, threshold2=200)
#     # img_gray = cvtColor(img, COLOR_RGB2GRAY)
#     # img_sobel = Sobel(img_gray, CV_64F, 1, 1, ksize=3)
#     # _, ret = threshold(abs(img_sobel), 0.1, 255, THRESH_BINARY)
#     # return ret


# def detect_pattern(template, img, template_scale=1.0, threshold=0.75):
#     aux_template = template.copy()
#     # Template dimensions
#     h, w = aux_template.shape

#     # Rescale template if needed
#     if template_scale != 1.0:
#         h, w = int(h * template_scale), int(w * template_scale)
#         aux_template = resize(aux_template, (h, w))

#     # Apply template detection
#     res = matchTemplate(img, aux_template, TM_CCOEFF_NORMED)
#     loc = where(res >= threshold)

#     # Filtrar detecciones duplicadas
#     rectangles = []
#     for pt in zip(*loc[::-1]):
#         rectangles.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
#     rectangles, weights = groupRectangles(rectangles, 1, 0.5)

#     # Dibujar bounding boxes en la imagen original
#     result_image = img.copy()
#     for x, y, x2, y2 in rectangles:
#         result_image = rectangle(result_image, (x, y), (x2, y2), (0, 255, 0), 2)

#     return result_image, rectangles


def detect_pattern(template, img, template_scale=1.0, threshold=0.75):
    aux_template = template.copy()
    # Template dimensions
    h, w = aux_template.shape

    # Rescale template if needed
    if template_scale != 1.0:
        h, w = int(h * template_scale), int(w * template_scale)
        aux_template = resize(aux_template, (h, w))

    # Convertir la imagen y la plantilla a escala de grises
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)

    # Realizar la coincidencia de plantillas con la imagen
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Encontrar las ubicaciones donde la correlación es mayor que el umbral
    loc = np.where(res >= threshold)

    # Dibujar bounding boxes en la imagen original
    result_image = img.copy()
    for pt in zip(*loc[::-1]):
        cv2.rectangle(result_image, pt, (pt[0] + new_w, pt[1] + new_h), (0, 255, 0), 2)

    return result_image


# Ejemplo de uso
template_path = "path/to/template.png"
image_path = "path/to/image.png"

# Cargar la plantilla y la imagen
template = cv2.imread(template_path)
image = cv2.imread(image_path)

# Llamar a la función detect_pattern
result_image = detect_pattern(template, image, template_scale=1.0, threshold=0.75)

# Visualizar resultados
cv2.imshow("Result", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
