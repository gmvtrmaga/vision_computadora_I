from cv2 import imread, cvtColor, COLOR_BGR2RGB

def cargar_en_RGB(path:str):
    ret = imread(path)
    return cvtColor(ret, COLOR_BGR2RGB)