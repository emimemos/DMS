import cv2 as cv
import numpy as np
import time

def float_to_int(x):
    if x == float('inf') or x == float('-inf'):
        return 0 # or a large value you choose
    return int(x)


def obtenerParametrosIniciales():
    image=[]
    #en el rango van los frames que queres usar 
    for i in range(1,100): 
    	image.append(cv.imread('25min/%i.BMP' %i, cv.IMREAD_GRAYSCALE))
    imagen=np.array(image)

    cuadros = len(imagen)
    alto = len(imagen[0])
    ancho = len(imagen[0][0])

    return imagen, cuadros, alto, ancho

    
