import numpy as np
import time

def fuji(imagen, cuadros, alto, ancho):
    tiempo_inicial = time.time()

    fuji = np.zeros((alto,ancho), np.float)

    for k in range(2,cuadros):  
        for i in range(0,alto):
            for j in range(0,ancho):

                pixel1 = int(imagen[k][i][j])
                pixel2 = int(imagen[k-1][i][j])
                #print("pixel " ,k,i,j, pixel1,pixel2 , "suma: ", pixel1+pixel2, "resta: ", pixel1-pixel2 ,int((imagen[k][i][j] + imagen[k+1][i][j])))
                if ((pixel1+pixel2)>0):
                    fuji[i][j] = fuji[i][j] + abs((pixel1-pixel2) / (pixel1+pixel2))

    min = np.amin(fuji)
    max = np.amax(fuji)
    fujisalida = (np.array((fuji/((max-min+1))*256)))
    fujisalida = fujisalida.astype(np.uint8)

    tiempo_final = time.time()
    print("tardo ", tiempo_final - tiempo_inicial, " segundos")

    return fujisalida

    