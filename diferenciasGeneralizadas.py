import numpy as np
import time

def difGen(imagen, cuadros, alto, ancho): 

	tiempo_inicial = time.time()

	ventana = 5

	DG	= np.zeros((alto,ancho), np.float)
	DGS = np.zeros((alto,ancho), np.uint8)

	for k in range(0,cuadros-ventana):
		for i in range(0,alto):
			for j in range(0,ancho):
				pixel1 = int(imagen[k][i][j])
				for c in range(1,ventana):		
					pixel2 = int(imagen[k+c][i][j])
					DG[i][j] = DG[i][j] + abs(pixel1-pixel2)
					

	print (DG)
	min = np.amin(DG)
	max = np.amax(DG)
	print("Minimo: ",min," Maximo: ",max)
	DGS = (np.array((DG/((max-min))*255)))
	DGS = DGS.astype(np.uint8)
	print(DGS)

	tiempo_final = time.time()
	print("tardo ", tiempo_final - tiempo_inicial, " segundos")

	return DGS



	