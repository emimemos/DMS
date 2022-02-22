import numpy as np
import time
from funcionesIniciales import float_to_int

def br(imagen, cuadros, alto, ancho):
	tiempo_inicial = time.time()
	media = []
	desviacion = []
	BR = np.zeros((alto,ancho), np.uint8)
	tContraste = np.zeros((alto,ancho), np.uint8)	
	for i in range(0,alto):
		for j in range(0,ancho):	
			linea = imagen[:,i,j]		
			media = linea.mean()		
			desviacion = linea.std(ddof=1)
			suma = linea.sum()
			if media == 0 :
				tContraste[i][j] = 0
				BR[i][j] = 0
			else:
				tContraste[i][j] = int((desviacion / media)*255)
				numero = media / ((1/cuadros)* sum(abs(linea - media)) )
				BR[i][j] = float_to_int(numero)

	tiempo_final = time.time()
	print("tardo ", tiempo_final - tiempo_inicial, " segundos")

	return BR

	