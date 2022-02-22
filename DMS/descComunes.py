import numpy as np
import time

def calcMinimos(imagen, alto, ancho):
	minimos = np.zeros((alto,ancho), np.uint8)
	np.amin(imagen, axis = 0, out=minimos)
	return minimos

def minimos(imagen, alto, ancho):
	tiempo_inicial = time.time()
	minimos = calcMinimos(imagen, alto, ancho)
	tiempo_final = time.time()
	print("tardo ", tiempo_final - tiempo_inicial, " segundos")
	return minimos

def calcMaximos(imagen, alto, ancho):
	maximos = np.zeros((alto,ancho), np.uint8)
	np.amax(imagen, axis = 0, out=maximos)
	return maximos

def maximos(imagen, alto, ancho):
	tiempo_inicial = time.time()
	maximos = calcMaximos(imagen, alto, ancho)
	tiempo_final = time.time()
	print("tardo ", tiempo_final - tiempo_inicial, " segundos")
	return maximos

def calcPromedio(imagen, alto, ancho):
	promedio = np.zeros((alto,ancho), np.uint32)
	np.mean(imagen, axis = 0, out=promedio)
	promedio = promedio.astype(np.uint8)
	return promedio

def promedio(imagen, alto, ancho):
	tiempo_inicial = time.time()
	promedio = calcPromedio(imagen, alto, ancho)
	tiempo_final = time.time()
	print("tardo ", tiempo_final - tiempo_inicial, " segundos")
	return promedio

def calcRango(imagen, alto, ancho):
	rango = np.zeros((alto,ancho), np.uint8)
	maxi = calcMaximos(imagen, alto, ancho)
	mini = calcMinimos(imagen, alto, ancho)
	rango = maxi - mini
	return rango

def rango(imagen, alto, ancho):
	tiempo_inicial = time.time()
	rango = calcRango(imagen, alto, ancho)
	tiempo_final = time.time()
	print("tardo ", tiempo_final - tiempo_inicial, " segundos")
	return rango


