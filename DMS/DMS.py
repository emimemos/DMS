import tkinter
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from PIL import Image, ImageTk
from diferenciasGeneralizadas import difGen
from fuji import fuji
from descComunes import minimos, maximos, promedio, rango
from br import br
from tContraste import tContraste
import threading
import sompy
from sklearn.cluster import KMeans
from sompy.sompy import SOMFactory
from collections import Counter
from math import sqrt,ceil
import scipy
from icon import Icon
from png import Png
import base64
import os

descriptores =[]
results = [] 
images = []
names = []
index = 0
names.append("calculando")


def ocultarGrid(widget):
	
	''' Oculta el widget de la grilla Tkinter'''
	
	widget._grid_info = widget.grid_info()
	widget.grid_remove()

def mostrarGrid(widget):
	
	''' Muestra el widget en la grilla Tkinter'''
	
	widget.grid(**widget._grid_info)


global descSelected, matrizDescriptores, nombresSelected
def schedule_check(t):
    global raiz
    raiz.after(50, check_if_done, t)
def check_if_done(t):
    if not t.is_alive():
        print("termino")

    else:
        # Si no, volver a chequear en unos momentos.
        schedule_check(t)

def maximosFn():
    global labelA, images,results
    maximosResult = maximos(imagen, alto, ancho)
    results.append(maximosResult)
    image = Image.fromarray(maximosResult)
    image = ImageTk.PhotoImage(image)
    images.append(image)
    names.append("Máximos")
    print('maximos',maximosResult)

def minimosFn():
    global labelA
    minimosResult = minimos(imagen, alto, ancho)
    results.append(minimosResult)
    image = Image.fromarray(minimosResult)
    image = ImageTk.PhotoImage(image)
    images.append(image)
    names.append("Mínimos")
    print('minimos',minimosResult)

def promedioFn():
    global labelA
    promedioResult = promedio(imagen, alto, ancho)
    results.append(promedioResult)
    image = Image.fromarray(promedioResult)
    image = ImageTk.PhotoImage(image)
    images.append(image)
    names.append("Promedio")
    print('promedio',promedioResult)

def rangoFn():
    global labelA
    rangoResult = rango(imagen, alto, ancho)
    results.append(rangoResult)
    image = Image.fromarray(rangoResult)
    image = ImageTk.PhotoImage(image)
    images.append(image)
    names.append("Rango")
    print('rango',rangoResult)

def tContrasteFn():
    global labelA
    tContrasteResult = tContraste(imagen, cuadros, alto, ancho)
    results.append(tContrasteResult)
    image = Image.fromarray(tContrasteResult)
    image = ImageTk.PhotoImage(image)
    images.append(image)
    names.append("TContraste")
    print('tContraste',tContrasteResult)

def brFn():
    global labelA
    brResult = br(imagen, cuadros, alto, ancho)
    results.append(brResult)
    image = Image.fromarray(brResult)
    image = ImageTk.PhotoImage(image)
    images.append(image)
    names.append("BR")
    print('br',brResult)

def fujiFn():
    global labelA
    fujiResult = fuji(imagen, cuadros, alto, ancho)
    results.append(fujiResult)
    image = Image.fromarray(fujiResult)
    image = ImageTk.PhotoImage(image)
    images.append(image)
    names.append("Fuji")
    print('fuji',fujiResult)

def difGenFn():
    global labelA
    difGenResult = difGen(imagen, cuadros, alto, ancho)
    results.append(difGenResult)
    image = Image.fromarray(difGenResult)
    image = ImageTk.PhotoImage(image)
    images.append(image)
    names.append("Diferencia Generalizada")
    print('difGen',difGenResult)

def error():
    print('error')

switch_descriptores = {
    0: maximosFn,
    1: minimosFn,
    2: promedioFn,
    3: rangoFn,
    4: tContrasteFn,
    5: brFn,
    6: fujiFn,
    7: difGenFn,
}

def cargarFrames():
	global imagen, contenedorImagen
	global image, imageCopy
	global img,rep
	global canvas
	global ancho
	global alto
	global cuadros
	rep = 1
	files = filedialog.askopenfilename(multiple=True, ) 
	var = raiz.tk.splitlist(files)
	filePaths = []
	for f in var:
		filePaths.append(f)
	filePaths
	image =[]
	imageCopy = []
	for i in filePaths: 
		image.append(cv.imread(i, cv.IMREAD_GRAYSCALE))
	imageCopy = np.array(image)
	imagen=np.array(image, dtype=np.uint8)
	cuadros = len(imagen)
	alto = len(imagen[0])
	ancho = len(imagen[0][0])
	botonGraficar.config(state = 'normal')
	img = ImageTk.PhotoImage(image=Image.fromarray(image[0]))
	canvas.config(width=ancho, height=alto)
	canvas.itemconfig(contenedorImagen, image=img, anchor=NW)
	frameTrabajo.grid(row = 0, column = 1, sticky = N+W)
	graficar()
	reproducir()
	return alto, ancho, imagen	

def graficar():

	cx = min(int(coordenadax.get()),ancho-1)
	cy =min( int(coordenaday.get()), alto-1)
	coordenadax.delete(0,"end")
	coordenaday.delete(0,"end")
	coordenadax.insert(0,cx)
	coordenaday.insert(0,cy)
	linea = imagen[:,cy,cx]
	x = np.arange(0,len(linea),1)
	fig = Figure(figsize=(5, 4), dpi=100)
	fig.add_subplot(111).plot(x, linea)
	canvasResultadoPIT = FigureCanvasTkAgg(fig, master=frameResultadoPIT)
	canvasResultadoPIT.draw()
	canvasResultadoPIT.get_tk_widget().grid(row = 1 , column= 0)
	etiquetaPIT.grid()
	return

def reproducir():

	''' Reproduce las imágenes cargadas y marca las coordenadas selecionadas '''
	
	global count, rep
	global img
	global canvas
	global contenedorImagen, coordenadax, coordenaday
	global imageCopy
	cx = min(int(coordenadax.get()),ancho-1)
	cy =min( int(coordenaday.get()), alto-1)
	if rep == 1:		
		if count<len(image):	
			for i in range(min(cx+5,ancho-1)):
				imageCopy[count][cy][i] = 255
			for i in range(max(cx-5,ancho-1)):
				imageCopy[count][cy][i] = 255
			for i in range(min(cy+5,alto-1-1)):
				imageCopy[count][i][cx] = 255
			for i in range(max(cy-5,alto-1)):
				imageCopy[count][i][cx] = 255
			img = ImageTk.PhotoImage(image=Image.fromarray(imageCopy[count]))
			count = count + 1
			canvas.itemconfig(contenedorImagen, image=img)#, anchor="center", activeimage=img ,state='normal' )
		else:
			count = 0
		raiz.after(50,reproducir) 

def obtenerDescSelected(descriptores, names, seleccion):
	
	'''
	Obtiene los descriptores seleccionados para hacer el análisis correspondiente
	'''
	
	global descSelected , nombresSelected
	descSelected = []
	nombresSelected = []
	for idx in seleccion:
		descSelected.append(descriptores[idx])
		nombresSelected.append(names[idx])
	return descSelected,nombresSelected

def analizarDescriptores(descriptores):
	'''
	Recoge los descriptores seleccionados para enviarlos a la ventana de analisis
	'''
	global names, box1, matrizDescriptores
	descSelected = []
	if len(box1.curselection())>1:
		descSelected,nombresSelected = obtenerDescSelected(descriptores, names, box1.curselection())
		matrizDescriptores = np.array(descSelected)
		nb.select(1)
		botonSOM["state"] = "normal"
		#nombrearch = filedialog.asksaveasfilename(initialdir = "/",title = "Guardar como", filetypes=[("descriptores","*.npy")])
		#if (nombrearch != ''):
		#    np.analizarDescriptores(nombrearch, descriptores)
		#    mb.showinfo("Información", "El archivo fue guardado correctamente.")
		ocultarGrid(frameOpcionesVisualizacion)
	else:
		messagebox.showinfo(message="Debe seleccionar al menos 2 descriptores", title="¡Atención!")
	return descSelected

def calcular(selected):
        for idx in selected:
            switch_descriptores.get(idx, error)()

def clicked():
	global index, images, names, labelA, panelA, button, box1, framesDescriptores,botonSOM,frameOpciones
	pb1 = ttk.Progressbar(
	frameOpciones,
	orient='horizontal',
	mode='indeterminate',
	length=280
	)
	
	names = []
	selected = box.curselection()  # returns a tuple
	if len(selected)>1:
		
		calcular(selected)
		index = 0 
		panelA = Label(frameImg ,image=images[index])
		panelA.configure(image=images[index])
		panelA.grid(row = 0, column = 0)
		panelA.image = images[index]
		frameButtons = Frame(frameImg)
		frameButtons.grid(row = 1, column = 0)#, columnspan= 2)
		labelA = Label(frameButtons, text="")
		labelA.grid(row = 0, column = 0, columnspan = 2)
		labelA.config(text=names[index])
		buttonIzq = Button(frameButtons, text='<', width=10, command=cambiarIzquierda)
		buttonIzq.grid(row = 1, column = 0)
		buttonDer = Button(frameButtons, text='>', width=10, command=cambiarDerecha)
		buttonDer.grid(row = 1, column = 1)
		box1 = Listbox(frameOpciones, selectmode=MULTIPLE, height=8)
		for val in names:
			box1.insert(END, val)
		box1.grid(row = 2, column = 0)
		descriptores = np.array(results)
		buttonSave = Button(frameOpciones, text='Analizar', width=20, command=lambda:analizarDescriptores(descriptores))
		buttonSave.grid(row = 3, column = 0)

		botonCalcularDescriptores.config(text='Calcular')
		nb.tab(0, text='Seleccionar Descriptores')
		ocultarGrid(botonCalcularDescriptores)
		ocultarGrid(box)
		ocultarGrid(pb1)
	else:
		messagebox.showinfo(message="Debe seleccionar al menos 2 descriptores", title="Atención!")
    #buttonRecognition = Button(frameDescriptores, text='Reconocimiento', width=20, command=lambda:openRecognition(root))#, descriptores))
    #buttonRecognition.grid(row = 2, column = 1)
    #np.analizarDescriptores("prueba.npy", descriptores)

def cambiarDerecha():
    global index
    if(index == len(images)-1):
        index = 0
    else:
        index += 1
    labelA.config(text=names[index])
    panelA.configure(image=images[index])
    panelA.image = images[index]

def cambiarIzquierda():
    global index
    if(index == 0):
        index = len(images)-1
    else:
        index -= 1
    labelA.config(text=names[index])
    panelA.configure(image=images[index])
    panelA.image = images[index]
        
event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y))

clusters = [2,3,4,5,6,7,8,9,10,11,12]

def graficarHitsMap(som):
	'''
	Grafica el Hits Map del SOM
	'''
	fig = plt.figure(figsize =(8, 8))
	fig.suptitle('Hits Map', fontsize=16)
	acontar=som._bmu[0]
	counts = Counter(acontar)
	x=0
	counts = [counts.get(x, 0) for x in range(som.codebook.mapsize[0] * som.codebook.mapsize[1])]
	mp = np.array(counts).reshape(som.codebook.mapsize[0],som.codebook.mapsize[1])
	norm = matplotlib.colors.Normalize(vmin=0,vmax=np.max(mp.flatten()),clip=True)
	ax = plt.gca()
	pl = plt.pcolor(mp[::-1], norm=norm, cmap="jet")
	plt.axis([0, som.codebook.mapsize[1], 0, som.codebook.mapsize[0]])
	ax.set_yticklabels([])
	ax.set_xticklabels([])
	plt.colorbar(pl)
	canvasResultado = FigureCanvasTkAgg(fig, master=frameResultado) 
	canvasResultado.draw()
	canvasResultado.get_tk_widget().grid(row = 0 , column= 0)

def contruirUmatix(som, distance=1):
	'''
	Crea la Umatrix del SOm pasado como parametro
	'''
	UD2 = som.calculate_map_dist()
	Umatrix = np.zeros((som.codebook.nnodes, 1))
	codebook = som.codebook.matrix
	vector = codebook
	for i in range(som.codebook.nnodes):
		codebook_i = vector[i][np.newaxis, :]
		neighborbor_ind = UD2[i][0:] <= distance
		neighborbor_codebooks = vector[neighborbor_ind]
		Umatrix[i] = scipy.spatial.distance_matrix(codebook_i, neighborbor_codebooks).mean()
	return Umatrix.reshape(som.codebook.mapsize)

def graficarUMatrix(som, distance2=1):
	'''
	Grafica la Umatrix del SOM pasado como parametro
	'''
	umat = contruirUmatix(som, distance=distance2)
	fig = Figure(figsize=(8, 8), dpi=100)
	fig.add_subplot(111).imshow(umat, cmap=plt.cm.get_cmap('RdYlBu_r'), alpha=1)
	fig.suptitle('U-Matrix', fontsize=16)
	canvasResultado = FigureCanvasTkAgg(fig, master=frameResultado) 
	canvasResultado.draw()
	canvasResultado.get_tk_widget().grid(row = 0 , column= 0)

def graficarClusters(som):
	'''
	Colorea las celdas del SOM dependiendo del cluster.
	'''
	cantidadClusters = int(spinboxCantClusters.get())
	clusters =som.cluster(cantidadClusters)
	msz = som.codebook.mapsize
	cents = som.bmu_ind_to_xy(np.arange(0, msz[0]*msz[1]))
	cantidadClusters = 8
	fig = plt.figure(figsize =(8, 8))
	ax = plt.subplot(1, 1, 1)
	ax = plt.gca()
	fig.suptitle('Clusters', fontsize=16)
	plt.imshow(np.flip(clusters.reshape(msz[0], msz[1])[::],axis=0),cmap=plt.cm.get_cmap('jet'), alpha=1)
	#plt.imshow(clusters.reshape(msz[0], msz[1]), alpha=0.5)
	for i, txt in enumerate(clusters):
		c = cents[i, 1], cents[-(i + 1), 0]
		ax.annotate(txt, c, va="center", ha="center", size=7)
	canvasResultado = FigureCanvasTkAgg(fig, master=frameResultado) 
	canvasResultado.draw()
	canvasResultado.get_tk_widget().grid(row = 0 , column= 0)

def graficarSalidaCluster(som):
	''' 
	Crea una imagen de salida asiganando el color de cluster al pixel dependiendo
	del cluster asignado a la celda ganadora del pixel en el SOM
	'''
	cantidadClusters = int(spinboxCantClusters.get())
	clusters =som.cluster(cantidadClusters).copy()

	msz = som.codebook.mapsize
	bmus=som._bmu[0].copy()
	clusters =np.flip(clusters)
	print("clusters",clusters)

	bmusIndex = np.array(bmus, np.uint8)
	for i in range (bmusIndex.size):
		bmus[i] = clusters[bmusIndex[i]]#clusterAplastado[bmusIndex[i]]
	fig = plt.figure(figsize=(8, 8), dpi=100)
	fig.suptitle('Salida Generada', fontsize=16)
	print(bmus)
	bmus = bmus.reshape(ancho,alto).transpose()
	bmusInt = np.array(bmus, np.uint8)
	print(bmusInt)
	norm = matplotlib.colors.Normalize(vmin=0,vmax=np.max(bmus.flatten()),clip=True)
	#plt.imshow(bmusInt, cmap=plt.cm.get_cmap('Spectral'), alpha=0.8)
	#plt.pcolormesh(bmusInt,cmap=plt.cm.get_cmap('jet'), alpha=0.8)
	pl = plt.pcolormesh(np.flip(bmusInt[::],axis=0),norm = norm,cmap=plt.cm.get_cmap('jet'), alpha=0.8)
	#plt.colorbar(pl)
	canvasResultado = FigureCanvasTkAgg(fig, master=frameResultado)
	canvasResultado.draw()
	canvasResultado.get_tk_widget().grid(row = 0 , column= 0)
	#plt.close()

def graficarComponentes(som):
	indtoshow, sV, sH = None, None, None
	dim = som._dim
	axis_num = 0
	col_sz = 7
	row_sz = np.ceil(float(dim) / col_sz) 
	msz_row, msz_col = som.codebook.mapsize
	ratio_hitmap = msz_row / float(msz_col)
	ratio_fig = row_sz / float(col_sz)
	indtoshow = np.arange(0, dim).T
	sH, sV = 32, 32*ratio_fig*ratio_hitmap
	codebook = som.codebook.matrix
	subplots = ceil(sqrt(dim))
	names = nombresSelected
	i = 0
	fig = plt.figure(figsize =(8, 8))
	fig.suptitle('Componentes', fontsize=16)
	while axis_num < len(indtoshow):
		axis_num += 1
		ax = plt.subplot(subplots, subplots, axis_num)
		ind = int(indtoshow[axis_num-1])
		min_color_scale = np.mean(codebook[:, ind].flatten()) - 1 * np.std(codebook[:, ind].flatten())
		max_color_scale = np.mean(codebook[:, ind].flatten()) + 1 * np.std(codebook[:, ind].flatten())
		min_color_scale = min_color_scale if min_color_scale >= min(codebook[:, ind].flatten()) else \
		    min(codebook[:, ind].flatten())
		max_color_scale = max_color_scale if max_color_scale <= max(codebook[:, ind].flatten()) else \
		    max(codebook[:, ind].flatten())
		norm = matplotlib.colors.Normalize(vmin=min_color_scale, vmax=max_color_scale, clip=True)
		mp = codebook[:, ind].reshape(som.codebook.mapsize[0],som.codebook.mapsize[1])
		pl = plt.pcolor(mp[::-1], norm=norm)
		plt.axis([0, som.codebook.mapsize[1], 0, som.codebook.mapsize[0]])
		plt.title(names[axis_num - 1])
		ax.set_yticklabels([])
		ax.set_xticklabels([])
		plt.colorbar(pl)
	canvasResultado = FigureCanvasTkAgg(fig, master=frameResultado)  
	canvasResultado.draw()
	canvasResultado.get_tk_widget().grid(row = 0 , column= 0)

def darForma(entrada):
	'''
	formatea el arrglo que contine los descriptores para poder ser analizados por los distintos algoritmos
	'''
	global ancho, alto ,cantDescriptores
	cantPixels = entrada[0].size
	cantDescriptores = len(entrada)
	alto = len(entrada[0])
	ancho = len(entrada[0][0])
	transpuesta = entrada.transpose()
	aplastada = transpuesta.ravel()
	final = aplastada.reshape(cantPixels,cantDescriptores)
	return final, ancho, alto ,cantDescriptores

def hiloSOM():
	'''
	hilo principal de la ejecucion del SOM
	'''
	global som,descSelected,matrizDescriptores,errorTopo
	#entradaSOM = darForma(descSelected[0])
	entradaSOM = darForma(matrizDescriptores)
	cantFilas = int(spinboxFilasSOM.get())
	cantColumnas = int(spinboxColumnasSOM.get())
	mapsize = [cantFilas,cantColumnas]

	som = sompy.SOMFactory.build(entradaSOM[0], mapsize, mask=None, mapshape='planar',
	 lattice='rect', normalization='var', initialization='pca', neighborhood=spinboxVecindario.get(),
	 training='batch', name='sompy')

	som.train(n_job=1, verbose='info', train_rough_len=int(spinboxEpocas.get()), train_finetune_len=1)  # verbose='debug' will print more, and verbose=None wont print anything
	mostrarGrid(frameOpcionesVisualizacion)
	botonHitsMap["state"] = "normal"
	botonUmatrix["state"] = "normal"
	botonClusters["state"] = "normal"
	botonSalida["state"] = "normal"
	botonComponentes["state"] = "normal"

	errorTopo = str(som.calculate_topographic_error())


	return som
def chequear_schedule(t):
    '''
    Programa la ejecución de la función `chequearSiFinalizo()` dentro de 
    x milisegundos.
    '''
    raiz.after(50, chequearSiFinalizo, t)

def chequearSiFinalizo(t):
	
	global som
	# Si el hilo ha finalizado, restaruar el botón y mostrar un mensaje.
	if not t.is_alive():
	    # Restablecer el botón.
		#botonCluster["state"] = "normal"
		#botonSOM["state"] = "normal"
		etiqueteImagenSalida["text"] = "Procesaminto finalizado. Error Topográfico "+errorTopo
		barraProgresoSOM.stop()
		ocultarGrid(barraProgresoSOM)
		mostrarGrid(frameOpcionesVisualizacion)
		graficarUMatrix(som)
	else:
		# Si no, volver a chequear en unos momentos.
		chequear_schedule(t)
def mapaAutoOrganizado():
	mostrarGrid(barraProgresoSOM)
	ocultarGrid(frameOpcionesVisualizacion)
	barraProgresoSOM.start()
	etiqueteImagenSalida["text"] = "Ejecutando Mapa Auto Organizado..."
	# Deshabilitar el botón mientras se ejecuta el cluster.
	#botonCluster ["state"] = "disabled"
	#botonSOM ["state"] = "disabled"
	# Iniciar la descarga en un nuevo hilo.
	t = threading.Thread(target=hiloSOM)
	t.start()
	# Comenzar a chequear periódicamente si el hilo ha finalizado.
	chequear_schedule(t)

global rep 
ancho,alto,rep,count = 512,512,0,0

raiz = Tk()

raiz.title("Proyecto Final DMS")
raiz.resizable(True,True)
with open('imagenGrupo.ico','wb') as tmp:
	tmp.write(base64.b64decode(Icon().img))
raiz.iconbitmap("imagenGrupo.ico")
os.remove('imagenGrupo.ico')
with open('imagenGrupo.png','wb') as tmp:
	tmp.write(base64.b64decode(Png().img))
img = PhotoImage(file = "imagenGrupo.png")
os.remove('imagenGrupo.png')

menubar = Menu(raiz)
menuProyecto = Menu(menubar, tearoff=0)
menuProyecto.add_command(label="Cargar imagenes", command=cargarFrames)
menuProyecto.add_separator()
menuProyecto.add_command(label="Salir", command=raiz.destroy)
menubar.add_cascade(label="Menu principal", menu=menuProyecto)
raiz.config(menu=menubar)

frameInicio = Frame(raiz)
frameTrabajo = Frame(raiz)

frameInicio = Frame(raiz, width = "600", height = "800")
frameInicio.grid_rowconfigure(0, weight = 1)
frameInicio.grid_columnconfigure(0, weight = 1)

frameInicio.grid(row = 0, column = 0, sticky = N )

frameVideo = Frame(frameInicio, width="800", height="800")
frameVideo.grid_rowconfigure(0, weight=1)
frameVideo.grid_columnconfigure(0, weight=1)
frameVideo.config(bg= "#442203")
canvas = Canvas(frameVideo,  bd=0,  width=ancho, height=alto ) 
canvas.grid(row=0, column=0, sticky=N+S+E+W)
contenedorImagen = canvas.create_image(0,0,image=img,anchor=NW)#,anchor="center")
frameVideo.grid(row = 0, column = 0, columnspan = 2)

def printcoords(event):
	global imageCopy,rep
	if rep == 1:
		cx, cy = event2canvas(event, canvas)
		coordenadax.delete(0, END)
		coordenadax.insert(0, event.x-2)
		coordenaday.delete(0, END)
		coordenaday.insert(0, event.y-2)
		imageCopy = np.array(image)
canvas.bind("<ButtonPress-1>",printcoords)

frameinfo = Frame(frameInicio)
frameinfo.grid(row = 1, column = 0, columnspan = 2)
etiquetax = Label(frameinfo,text = "coordenada x")
etiquetax.grid ( row = 0 , column = 0)

etiquetay = Label(frameinfo, text = "coordenada y")
etiquetay.grid ( row = 0 , column = 2)

coordenadax = Entry(frameinfo)
coordenadax.insert(0,"0")
coordenadax.grid(row = 0, column = 1)
#coordenadax.config(state='readonly')
coordenaday = Entry(frameinfo)
coordenaday.insert(0,"0")
coordenaday.grid(row = 0, column = 3)

botonGraficar = Button(frameinfo, text = "graficar pixel", command = graficar, state= "disabled" )
botonGraficar.grid(row = 0 , column= 4, sticky=E+W)

frameResultadoPIT = Frame(frameInicio,height=420,width=512)
frameResultadoPIT.grid(row = 2, column = 0, columnspan = 2)
etiquetaPIT = Label(frameResultadoPIT, text = "intensidad pixel / frame")
etiquetaPIT.grid(row = 2, column = 0, sticky = E + W)
etiquetaPIT.grid_remove()

frameTrabajo = Frame(raiz, width = "600", height = "900")
frameTrabajo.grid_rowconfigure(0, weight = 1)
frameTrabajo.grid_columnconfigure(0, weight = 1)
frameTrabajo.config(bg = "#442243")
#frameTrabajo.grid(row = 0, column = 1, sticky = N+W)

nb = ttk.Notebook(frameTrabajo)
#nb.pack(fill='both',expand='yes')
nb.grid(row = 0, column = 0, sticky =N+E+S+W) 
frameDescriptores = Frame(nb, width = "600", height = "800")
frameDescriptores.grid_rowconfigure(0, weight = 1)
frameDescriptores.grid_columnconfigure(0, weight = 1)

frameDescriptores.grid(row = 0, column = 0, sticky = N )

frameOpciones = Frame(frameDescriptores)
frameOpciones.grid(row = 0, column = 0, sticky = N+W+S+E)
box = Listbox(frameOpciones, selectmode=MULTIPLE, height=8)
values = ['Máximos', 'Mínimos', 'Promedio', 'Rango', 'TContraste', 'BR', 'Fuji', 'Diferencia Generalizada']
for val in values:
    box.insert(END, val)
box.grid(row = 0 , column = 0)
botonCalcularDescriptores = Button(frameOpciones, text='Calcular', width=20, command=clicked)
botonCalcularDescriptores.grid(row = 1, column = 0)

frameImg = Frame(frameDescriptores)
frameImg.grid(row = 0, column = 1, sticky=N+W+E+S)

frameAnalisis = Frame(nb, width = "600", height = "800")
frameAnalisis.grid_rowconfigure(0, weight = 1)
frameAnalisis.grid_columnconfigure(0, weight = 1)

frameAnalisis.grid(row = 0, column = 0, sticky = N )

frameBotones = Frame(frameAnalisis)
frameBotones.grid(row = 0, column = 0, sticky= N+W+E , padx =5, pady =5)

frameOpcionesSOM = Frame(frameBotones)
frameOpcionesSOM.grid(row = 0, column = 0, sticky= N+S+W )

#labelSOM = Label(frameOpcionesSOM,
#	text ="Elija dimensiones som")
#labelSOM.grid(row = 0, column = 0, columnspan =2, sticky= N+S+W)

etiquetaFilas = Label(frameOpcionesSOM,
	text = "Filas SOM:"
	)
etiquetaFilas.grid(row = 0, column = 0,  sticky=W)
spinboxFilasSOM = ttk.Spinbox(frameOpcionesSOM, from_ = 2, to = 60,  state='readonly', wrap = True, width=3)
spinboxFilasSOM.set(12)
spinboxFilasSOM.grid(row = 0 , column = 1)

etiquetaColumnas = Label(frameOpcionesSOM,
	text = "Columnas SOM:",
	)
etiquetaColumnas.grid(row = 1, column = 0,  sticky=W)
spinboxColumnasSOM = ttk.Spinbox(frameOpcionesSOM, from_ = 2, to = 60,  state='readonly', wrap = True, width=3)
spinboxColumnasSOM.set(12)
spinboxColumnasSOM.grid(row = 1 , column = 1)

botonSOM = Button(frameOpcionesSOM, 
	text="Calcular SOM :" ,
	state= "disabled",
	command=mapaAutoOrganizado
	)
botonSOM.grid(row=2, column = 0, columnspan =4, sticky=N+S+E+W)

etiquetaVecindario = Label(frameOpcionesSOM,
	text = "Func. vecindario:"
	)
etiquetaVecindario.grid(row =0, column=2,  sticky=W)
vecindario=("gaussian", "bubble")
spinboxVecindario=ttk.Spinbox(frameOpcionesSOM, values=vecindario,  state='readonly', wrap = True, width=8)        
spinboxVecindario.set(vecindario[0]) 
spinboxVecindario.grid(row = 0 , column = 3, sticky=E)

etiquetaEpocas = Label(frameOpcionesSOM,
	text = "Epocas Entrenamiento:"
	)
etiquetaEpocas.grid(row =1, column=2, sticky=W)
spinboxEpocas=ttk.Spinbox(frameOpcionesSOM, from_ = 2, to = 20,  state='readonly', wrap = True, width=3)  
spinboxEpocas.set(10)       
spinboxEpocas.grid(row = 1 , column = 3, sticky=E)

frameOpcionesVisualizacion = Frame(frameBotones)
frameOpcionesVisualizacion.grid(row = 0, column = 1,rowspan = 3,sticky= N+S+W+E, padx =10, )

botonHitsMap = Button(frameOpcionesVisualizacion,
	text = "Hits Map",
	state= "disabled",
	height = 4,
	command = lambda:graficarHitsMap(som)
	)
botonHitsMap.grid(row = 0 , column= 0, rowspan = 2, columnspan = 1, sticky=E+W)

botonUmatrix = Button(frameOpcionesVisualizacion, 
	text = "Umatrix",
	state= "disabled", 
	height = 4,
	command = lambda:graficarUMatrix(som)
	)
botonUmatrix.grid(row = 0 , column= 1, rowspan = 2,columnspan = 1, sticky=E+W)

botonClusters = Button(frameOpcionesVisualizacion, 
	text = "Clusters",
	state= "disabled", 
	command = lambda:graficarClusters(som)
	)
botonClusters.grid(row = 0 , column= 2, columnspan = 1, sticky=E+W+S+N)

botonSalida = Button(frameOpcionesVisualizacion, 
	text = "SalidaCluster",
	state= "disabled", 
	height = 4,
	command = lambda:graficarSalidaCluster(som)
	)
botonSalida.grid(row = 0 , column= 4,rowspan = 2, columnspan = 1, sticky=E+W)

botonComponentes = Button(frameOpcionesVisualizacion, 
	text = "Componentes",
	state= "disabled", 
	height = 4,
	command = lambda:graficarComponentes(som)
	)
botonComponentes.grid(row = 0 , column= 5,rowspan = 2, columnspan = 1, sticky=E+W)

spinboxCantClusters = ttk.Spinbox(frameOpcionesVisualizacion, from_ = 2, to = 20,  state='readonly', wrap = True, width=4)
spinboxCantClusters.grid(row=1, column = 2)
spinboxCantClusters.set(6) # default value

frameResultado = Frame(frameAnalisis)
frameResultado.grid(row = 1, column = 0, columnspan = 2)
etiqueteImagenSalida = Label(frameResultado)
etiqueteImagenSalida.grid(row=1, column = 0, columnspan = 5, sticky=E+W)
barraProgresoSOM = ttk.Progressbar(
    frameResultado,
    orient='horizontal',
    mode='indeterminate',
    length=500
)
# place the progressbar
barraProgresoSOM.grid(column=0, row=2, columnspan=2, padx=10, pady=20)
ocultarGrid(barraProgresoSOM)

nb.add(frameDescriptores,text='Cálculo Descriptores')
nb.add(frameAnalisis,text='Generar Salida')#, state="disabled")

raiz.after(50,reproducir) 

raiz.mainloop()