Instalación de Python:
-instructivo de instalacion de python 'https://datascience.com.co/how-to-install-python-2-7-and-3-6-in-windows-10-add-python-path-281e7eae62a'
-instalar python 2.7 desde 'https://www.python.org/downloads/release/python-2717/'
-configurar el PATH desde windows
-instalar python 3 desde 'https://www.python.org/downloads/'
-configurar el PATH desde windows


Dependencias:
-instalar opencv desde consola con el comando 'pip install opencv-python'
-instalar matplotlib desde consola con el comando 'pip install matplotlib'
-instalar sompy desde consola con el comando 'pip3 install git+https://github.com/compmonks/SOMPY.git'
-instalar ipbd desde consola con el comando 'pip install ipdb'
-instalar sklearn desde consola con el comando 'pip install scikit-learn'
-instalar pyinstaller desde consola con el comando 'pip install pyinstaller'


Agregar descriptores:
-crear el archivo .py con la funcion del nuevo descriptor. (tomar de ejemplo: 'descComunes.py')
-en el archivo DMS.py agregar el import del nuevo descriptor. Ejemplo:
	from nuevoDescriptor import nuevoDescriptor 
-en el archivo DMS.py (linea 143 aprox) agregar una funcion que utilice el nuevo descriptor. Ejemplo:
	def nuevoDescriptorFn():
    		global labelA
    		nuevoDescriptorResult = nuevoDescriptor(imagen, cuadros, alto, ancho)
    		results.append(nuevoDescriptorResult)
    		image = Image.fromarray(nuevoDescriptorResult)
    		image = ImageTk.PhotoImage(image)
    		images.append(image)
    		names.append("Nuevo Descriptor")
    		print('Nuevo Descriptor',nuevoDescriptorResult)
-en el archivo DMS.py (linea 147, luego de las definicion de funciones) 
agregar en la variable 'switch_descriptores' la nueva funcion recientemente creada al final del arreglo. Ejemplo:
	switch_descriptores = {
    		0: maximosFn,
    		1: minimosFn,
    		2: promedioFn,
    		3: rangoFn,
    		4: tContrasteFn,
    		5: brFn,
    		6: fujiFn,
    		7: difGenFn,
		8: nuevoDescriptorFn,
	}


Generar .exe:
-desde consola ejecutar el siguiente comando:

pyinstaller --noconfirm --onefile --windowed --icon "C:/Users/emi_m/Desktop/DMS/imagenGrupo.ico" --add-data "C:/Users/emi_m/Desktop/DMS/br.py;." --add-data "C:/Users/emi_m/Desktop/DMS/descComunes.py;." --add-data "C:/Users/emi_m/Desktop/DMS/diferenciasGeneralizadas.py;." --add-data "C:/Users/emi_m/Desktop/DMS/fuji.py;." --add-data "C:/Users/emi_m/Desktop/DMS/funcionesIniciales.py;." --add-data "C:/Users/emi_m/Desktop/DMS/tContraste.py;." --hidden-import "sklearn.utils._typedefs" --hidden-import "sklearn.neighbors._partition_nodes" --add-data "C:/Users/emi_m/Desktop/DMS/icon.py;." --add-data "C:/Users/emi_m/Desktop/DMS/png.py;."  "C:/Users/emi_m/Desktop/DMS/DMS.py"

(en el caso de haber agregado algun nuevo descriptor, en la zona de los --add-data agregar el archivo nuevo.)

########     ##     ##     ######
##     ##    ###   ###    ##    ##
##     ##    #### ####    ##
##     ##    ## ### ##     ######
##     ##    ##     ##          ##
##     ##    ##     ##    ##    ##
########     ##     ##     ######
