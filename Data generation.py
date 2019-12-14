# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 12:53:52 2019

@author: irodr
"""
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import xml.etree.ElementTree as ET
import keras
#Para generar los datos necesitamos ejecutar las funciones del archivo Functions.py


#En las carpetas de train y test estan juntas las imagenes con los xml con las etiquetas
train_dir=("C:/Users/adrin/Desktop/isa/Yolo_isa-master/Yolo_isa-master/fruit-images-for-object-detection/train/") 
test_dir=("C:/Users/adrin/Desktop/isa/Yolo_isa-master/Yolo_isa-master/fruit-images-for-object-detection/test/")

## SI TENGO TIEMPO, ES MUCHO MEJOR PONER TODAS MIS IMGENES Y CARPETAS EN UN MISMO DIRECTORIO, CONVERTIR TODO A ARRAY Y LUEGO HACER EL SPLIT DEL TRAIN, VAL Y TEST

# =============================================================================
#  TRAIN - X DATA
# =============================================================================

onlyfiles_train = [f for f in listdir(train_dir) if isfile(join(train_dir, f))]
num_imagenes_train=len(onlyfiles_train)/2

train_img=[]
for file in onlyfiles_train:
    if file[-1]=='g': # =='g' Para que me selecciones solo los archivos jpg
        img=keras.preprocessing.image.load_img(
        path=train_dir + file,
        grayscale=False,
        color_mode='rgb',
        target_size=(248,248,3), #448
        interpolation='nearest')
        img_arr=keras.preprocessing.image.img_to_array(
        img,
        data_format=None,
        dtype='float64')
        img_arr=img_arr/255
        train_img.append(img_arr)

fruit=train_img[0]
import matplotlib.pyplot as plt
plt.imshow(fruit, cmap=plt.cm.binary)
plt.show()

x_data_train=np.asarray(train_img)
x_data_train.shape


#X_TRAIN / X_VAL

x_val=x_data_train[201:]
x_val.shape
x_train=x_data_train[:201]
x_train.shape

# =============================================================================
# TRAIN - Y DATA
# =============================================================================

##Grid (funcion definida en el archivo Functions.py)
#def grid(ncell):
#    grid=[]
#    for i in range(ncell):
#        grid_fila=[]
#        for j in range(ncell):
#            grid_fila.append([j*1/ncell,j*1/ncell+1/ncell,i*1/ncell,i*1/ncell+1/ncell])
#        grid.append(grid_fila)
#    return grid



#Bucle para crear y_true con los datos de todas las imagenes
'''
El bucle inserta un vector [0,0,0,0,0,0,0,0] en y_true por cada celda
de la grid siempre que esa celda no contenga un objeto. Si contiene un objeto
(linea a<box[0]<b and c<box[1]<d  -> el centro de la caja que contine al
objeto esta entre las coordenadas de la celda) entonces introducimos
el vector de clases y coordenadas del objeto [c1,c2,c3,x,y,w,h,p])
'''

grid=grid(7)
y_data=[]        
for file in onlyfiles_train:
    if file[-1]=='l': # file =='l' Para que me selecciones solo los archivos xml
        classes=parseo_class(train_dir+file) #Funcion en archivo Functions.py
        pr_obj=[1]
        boxes,size=parseo_coord(train_dir+file) #Funcion en archivo Functions.py
        boxes_CNN,X,Y=coord_CNN(boxes,size,448) #Funcion en archivo Functions.py
        boxes_yolo=convert(448,boxes_CNN) #Funcion en archivo Functions.py
        for i in range(len(grid)):
            for j in range(len(grid)):
                #coordenadas de las celdas:
                xmin=grid[i][j][0]
                xmax=grid[i][j][1]
                ymin=grid[i][j][2]
                ymax=grid[i][j][3]
                insert=[0,0,0,0,0,0,0,0]
                count=0
                for box in boxes_yolo:
                    if xmin<box[0]<xmax and ymin<box[1]<ymax:
                        insert=classes[count]+box+pr_obj
                        count+=1
                        break
                    else:                
                        continue
                y_data.append(insert)

y_data=np.asarray(y_data)
y_data=np.reshape(y_data,(-1,(7*7),8))
y_data.shape


# Y_TRUE / Y_VAL

y_val=y_data[201:]
y_val.shape
y_true=y_data[:201]
y_true.shape


# =============================================================================
# TEST - X DATA
# =============================================================================

onlyfiles_test = [f for f in listdir(test_dir) if isfile(join(test_dir, f))]
num_imagenes=len(onlyfiles_test)/2

test_img=[]
for file in onlyfiles_test:
    if file[-1]=='g': # =='g' Para que me selecciones solo los archivos jpg
        img=keras.preprocessing.image.load_img(
        path=test_dir + file,
        grayscale=False,
        color_mode='rgb',
        target_size=(248,248,3), #448
        interpolation='nearest')
        img_arr=keras.preprocessing.image.img_to_array(
        img,
        data_format=None,
        dtype='float64')
        img_arr=img_arr/255
        test_img.append(img_arr)

fruit=test_img[0]
import matplotlib.pyplot as plt
plt.imshow(fruit, cmap=plt.cm.binary)
plt.show()

x_test=np.asarray(test_img)
x_test.shape


# =============================================================================
# TEST - Y DATA
# =============================================================================        

#Buscamos imagenes con size 0 (no valen, hay que eliminarlas)
for file in onlyfiles_test:  
    if file[-1]=='l':
        tree = ET.parse(test_dir+file)
        root = tree.getroot()
        size_w=float(root[4][0].text)
        size_h=float(root[4][1].text)
        if size_w == 0 or size_h == 0:
            print(file)          
        
###       
y_test=[]        
for file in onlyfiles_test:
    if file[-1]=='l': # file =='l' Para que me selecciones solo los archivos xml
        classes=parseo_class(test_dir+file) #Funcion en archivo Functions.py
        pr_obj=[1]
        boxes,size=parseo_coord(test_dir+file) #Funcion en archivo Functions.py
        boxes_CNN,X,Y=coord_CNN(boxes,size,448) #Funcion en archivo Functions.py
        boxes_yolo=convert(448,boxes_CNN) #Funcion en archivo Functions.py
        for i in range(len(grid)):
            for j in range(len(grid)):
                #coordenadas de las celdas:
                xmin=grid[i][j][0]
                xmax=grid[i][j][1]
                ymin=grid[i][j][2]
                ymax=grid[i][j][3]
                insert=[0,0,0,0,0,0,0,0]
                count=0
                for box in boxes_yolo:
                    if xmin<box[0]<xmax and ymin<box[1]<ymax:
                        insert=classes[count]+box+pr_obj
                        count+=1
                        break
                    else:                
                        continue
                y_test.append(insert)

y_test=np.asarray(y_test)
y_test=np.reshape(y_test,(-1,(7*7),8))
y_test.shape
