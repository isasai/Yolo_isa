# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 06:58:42 2019

@author: irodr
"""

# =============================================================================
# LABELS - XML FORMAT
# =============================================================================

#Archivo en xml:
#Coordenadas de bboxes en pixeles correspondientes a las esquians de las cajas

'''
<annotation>
	<folder>test</folder>
	<filename>apple_77.jpg</filename>
	<path>...\images\test\apple_77.jpg</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>300</width>
		<height>229</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>apple</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>71</xmin>
			<ymin>60</ymin>
			<xmax>175</xmax>
			<ymax>164</ymax>
		</bndbox>
	</object>
    <object>
    ...
    </object>
    <object>
    ...
    </object>
    ...
</annotation>
'''

#%%
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

train_dir=('C:/Users/irodr/Documents/NEOLAND/Proyecto/YOLO/fruit-images-for-object-detection/train/') 
test_dir=('C:/Users/irodr/Documents/NEOLAND/Proyecto/YOLO/fruit-images-for-object-detection/test')

onlyfiles = [f for f in listdir(train_dir) if isfile(join(train_dir, f))]

#Path de prueba para testear mis funciones
path=train_dir+'apple_3.xml'

# =============================================================================
# FUNCION PARA PARSEAR EL XML Y OBTENER LAS CLASES
# =============================================================================

def parseo_class(path):
    classes=[]
    tree = ET.parse(path)
    root = tree.getroot()
    n_obj=(len(root))-6
    if n_obj==1:
        c_type=root[6][0].text
        if c_type=='apple':
            c_type=[1,0,0]
        elif c_type=='banana':
            c_type=[0,1,0]
        elif c_type=='orange':
            c_type=[0,0,1]
        classes.append(c_type)
    elif n_obj>1:
        for i in range(n_obj):
            c_type=root[i+6][0].text
            if c_type=='apple':
                c_type=[1,0,0]
            elif c_type=='banana':
                c_type=[0,1,0]
            elif c_type=='orange':
                c_type=[0,0,1]
            classes.append(c_type)
    return classes

classes=parseo_class(path)


# =============================================================================
# FUNCIONES PARA PARSEAR EL XML Y TRANSFORMAR COORDENADAS DE LAS CAJAS A 
#(1)COORDENADAS CONSIDERANDO EL TAMANO DE LA IMG INTRODUCIDA EN LA CNN
#(2)COORD A ESCALA 0-1
# =============================================================================

#Funcion para parsear un archivo xml y devolver coordenadas de las cajas
def parseo_coord(path):
    boxes=[]
    tree = ET.parse(path)
    root = tree.getroot()
    size_w=float(root[4][0].text)
    size_h=float(root[4][1].text)
    size=[size_w,size_h]
    n_obj=(len(root))-6
    if n_obj==1:
        xmin=float(root[6][4][0].text)
        xmax=float(root[6][4][2].text)
        ymin=float(root[6][4][1].text)
        ymax=float(root[6][4][3].text)
        box=[xmin,xmax,ymin,ymax]
        boxes.append(box)
    elif n_obj>1:
        for i in range(n_obj):
            xmin=float(root[i+6][4][0].text)
            xmax=float(root[i+6][4][2].text)
            ymin=float(root[i+6][4][1].text)
            ymax=float(root[i+6][4][3].text)
            box=[xmin,xmax,ymin,ymax]
            boxes.append(box)
    return boxes,size

boxes,size=parseo_coord(path)

#Funcion para reescalar coord de la imagen a las coord de la imagen de la CNN 
def coord_CNN(boxes,size,sizeCNN):
    #size*X=sizeCNN -> X=sizeCNN/size
    boxes_CNN=[]
    X=sizeCNN/size[0]
    Y=sizeCNN/size[1]
    for box in boxes:
        xminCNN=box[0]*X
        xmaxCNN=box[1]*X
        yminCNN=box[2]*Y
        ymaxCNN=box[3]*Y
        box_CNN=[xminCNN,xmaxCNN,yminCNN,ymaxCNN]
        boxes_CNN.append(box_CNN)
    return boxes_CNN,X,Y

boxes_CNN,X,Y=coord_CNN(boxes,size,448)

#Funcion para transformar a coordenadas de las cajas a escala 0-1
def convert(sizeCNN, boxes_CNN):
    boxes_yolo=[]
    for box in boxes_CNN:
        dw = 1./sizeCNN
        dh = 1./sizeCNN
        x = (box[0] + box[1])/2.0
        y = (box[2] + box[3])/2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = (x*dw)
        w = (w*dw)
        y = (y*dh)
        h = (h*dh)
        box_yolo=[x,y,w,h]
        boxes_yolo.append(box_yolo)
    return boxes_yolo

boxes_yolo=convert(448,boxes_CNN)

# =============================================================================
# FUNCION PARA CREAR LAS COORDENADAS DE LAS CELDAS DE LA GRID
# =============================================================================
'''
grid es una lista de listas que consiste en ncell*ncell celdas, estando
cada celda definida por una lista con 4 coordenadas [a=xmin,b=xmax,c=ymin,d=ymax]
'''   
def grid(ncell):
    grid=[]
    for i in range(ncell):
        grid_fila=[]
        for j in range(ncell):
            grid_fila.append([j*1/ncell,j*1/ncell+1/ncell,i*1/ncell,i*1/ncell+1/ncell])
        grid.append([grid_fila])
    return grid

grid=grid(7)
len(grid)


#Bucle para crear y_true para una sola imagen------------------------------------------------------
# y_true2=[]
# conteo_cellOK=0
# classes=parseo_class(train_dir+'apple_3.xml')
# pr_obj=[1]
# boxes,size=parseo_coord(train_dir+'apple_3.xml')
# boxes_CNN,X,Y=coord_CNN(boxes,size,448)
# boxes_yolo=convert(448,boxes_CNN)
# print(boxes_yolo)
# for i in range(len(grid)):
#     for j in range(len(grid)):
#         a=grid[i][j][0]
#         b=grid[i][j][1]
#         c=grid[i][j][2]
#         d=grid[i][j][3]
#         insert=[0,0,0,0,0,0,0,0]
#         count=0
#         for box in boxes_yolo:
#             if a<box[0]<b and c<box[1]<d:
#                 insert=classes[count]+box+pr_obj
#                 count+=1
#                 conteo_cellOK+=1
#                 print(conteo_cellOK)
#                 break
#             else:                
#                 continue
#         y_true2.append(insert)
# y_true2


     
                    
                
                