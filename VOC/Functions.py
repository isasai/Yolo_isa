# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 06:58:42 2019

@author: irodr
"""

import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import keras
import tensorflow as tf
from keras import backend as K

#train_dir=("C:/Users/adrin/Desktop/isa/Yolo_isa-master/Yolo_isa-master/fruit-images-for-object-detection/train/") 
#test_dir=("C:/Users/adrin/Desktop/isa/Yolo_isa-master/Yolo_isa-master/fruit-images-for-object-detection/test/")
#
#onlyfiles = [f for f in listdir(train_dir) if isfile(join(train_dir, f))]
#
##Path de prueba para testear mis funciones
#path_xml=train_dir+xml_path+'001627.xml'
#path_img=train_dir+img_path+'001627.img'

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

# =============================================================================
# CLASES
# =============================================================================
class_dict={
"bird":        [1,0,0,0,0,0],
"cat":         [0,1,0,0,0,0],
"cow":         [0,0,1,0,0,0],
"dog":         [0,0,0,1,0,0],
"horse":       [0,0,0,0,1,0],
"sheep":       [0,0,0,0,0,1],
}


# =============================================================================
# FUNCION PARA PARSEAR EL XML Y OBTENER LAS CLASES
# =============================================================================

def parseo_class(path,class_dict):
    classes=[]
    tree = ET.parse(path)
    root = tree.getroot()
    n_obj=(len(root))-6
    if n_obj==1:
        c_type=root[6][0].text
        classes.append(class_dict[c_type])
    elif n_obj>1:
        for i in range(n_obj):
            c_type=root[i+6][0].text
            classes.append(class_dict[c_type])
    return classes

#PARA DATASET DE FRUTAS
#def parseo_class(path):
#    classes=[]
#    tree = ET.parse(path)
#    root = tree.getroot()
#    n_obj=(len(root))-6
#    if n_obj==1:
#        c_type=root[6][0].text
#        if c_type=='apple':
#            c_type=[1,0,0]
#        elif c_type=='banana':
#            c_type=[0,1,0]
#        elif c_type=='orange':
#            c_type=[0,0,1]
#        classes.append(c_type)
#    elif n_obj>1:
#        for i in range(n_obj):
#            c_type=root[i+6][0].text
#            if c_type=='apple':
#                c_type=[1,0,0]
#            elif c_type=='banana':
#                c_type=[0,1,0]
#            elif c_type=='orange':
#                c_type=[0,0,1]
#            classes.append(c_type)
#    return classes



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
        grid.append(grid_fila)
    return grid


# =============================================================================
# FUNCION DE PERDIDA
# =============================================================================

'''
For positional coordinates, we don’t want the network to predict coordinates 
outside of the box so we use the cell system of coordinate. X=0 will place the 
center of the box on the left of the cell. X=1 will place the center of the box 
on the right of the cell. During the training we’ll need to add the offset of 
the cell so that we are in image space. We do this by adding a tensor of offset 
to the network prediction in the loss function.
'''

def loss_yolo(y_pred,y_true):
     
    # pred_boxes = K.Reshape(y_pred[...,3:], (-1,7*7,B,5)) ** QUITAMOS B POR AHORA
    pred_boxes = K.reshape(y_pred[...,3:], (-1,7*7,5))#245
    true_boxes = K.reshape(y_true[...,3:], (-1,7*7,5))#245
    pred_boxes.shape
    true_boxes.shape
    
    # probabilidad de que haya un objeto
    y_pred_conf = pred_boxes[...,4]
    y_true_conf = true_boxes[...,4]
    y_pred_conf.shape
    y_true_conf.shape        
       
    ### xy_loss--------------------------------------
    y_pred_xy   = pred_boxes[...,0:2]  
    y_true_xy   = true_boxes[...,0:2]
    y_pred_xy.shape
    y_true_xy.shape
    
    xy_loss    = 5*(K.sum(K.sum(K.square(y_true_xy - y_pred_xy),axis=-1)*y_true_conf, axis=-1))
    
    ### wh_loss---------------------------------------
    y_pred_wh   = pred_boxes[...,2:4]
    y_true_wh   = true_boxes[...,2:4]
    
    wh_loss    = 5*(K.sum(K.sum(K.square(tf.math.sqrt(y_true_wh) - tf.math.sqrt(y_pred_wh)),axis=-1)*y_true_conf, axis=-1)) 
    
    ### class_loss----------------------------------
    #y_pred_class = y_pred[...,0:3]
    #y_true_class = y_true[...,0:3]
    y_pred_class = K.reshape(y_pred[...,0:3], (-1,7*7,3))
    y_true_class = K.reshape(y_true[...,0:3], (-1,7*7,3))

    
    clss_loss  = K.sum(K.sum(K.square(y_true_class - y_pred_class), axis=-1)*y_true_conf,axis=-1)
    
    ### Conf_loss-------------------------------------- 
    #(***Creo que esto solo tiene sentido cuando tenemos mas de una prediccion por celda (B)!!!!!)

    #Calculo de intersection over union (iou)
        #Coordenadas (xy) superior izquierda e inferior derecha de las cajas predichas y reales
    x1y1_pred=y_pred_xy-(y_pred_wh/2)
    x2y2_pred=y_pred_xy+(y_pred_wh/2)
    x1y1_true=y_true_xy-(y_true_wh/2)
    x2y2_true=y_true_xy+(y_true_wh/2)
        #Coordenadas superior izquierda e inferior derecha del cuadrado de interseccion
    xi1 = K.maximum(x1y1_pred[...,0],x1y1_true[...,0])
    yi1 = K.maximum(x1y1_pred[...,1],x1y1_true[...,1])
    xi2 = K.minimum(x2y2_pred[...,0],x2y2_true[...,0])
    yi2 = K.minimum(x2y2_pred[...,1],x2y2_true[...,1])
        #Calculo de areas
    inter_area = (xi2 - xi1)*(yi2 - yi1)
    true_area = y_true_wh[...,0] * y_true_wh[...,1]
    pred_area = y_pred_wh[...,0] * y_pred_wh[...,1]
    union_area = pred_area + true_area - inter_area
    iou = inter_area / union_area
    
    # -> Calculo del Primer termino de conf_loss (penaliza predicciones incorrectas)
    conf_loss1 = K.sum(K.square(y_true_conf*iou - y_pred_conf)*y_true_conf, axis=-1) 
    
    # -> Calculo del Segundo termino de conf_loss  (penaliza predicciones cuando no hay en realidad objeto)
    '''
        Creamos el tensor y_true_conf_op que es igual que y_true_conf pero intercambiando 
        ceros por unos. Asi tenemos en cuenta las celdas donde no hay objetos y podemos calcular
        la funcion de perdida cuando y_pred_conf != 0  (debe ser cero en las celdas donde no 
        hay objetos)
    '''
    ones_tensor=tf.ones(tf.shape(y_true_conf),dtype='float64')
    y_true_conf_op=ones_tensor-y_true_conf
    conf_loss2 = 0.5*(K.sum(K.square(y_true_conf*iou - y_pred_conf)*y_true_conf_op, axis=-1))
    
        
    ### LOSS FUNCTION
    loss = clss_loss + xy_loss + wh_loss + conf_loss1 + conf_loss2
    
    return loss     
                    
