# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:56:38 2019

@author: irodr
"""

# =============================================================================
# YOLO V1 ALGORITHM WITH KERAS - FROM SCRATCH
# =============================================================================

#%%
# =============================================================================
# PACKAGES
# =============================================================================

import os
from os import listdir
from os.path import isfile, join
from keras import models
from keras.models import load_model, Model
from keras import layers
from keras import regularizers
from keras import losses
from keras import metrics
from keras import optimizers
import numpy as np
import keras
import tensorflow as tf
from keras import backend as K #para hacer operaciones entre vectores
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, Reshape, MaxPooling2D


#%%
# =============================================================================
# DATA
# =============================================================================

'''
En las carpetas de train y test estan juntas las imagenes con los xml con las etiquetas
'''

train_dir=('C:/Users/irodr/Documents/NEOLAND/Proyecto/YOLO/YOLO_ISA/fruit-images-for-object-detection/train/') 
test_dir=('C:/Users/irodr/Documents/NEOLAND/Proyecto/YOLO/YOLO_ISA/fruit-images-for-object-detection/test')

onlyfiles = [f for f in listdir(train_dir) if isfile(join(train_dir, f))]
num_imagenes=len(onlyfiles)/2

# =============================================================================
# x_train
# =============================================================================

train_img=[]
for file in onlyfiles:
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

x_train=np.asarray(train_img)

x_train.dtype
type(x_train)
x_train.shape



#%%
# =============================================================================
# y_train(==y_true)
# =============================================================================

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

#Grid (funcion definida en el archivo Functions.py)
def grid(ncell):
    grid=[]
    for i in range(ncell):
        grid_fila=[]
        for j in range(ncell):
            grid_fila.append([j*1/ncell,j*1/ncell+1/ncell,i*1/ncell,i*1/ncell+1/ncell])
        grid.append(grid_fila)
    return grid
grid=grid(7)

#Bucle para crear y_true con los datos de todas las imagenes
'''
El bucle inserta un vector [0,0,0,0,0,0,0,0] en y_true por cada celda
de la grid siempre que esa celda no contenga un objeto. Si contiene un objeto
(linea a<box[0]<b and c<box[1]<d  -> el centro de la caja que contine al
objeto esta entre las coordenadas de la celda) entonces introducimos
el vector de clases y coordenadas del objeto [c1,c2,c3,x,y,w,h,p])
'''
y_true=[]        
for file in onlyfiles:
    if file[-1]=='l': # file =='l' Para que me selecciones solo los archivos xml
        classes=parseo_class(train_dir+file) #Funcion en archivo Functions.py
        pr_obj=[1]
        boxes,size=parseo_coord(train_dir+file) #Funcion en archivo Functions.py
        boxes_CNN,X,Y=coord_CNN(boxes,size,448) #Funcion en archivo Functions.py
        boxes_yolo=convert(448,boxes_CNN) #Funcion en archivo Functions.py
        for i in range(len(grid)):
            for j in range(len(grid)):
                #coordenadas de las celdas:
                a=grid[i][j][0]
                b=grid[i][j][1]
                c=grid[i][j][2]
                d=grid[i][j][3]
                insert=[0,0,0,0,0,0,0,0]
                count=0
                for box in boxes_yolo:
                    if a<box[0]<b and c<box[1]<d:
                        insert=classes[count]+box+pr_obj
                        count+=1
                        break
                    else:                
                        continue
                y_true.append(insert)

len(y_true)
y_true=np.asarray(y_true)
y_true=np.reshape(y_true,(207,(7*7),8))
          
#%%
# =============================================================================
# MODEL
# =============================================================================

#The YOLO v1 is consist of 24 convolution layers and 3 full connected layers. 
#Each convolution layer consists of convolution, leaky relu and max pooling operations. 
#The first 24 convolution layers can be understood as the feature extractor, 
#whereas the last three full connected layers can be understood as the 
#"regression head" that predicts the bounding boxes.
    #1-First 20 convolutional layers followed by an average pooling layer and a fully connected layer is pre-trained on the ImageNet 1000-class classification dataset
    #2-The pretraining for classification is performed on dataset with resolution 224 x 224
    #3-The layers comprise of 1x1 reduction layers and 3x3 convolutional layers
    #4-Last 4 convolutional layers followed by 2 fully connected layers are added to train the network for object detection
    #5-Object detection requires more granular detail hence the resolution of the dataset is bumped to 448 x 448
    #6-The final layer predicts the class probabilities and bounding boxes.

#NET STRUCTURE - YOLO V1
i=Input(shape=(448,448,3))
x=Conv2D(64, (7,7))(i) #Conv1  ###1 -> 224
x=keras.layers.LeakyReLU(alpha=0.3)(x)
x=MaxPooling2D((2, 2))(x) ###2 ->112
x=Conv2D(192, (3, 3))(x) #Conv2
x = keras.layers.LeakyReLU(alpha=0.3)(x)
x=MaxPooling2D((2, 2))(x) ###3 -> 56
x=Conv2D(128, (1,1))(x) #Conv3
x=Conv2D(256, (3, 3))(x) #Conv4
x=Conv2D(256, (1,1))(x) #Conv5
x=Conv2D(512, (1,1))(x) #Conv6
x = keras.layers.LeakyReLU(alpha=0.3)(x)
x=MaxPooling2D((2, 2))(x) ###4 -> 28
x=Conv2D(256, (1,1))(x) #Conv7
x=Conv2D(512, (3, 3))(x) ##Conv8
x=Conv2D(256, (1,1))(x) #Conv9
x=Conv2D(512, (3, 3))(x) #Conv10
x=Conv2D(256, (1,1))(x) #Conv11
x=Conv2D(512, (3, 3))(x) #Conv12
x=Conv2D(256, (1,1))(x) #Conv13
x=Conv2D(512, (3, 3))(x) #Conv14
x=Conv2D(512, (1,1))(x) #Conv15
x=Conv2D(1024, (3,3))(x) #Conv16
x=keras.layers.LeakyReLU(alpha=0.3)(x)
x=MaxPooling2D((2, 2))(x) ###5 -> 14
x=Conv2D(512, (1,1))(x) #Conv17
x=Conv2D(1024, (3, 3))(x) #Conv18
x=Conv2D(512, (1,1))(x) #Conv19
x=Conv2D(1024, (3, 3))(x) #Conv20
x=Conv2D(1024, (3, 3))(x) #Conv21
x=Conv2D(1024, (3, 3))(x) #Conv22 ###6 -> 7
x=Conv2D(1024, (3, 3))(x) #Conv23
x=Conv2D(1024, (3, 3))(x) #Conv24
x=keras.layers.LeakyReLU(alpha=0.3)(x)
x=Flatten()(x)
x=Dense(256,activation='sigmoid')(x)
x=Dropout(0.5)(x)
x=Dense(7*7*(1*5+3),activation='sigmoid')(x)
x=Reshape((7*7,(1*5+3)))(x) 
##Me falta poner data augmentation!!!

#grid=numero de celdas en las que queremos evaluar si hay objetos
#B=1: numero de predicciones. Voy a empezar probando con 1
#C=3: numero de clases, en el caso de mi dataset son 3

'''
The trick is to have the network to predict coordinates that are limited in 
their range to a single cell. To limit the values that the network can 
output we use sigmoid on it’s output. Sigmoid is a common activation function 
in deep learning but here we use it for it’s property of only taking values 
between 0 and 1.
Since we only predict value between 0 and 1, we will consider width and height
to be in the system of coordinate of the image. A box of image width will have 
a width of 1.
'''
#%%
#SIMPPLIFIED NET
tf.keras.backend.set_floatx('float64')

i=Input(shape=(248,248,3))
x=Conv2D(64, (7,7),padding='same')(i) #Conv1  
x=keras.layers.LeakyReLU(alpha=0.3)(x)
x=MaxPooling2D((2, 2))(x) ###1 ->124
x=Conv2D(128, (1,1),padding='same')(x) #Conv3
x=Conv2D(192, (3, 3),padding='same')(x) #Conv2
x = keras.layers.LeakyReLU(alpha=0.3)(x)
x=MaxPooling2D((2, 2))(x) ###2 -> 62
x=Conv2D(128, (1,1),padding='same')(x) #Conv3
x=Conv2D(256, (3, 3),padding='same')(x) #Conv4
x=Conv2D(256, (1,1),padding='same')(x) #Conv5
x=Conv2D(512, (1,1),padding='same')(x) #Conv6
x = keras.layers.LeakyReLU(alpha=0.3)(x)
x=MaxPooling2D((2, 2),padding='same')(x) ###3 -> 31
x=Conv2D(256, (1,1),padding='same')(x) #Conv7
x=Conv2D(512, (3, 3),padding='same')(x) ##Conv8
x=Conv2D(256, (1,1),padding='same')(x) #Conv9
x=Conv2D(512, (3, 3),padding='same')(x) #Conv10
x=Flatten()(x)
x=Dense(250,activation='sigmoid')(x)
x=Dropout(0.5)(x)
x=Dense(7*7*(1*5+3),activation='sigmoid')(x)
x=Reshape((7*7,(1*5+3)))(x)

#%%
# =============================================================================
# LOSS FUNCTION
# =============================================================================

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
    
    ### class_loss-----------------------------------mirar dimension de y_pred_class
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

#%%
# =============================================================================
# COMPILE
# =============================================================================

#TODO DEBE ESTAR EN LAS MISMAS DIMENSIONES Y CON EL MISMO DTYPE!!!!!
x_train.shape
x_train.dtype
y_true.shape
y_true.dtype
x.shape
x.dtype

#Prueba con pocas imagenes
x_train_lit=x_train[0:6]
x_train_lit.shape
y_true_lit=y_true[0:6]
y_true_lit.shape

model=Model(i,x)
model.compile(optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0),
              loss=loss_yolo,
              metrics=['acc']) 

model.summary()

#crear una imagen de prueba
#: b=np.random.random((49,8))

# =============================================================================
# FIT
# =============================================================================

model.fit(x_train,y_true, batch_size=50, epochs=8, verbose=1)


'''
We train the network for about 135 epochs on the training and validation data sets from PASCAL VOC 2007 and
2012. When testing on 2012 we also include the VOC 2007
test data for training. Throughout training we use a batch
size of 64, a momentum of 0.9 and a decay of 0.0005.
Our learning rate schedule is as follows: For the first
epochs we slowly raise the learning rate from 10−3
to 10−2
.
If we start at a high learning rate our model often diverges
due to unstable gradients. We continue training with 10−2
for 75 epochs, then 10−3
for 30 epochs, and finally 10−4
for 30 epochs.
To avoid overfitting we use dropout and extensive data
augmentation. A dropout layer with rate = .5 after the first
connected layer prevents co-adaptation between layers [18].
For data augmentation we introduce random scaling and
translations of up to 20% of the original image size. We
also randomly adjust the exposure and saturation of the image by up to a factor of 1.5 in the HSV color space.
'''

# =============================================================================
# PLOTING
# =============================================================================
img_box_coord=[]
for file in onlyfiles:
    if file[-1]=='l':
        tree = ET.parse(train_dir+file)
        root = tree.getroot()
        size_w=float(root[4][0].text)
        size_h=float(root[4][1].text)
        if size_w==0 or size_h==0:
            size=[250,250]
        else:
            size=[size_w,size_h]
        n_obj=(len(root))-6
        if n_obj==1:
            xmin=float(root[6][4][0].text)
            xmax=float(root[6][4][2].text)
            ymin=float(root[6][4][1].text)
            ymax=float(root[6][4][3].text)
            img_box_coord.append=[[xmin,xmax,ymin,ymax]]
