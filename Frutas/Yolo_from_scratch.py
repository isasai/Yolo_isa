# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:56:38 2019

@author: irodr
"""

'''
YOLO V1 ALGORITHM WITH KERAS - FROM SCRATCH
'''

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
import json
import xml.etree.ElementTree as ET

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
#
##NET STRUCTURE - YOLO V1
#tf.keras.backend.set_floatx('float64')

#i=Input(shape=(328,328,3))
#x=Conv2D(64, (7,7))(i) #Conv1  ###1 -> 224
#x=keras.layers.LeakyReLU(alpha=0.3)(x)
#x=MaxPooling2D((2, 2))(x) ###2 ->112
#x=Conv2D(192, (3, 3))(x) #Conv2
#x = keras.layers.LeakyReLU(alpha=0.3)(x)
#x=MaxPooling2D((2, 2))(x) ###3 -> 56
#x=Conv2D(128, (1,1))(x) #Conv3
#x=Conv2D(256, (3, 3))(x) #Conv4
#x=Conv2D(256, (1,1))(x) #Conv5
#x=Conv2D(512, (1,1))(x) #Conv6
#x = keras.layers.LeakyReLU(alpha=0.3)(x)
#x=MaxPooling2D((2, 2))(x) ###4 -> 28
#x=Conv2D(256, (1,1))(x) #Conv7
#x=Conv2D(512, (3, 3))(x) ##Conv8
#x=Conv2D(256, (1,1))(x) #Conv9
#x=Conv2D(512, (3, 3))(x) #Conv10
#x=Conv2D(256, (1,1))(x) #Conv11
#x=Conv2D(512, (3, 3))(x) #Conv12
#x=Conv2D(256, (1,1))(x) #Conv13
#x=Conv2D(512, (3, 3))(x) #Conv14
#x=Conv2D(512, (1,1))(x) #Conv15
#x=Conv2D(1024, (3,3))(x) #Conv16
#x=keras.layers.LeakyReLU(alpha=0.3)(x)
#x=MaxPooling2D((2, 2))(x) ###5 -> 14
#x=Conv2D(512, (1,1))(x) #Conv17
#x=Conv2D(1024, (3, 3))(x) #Conv18
#x=Conv2D(512, (1,1))(x) #Conv19
#x=Conv2D(1024, (3, 3))(x) #Conv20
#x=Conv2D(1024, (3, 3))(x) #Conv21
#x=Conv2D(1024, (3, 3))(x) #Conv22 ###6 -> 7
#x=Conv2D(1024, (3, 3))(x) #Conv23
#x=Conv2D(1024, (3, 3))(x) #Conv24
#x=keras.layers.LeakyReLU(alpha=0.3)(x)
#x=Flatten()(x)
#x=Dense(256,activation='sigmoid')(x)
#x=Dropout(0.5)(x)
#x=Dense(7*7*(1*5+3),activation='sigmoid')(x)
#x=Reshape((7*7,(1*5+3)))(x) 
##Me falta poner data augmentation!!!
#
##grid=numero de celdas en las que queremos evaluar si hay objetos
##B=1: numero de predicciones. Voy a empezar probando con 1
##C=3: numero de clases, en el caso de mi dataset son 3
#
#'''
#The trick is to have the network to predict coordinates that are limited in 
#their range to a single cell. To limit the values that the network can 
#output we use sigmoid on it’s output. Sigmoid is a common activation function 
#in deep learning but here we use it for it’s property of only taking values 
#between 0 and 1.
#Since we only predict value between 0 and 1, we will consider width and height
#to be in the system of coordinate of the image. A box of image width will have 
#a width of 1.
#'''
#%%
#SIMPPLIFIED NET
tf.keras.backend.set_floatx('float64')

i=Input(shape=(288,288,3))
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

#%%
# =============================================================================
# COMPILE
# =============================================================================

#CHECK: TODO DEBE ESTAR EN LAS MISMAS DIMENSIONES Y CON EL MISMO DTYPE!!!!!
x_train.shape
x_train.dtype
x_val.shape
x_val.dtype
y_true.shape
y_true.dtype
y_val.shape
y_val.dtype
x_test.shape
x_test.dtype
y_test.shape
y_test.dtype
x.shape
x.dtype

#--------------------------------------
#Dataset de prueba con pocas imagenes
x_train_lit=x_train[0:6]
x_train_lit.shape
y_true_lit=y_true[0:6]
y_true_lit.shape
#--------------------------------------
 
model=Model(i,x)
model.compile(optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0005),
              loss=loss_yolo,
              metrics=['acc']) 

model.summary()

#crear una imagen de prueba
#: b=np.random.random((49,8))

# =============================================================================
# FIT 
# =============================================================================

#Creamos datagenerator
##DESCARTADO -> No parece que este implementado una fucnion que tambien aplique 
#las tranformaciones a las bouding boxes

history=model.fit(x_train,y_true, batch_size=30, epochs=35, validation_data=(x_val,y_val),verbose=1)


'''
We train the network for about 135 epochs on the training and validation data sets from PASCAL VOC 2007 and
2012. When testing on 2012 we also include the VOC 2007
test data for training. Throughout training we use a batch
size of 64, a momentum of 0.9 and a decay of 0.0005.
Our learning rate schedule is as follows: For the first
epochs we slowly raise the learning rate from 10−3
to 10−2.
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
# EVALUATE AND PREDICT
# =============================================================================

# The returned "history" object holds a record
# of the loss values and metric values during training
print('\nhistory dict:', history.history)

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = new_model.evaluate(x_test, y_test)
print('test loss, test acc:', results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print('\n# Generate predictions for 3 samples')
y_pred = model.predict(x_test[:3])
print('predictions shape:', predictions.shape)

##PLOTS
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


# =============================================================================
# SAVING MODEL 
# =============================================================================

#Model and weights
model_name="simple1_30_35.h5"
model.save("C:/Users/adrin/Desktop/isa/Yolo_isa-master/Yolo_isa-master/MODELOS/" + model_name)

#Model
config_simple1_30_35 = model.to_json()
config_name="config_simple1_30_35"
with open("C:/Users/adrin/Desktop/isa/Yolo_isa-master/Yolo_isa-master/MODELOS/" + config_name, 'w+') as f:
    json.dump(config_simple1_30_35, f)

#Weights
weights_name="w_simple1_30_35.h5"
model.save_weights("C:/Users/adrin/Desktop/isa/Yolo_isa-master/Yolo_isa-master/MODELOS/" + weights_name)

# =============================================================================
# LOADING MODEL
# =============================================================================
#Complete
model_name="simple_30_35.h5"
model = keras.models.load_model("C:/Users/adrin/Desktop/isa/Yolo_isa-master/Yolo_isa-master/models/complete/" + model_name, custom_objects={"loss_yolo":loss_yolo})

#Net and weights
config = model.get_config()
reinitialized_model = keras.Model.from_config("C:/Users/adrin/Desktop/isa/Yolo_isa-master/Yolo_isa-master/models/net/"+"w_simple_30_35.h5")
reinitialized_model = keras.models.model_from_json(config_simple_30_35)

weights = model.get_weights()
model.set_weights(weights) 

# Guardar configuración JSON en el disco
json_config = model.to_json()
with open('model_config.json', 'w') as json_file:
    json_file.write(json_config)
# Guardar pesos en el disco
model.save_weights('path_to_my_weights.h5')

# Recargue el modelo de los 2 archivos que guardamos
with open('model_config.json') as json_file:
    json_config = json_file.read()
new_model = keras.models.model_from_json(json_config)
new_model.load_weights('path_to_my_weights.h5')

# Verifique que el estado esté preservado
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

# Tenga en cuenta que el optimizador no se conservo.
