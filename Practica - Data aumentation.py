# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:59:45 2019

@author: irodr
"""

# =============================================================================
# To move a file in Python, we will need to import the os and shutil 
# modules that provide us the ability to copy, move, and remove files 
# in Python. Both of these modules provide methods to do so, although 
# in many cases the shutil module has more convenient methods. Keep in
# mind that the destination directory needs to exist in order for this
# example to work. Once youve set up the directories "d1" and "d2" 
# (or just changed the example to fit your directory structure), run 
# the code. Now check out the "d2" directory and you should see 
# the xfile.txt if now present in that directory.
# 
# import os
# import shutil
# # Move a file by renaming it's path
# os.rename('/Users/billy/d1/xfile.txt', '/Users/billy/d2/xfile.txt')
# # Move a file from the directory d1 to d2
# shutil.move('/Users/billy/d1/xfile.txt', '/Users/billy/d2/xfile.txt')
# for i in onlyfiles:
#     if i[:3]=="cat":
#         shutil.move(train_dir+i, train_dir+"cats/"+i)
#     if i[:3]=="dog":
#         shutil.move(train_dir+i, train_dir+"dogs/"+i)
# =============================================================================
# =============================================================================
# COMO SE USA IMAGEDATAGENERATOR
# =============================================================================
import os
import matplotlib.pyplot as plt

base_dir = "C:/Users/irodr/Documents/NEOLAND/Machine Learning/9- Redes neuronales/Redes convolucionales/dogs-vs-cats/dataset/ejercicio1/"
train_dir = os.path.join(base_dir, 'training_set\\')
test_dir = os.path.join(base_dir, 'test_set\\')

# ImageDataGenerator es una función que crea/devuelve generadores
from keras.preprocessing.image import ImageDataGenerator

#Definimos los parametros que usaremos para modificar las imagenes con la
#clase ImageDataGenerator
#Los parametros se aplicaran de forma aleatoria
train_datagen = ImageDataGenerator(rotation_range=40,# valor entre 0 y 180 que harà rotar de forma random las imagenes
                             width_shift_range=0.2, # transladar la imagen horizontalmente
                             height_shift_range=0.2,# transladar la imagen verticalmente
                             shear_range=0.2, # sharing transformations (aplicaciones lineales)
                             zoom_range=0.2, # hacer zoom a la imagen
                             horizontal_flip=True, # hacer el espejo
                             fill_mode='nearest') # la forma de rellenar los pixeles que quedan vacios tras las transformaciones
#Funcion que busca en el directorio archivos en las carpetas existentes ,
#diferenciandolas por categorias y crea batches de fotos modificadas
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),# Transforma todas las imagenes en 150x150
        batch_size=20,#Crea batches de 20 img modificadas
        class_mode='binary')# lo transformamos a binario ya que utilizaremos metricas de clasificación binaria

#Iteramos tantas veces como bathes de fotos queramos crear
i=0
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    print(i)
    i+=1
    if i==4:
        break
    
#Print de una imagen cualquiera del data_batch    
data_batch[7]
una_imagen=data_batch[7]
una_imagen=una_imagen.astype(int)
import matplotlib.pyplot as plt
plt.imshow(una_imagen, cmap=plt.cm.binary)
plt.show()

#%%
# =============================================================================
# MODELO ANADIENDO IMAGENES DISTORSIONADAS
# =============================================================================

#CREAMOS DATA-GENERATOR
#IMAGEDATAGENERATOR va a hacer distintas cosas:
    # - Genera imagenes nuevas distorsionadas a partir de un dataset (debemos
    #tenerlo organizado en diferentes carpetas por categoria)
    # - Si metemos el objeto train_datagen.flow_from_directory directamente
    #en el fit del modelo, directamente se nos generara la matriz de imagenes
    #(es decir, nos carga cada imagen, la transforma en array y nos la 
    #justa en la matriz de dimensiones que necesita el modelo)

#base_dir = "C:/Users/irodr/Documents/NEOLAND/Machine Learning/9- Redes neuronales/Redes convolucionales/dogs-vs-cats/dataset/ejercicio1/"

#Creamos datagenerator
train_datagen = ImageDataGenerator(rotation_range=40,# valor entre 0 y 180 que harà rotar de forma random las imagenes
                             width_shift_range=0.2, # transladar la imagen horizontalmente
                             height_shift_range=0.2,# transladar la imagen verticalmente
                             shear_range=0.2, # sharing transformations (aplicaciones lineales)
                             zoom_range=0.2, # hacer zoom a la imagen
                             horizontal_flip=True, # hacer el espejo
                             fill_mode='nearest') # la forma de rellenar los pixeles que quedan vacios tras las transformaciones

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),# Transforma todas las imagenes en 150x150
        batch_size=30,#Crea batches de 20 img modificadas
        class_mode='binary')

#%%CREAMOS ESTRUCTURA DEL MODELO

from keras import models
from keras import layers
from keras import regularizers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2))) # 13x13x32
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu')) # 11x11x64
model.add(layers.MaxPooling2D((2, 2)))# 5x5x64
model.add(layers.Conv2D(64, (3, 3), activation='relu'))# 3x3x64
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='sigmoid'))
model.add(layers.Dense(1, activation='sigmoid'))

from keras import losses
from keras import metrics
from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,#binary_crossentropy para una sola variable resultado de 1 y 0
                                              #categorical_corssentropy para variables resultado categoricas
              metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=250,#Tendria sentido utilizar esto sobre un dataset
                        #de (ej) 2000 datos? Tendriamos que poner un 
                        #step de mas de 2000 para asegurarnos que no 
                        #perdemos nuestra variedad de datos???
    epochs=3)

#TRAIN SET
from os import listdir
from os.path import isfile, join
import numpy as np
onlyfiles_test = [f for f in listdir(test_dir) if isfile(join(test_dir, f))]

label=[]
list_img=[]
for file in range(len(onlyfiles_test)):
    img=keras.preprocessing.image.load_img(path=test_dir + onlyfiles_test[file],grayscale=False,color_mode='rgb',target_size=(150,150,3),interpolation='nearest')
    img_arr=keras.preprocessing.image.img_to_array(img)
    img_arr=img_arr/255
    list_img.append(img_arr)
    label.append(onlyfiles_test[file][0:3])

len(label)
label
len(list_img)
list_img

y_test=np.asarray(label)
y_test=np.where(y_test=="dog",1,0)
x_test=np.asarray(list_img)

#Evaluate
#Returns the loss value & metrics values for the model in test mode.
model.evaluate(x_test,y_test)
#Generates output predictions for the input samples.
model.predict(x_test)
model.predict_classes(x_test)
