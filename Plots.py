# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:39:57 2019

@author: adrin
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

#%%
#Coord caja convertidas a escala 0-1 (x,y,w,h)
boxes_yolo[0]
#Factores de conversion para reconvertir a tamano de imagen
X
Y

###Cargamos la imagen y las cajas
im = np.array(Image.open('C:/Users/irodr/Documents/NEOLAND/Proyecto/YOLO/fruit-images-for-object-detection/train/apple_3.jpg'), dtype=np.uint8)

# Create figure and axes
fig,ax = plt.subplots()

# Display the image
ax.imshow(im)

# Create a Rectangle patch
rect = patches.Rectangle((
    (0.7254*(448/X))-((0.283*(448/X))/2),
    (0.8093*(448/Y))-((0.381*(448/Y))/2)),
    0.283*(448/X),
    0.3813*(448/Y),
    linewidth=1,edgecolor='r',facecolor='none')
#rect=x-w/2,y-(h/2,w,h)
#fact de conversion para volver a coordenadas iniciales de la foto 448/X y 448/Y

# Add the patch to the Axes
ax.add_patch(rect)
plt.show()

#%%
for img in range(x_test.shape[0]):
    fruit=x_test[img]
    fig,ax = plt.subplots()
    plt.imshow(fruit, cmap=plt.cm.binary)
    for box in y_pred[0]:
        rect = patches.Rectangle(((box[0]*(448/X))-((box[2]*(448/X))/2),(box[1]*(448/Y))-((box[3]*(448/Y))/2)),box[2]*(448/X),box[3]*(448/Y),linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.show()

y_true[0][...,0:3]
y_pred[0][...,0:3]

y_true[1][...,-1]
y_pred[1][...,-1]

x_test[0].shape

#Bucle para plotear las cajas de y_pred
boxes,size=parseo_coord("C:/Users/adrin/Desktop/isa/Yolo_isa-master/Yolo_isa-master/fruit-images-for-object-detection/test/apple_85.xml") 
boxes_CNN,X,Y=coord_CNN(boxes,size,288)  
fruit=x_test[0]
fig,ax = plt.subplots()
plt.imshow(fruit, cmap=plt.cm.binary)
for cell in y_pred[0]:
    rect = patches.Rectangle((((cell[3]*448/X))-((cell[5]*(448/X))/2),(cell[4]*(448/Y))-((cell[6]*(448/Y))/2)),cell[5]*(448/X),cell[6]*(448/Y),linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
plt.show()

