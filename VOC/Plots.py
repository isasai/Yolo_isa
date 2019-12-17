# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:39:57 2019

@author: adrin
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import pickle

#%%
#Coord caja convertidas a escala 0-1 (x,y,w,h)
boxes_yolo[0]
#Factores de conversion para reconvertir a tamano de imagen
X
Y

###Cargamos la imagen y las cajas desde el jp y xml
im = np.array(Image.open('C:/Users/irodr/Documents/NEOLAND/Proyecto/YOLO/YOLO_ISA/fruit-images-for-object-detection/test/apple_85.jpg'), dtype=np.uint8)
# Create figure and axes
fig,ax = plt.subplots()
# Display the image
ax.imshow(im)
# # Create a Rectangle patch
rect = patches.Rectangle((
    88,
    41),
    (415-88),
    (393-41),
    linewidth=1,edgecolor='r',facecolor='none')
# #rect=x-w/2,y-(h/2,w,h)
# #fact de conversion para volver a coordenadas iniciales de la foto 448/X y 448/Y
# # Add the patch to the Axes
ax.add_patch(rect)
plt.show()

  
##############################################################
#Obtenemos y_pred
with open('C:/Users/irodr/Documents/NEOLAND/Proyecto/YOLO/YOLO_ISA/y_pred.pkl', 'rb') as f:
    y_pred = pickle.load(f)

y_test[0][...,0:3]
y_pred[0][...,0:3]


y_test[0][...,-1]
y_pred[0][...,-1]

x_test[0].shape

#Bucle para plotear las cajas de y_pred
#Hay que reconvertir las coordenadas de y_pred a las coordenadas
#de la imagen que representan con los facores X e Y de la imagen
im = np.array(Image.open('C:/Users/irodr/Documents/NEOLAND/Proyecto/YOLO/YOLO_ISA/fruit-images-for-object-detection/test/apple_86.jpg'), dtype=np.uint8)
fig,ax = plt.subplots()
ax.imshow(im)
boxes,size=parseo_coord("C:/Users/irodr/Documents/NEOLAND/Proyecto/YOLO/YOLO_ISA/fruit-images-for-object-detection/test/apple_86.xml") 
boxes_CNN,X,Y=coord_CNN(boxes,size,288)  
for cell in y_pred[1]:
    rect_pred = patches.Rectangle(((cell[3]-(cell[5]/2))*(288/X),(cell[4]-(cell[6]/2))*(288/Y)),cell[5]*(288/X),cell[6]*(288/Y),linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect_pred)
for cell in y_test[8]:
    rect_true = patches.Rectangle(((cell[3]-(cell[5]/2))*(288/X),(cell[4]-(cell[6]/2))*(288/Y)),cell[5]*(288/X),cell[6]*(288/Y),linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rect_true)
plt.show()

# #Bucle para plotear las cajas de y_pred
# #Hay que reconvertir las coordenadas de y_pred a las coordenadas
# #de la imagen que representan con los facores X e Y de la imagen
# boxes,size=parseo_coord("C:/Users/irodr/Documents/NEOLAND/Proyecto/YOLO/YOLO_ISA/fruit-images-for-object-detection/test/apple_85.xml") 
# boxes_CNN,X,Y=coord_CNN(boxes,size,288)  
# fruit=x_test[7]
# fig,ax = plt.subplots()
# plt.imshow(fruit, cmap=plt.cm.binary)
# for cell in y_pred[0]:
#     rect_pred = patches.Rectangle((((cell[3]*448/X))-((cell[5]*(448/X))/2),(cell[4]*(448/Y))-((cell[6]*(448/Y))/2)),cell[5]*(448/X),cell[6]*(448/Y),linewidth=1,edgecolor='r',facecolor='none')
#     ax.add_patch(rect_pred)
# for cell in y_test[7]:
#     rect_true = patches.Rectangle((((cell[3]*448/X))-((cell[5]*(448/X))/2),(cell[4]*(448/Y))-((cell[6]*(448/Y))/2)),cell[5]*(448/X),cell[6]*(448/Y),linewidth=1,edgecolor='b',facecolor='none')
#     ax.add_patch(rect_true)
# plt.show()

