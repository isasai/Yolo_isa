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
import Functions as f
from Functions import parseo_class, parseo_coord, coord_CNN, convert, grid, loss_yolo
from sklearn.model_selection import train_test_split
#Para generar los datos necesitamos ejecutar las funciones del archivo Functions.py


# =============================================================================
# PATHS Y ARCHIVOS
# =============================================================================

train_dir=("C:/Users/adrin/Desktop/isa/Yolo_isa-master/Yolo_isa-master/pascal-voc-2007/Train/") 
test_dir=("C:/Users/adrin/Desktop/isa/Yolo_isa-master/Yolo_isa-master/pascal-voc-2007/Test/")

img_path=("VOCdevkit/VOC2007/JPEGImages/")
xml_path=("VOCdevkit/VOC2007/Annotations/")

onlyfiles_train_img = [f for f in listdir(train_dir+img_path) if isfile(join(train_dir+img_path, f))]
num_imagenes_train=len(onlyfiles_train_img)

onlyfiles_train_xml = [f for f in listdir(train_dir+xml_path) if isfile(join(train_dir+xml_path, f))]
num_xml_train=len(onlyfiles_train_xml)

onlyfiles_test_img = [f for f in listdir(test_dir+img_path) if isfile(join(test_dir+img_path, f))]
num_imagenes=len(onlyfiles_test_img)

onlyfiles_test_xml = [f for f in listdir(test_dir+xml_path) if isfile(join(test_dir+xml_path, f))]
num_imagenes=len(onlyfiles_test_xml)




###################  TRAIN  ###################################################

# =============================================================================
# BORRADO DE IMAGENES DE PERSONAS,TV,SOFA,pottedplant,,chair,"train","motorbike","car","bus","boat","bicycle","aeroplane"
# =============================================================================

#for file in onlyfiles_train_xml:
#    tree = ET.parse(train_dir+xml_path+file)
#    root = tree.getroot()
#    n_obj=(len(root))-6
#    count=0
#    if n_obj==1 and (root[6][0].text in ["bottle","diningtable","person","tvmonitor","sofa","pottedplant","train","motorbike","car","bus","boat","bicycle","aeroplane","chair"]):
#        count+=1
#        os.remove(train_dir+xml_path+file)
#        print(file+ " xml removed")
#        os.remove(train_dir+img_path+file[:-4]+".jpg")
#        print(file[:-4]+".jpg"+ " img removed")
#    elif n_obj>1:
#        for i in range(n_obj):
#            if root[i+6][0].text in ["bottle","diningtable","person","tvmonitor","sofa","pottedplant","train","motorbike","car","bus","boat","bicycle","aeroplane","chair"]:
#                count+=1
#                os.remove(train_dir+xml_path+file)
#                print(file+ " xml removed")
#                os.remove(train_dir+img_path+file[:-4]+".jpg")
#                print(file[:-4]+".jpg"+ " img removed")
#                break


# =============================================================================
# X DATA
# =============================================================================

train_img=[]
for file in onlyfiles_train_img:
    img=keras.preprocessing.image.load_img(
    path=train_dir + img_path + file,
    grayscale=False,
    color_mode='rgb',
    target_size=(288,288,3), #448  248
    interpolation='nearest')
    img_arr=keras.preprocessing.image.img_to_array(
    img,
    data_format=None,
    dtype='float64')
    img_arr=img_arr/255
    train_img.append(img_arr)

img=train_img[3]
import matplotlib.pyplot as plt
plt.imshow(img, cmap=plt.cm.binary)
plt.show()

x_data_train=np.asarray(train_img)
x_data_train.shape


# =============================================================================
# Y DATA
# =============================================================================

#Bucle para crear y_true con los datos de todas las imagenes
'''
El bucle inserta un vector [0,0,0,0,0,0,0,0] en y_true por cada celda
de la grid siempre que esa celda no contenga un objeto. Si contiene un objeto
(linea a<box[0]<b and c<box[1]<d  -> el centro de la caja que contine al
objeto esta entre las coordenadas de la celda) entonces introducimos
el vector de clases y coordenadas del objeto [c1,c2,c3,x,y,w,h,p])
'''
grid=f.grid(7)

y_data_train=[]        
for file in onlyfiles_train_xml:
    classes=f.parseo_class(train_dir+xml_path+file,class_dict) #Funcion en archivo Functions.py
    pr_obj=[1]
    boxes,size=f.parseo_coord(train_dir+xml_path+file) #Funcion en archivo Functions.py
    boxes_CNN,X,Y=f.coord_CNN(boxes,size,288) #Funcion en archivo Functions.py
    boxes_yolo=f.convert(288,boxes_CNN) #Funcion en archivo Functions.py
    for i in range(len(grid)):
        for j in range(len(grid)):
            #coordenadas de las celdas:
            xmin=grid[i][j][0]
            xmax=grid[i][j][1]
            ymin=grid[i][j][2]
            ymax=grid[i][j][3]
            insert=[0,0,0,0,0,0,0,0,0,0,0]
            count=0
            for box in boxes_yolo:
                if xmin<box[0]<xmax and ymin<box[1]<ymax:
                    insert=classes[count]+box+pr_obj
                    count+=1
                    break
                else:                
                    continue
            y_data_train.append(insert)

y_data_train=np.asarray(y_data_train)
y_data_train=np.reshape(y_data_train,(-1,(7*7),11))
y_data_train.shape


# =============================================================================
# SEPARAMOS EN TRAIN Y VALIDATION
# =============================================================================
x_train, x_val, y_true, y_val = train_test_split(x_data_train, y_data_train, test_size=0.10, random_state=2020)

x_train.shape
x_val.shape
y_true.shape
y_val.shape




###################  TEST  ###################################################=

# =============================================================================
# BORRADO DE IMAGENES DE PERSONAS,TV,SOFA,pottedplant,,chair,"train","motorbike","car","bus","boat","bicycle","aeroplane"
# =============================================================================

#for file in onlyfiles_test_xml:
#    tree = ET.parse(test_dir+xml_path+file)
#    root = tree.getroot()
#    n_obj=(len(root))-6
#    count=0
#    if n_obj==1 and (root[6][0].text in ["bottle","diningtable","person","tvmonitor","sofa","pottedplant","train","motorbike","car","bus","boat","bicycle","aeroplane","chair"]):
#        count+=1
#        os.remove(test_dir+xml_path+file)
#        print(file+ " xml removed")
#        os.remove(test_dir+img_path+file[:-4]+".jpg")
#        print(file[:-4]+".jpg"+ " img removed")
#    elif n_obj>1:
#        for i in range(n_obj):
#            if root[i+6][0].text in ["bottle","diningtable","person","tvmonitor","sofa","pottedplant","train","motorbike","car","bus","boat","bicycle","aeroplane","chair"]:
#                count+=1
#                os.remove(test_dir+xml_path+file)
#                print(file+ " xml removed")
#                os.remove(test_dir+img_path+file[:-4]+".jpg")
#                print(file[:-4]+".jpg"+ " img removed")
#                break

# =============================================================================
#  X DATA
# =============================================================================

test_img=[]
for file in onlyfiles_test_img:
    img=keras.preprocessing.image.load_img(
    path=test_dir+img_path+file,
    grayscale=False,
    color_mode='rgb',
    target_size=(288,288,3), #448
    interpolation='nearest')
    img_arr=keras.preprocessing.image.img_to_array(
    img,
    data_format=None,
    dtype='float64')
    img_arr=img_arr/255
    test_img.append(img_arr)

img=test_img[0]
import matplotlib.pyplot as plt
plt.imshow(img, cmap=plt.cm.binary)
plt.show()

x_test=np.asarray(test_img)
x_test.shape


# =============================================================================
# Y DATA
# =============================================================================        

##ESTO SIRVE PARA EL DATASET DE FRUTAS-> Buscamos imagenes con size 0 (no valen, hay que eliminarlas)
#for file in onlyfiles_test:  
#    if file[-1]=='l':
#        tree = ET.parse(test_dir+file)
#        root = tree.getroot()
#        size_w=float(root[4][0].text)
#        size_h=float(root[4][1].text)
#        if size_w == 0 or size_h == 0:
#            print(file)          
        
     
y_test=[]        
for file in onlyfiles_test_xml:
    classes=parseo_class(test_dir+xml_path+file,class_dict) #Funcion en archivo Functions.py
    pr_obj=[1]
    boxes,size=parseo_coord(test_dir+xml_path+file) #Funcion en archivo Functions.py
    boxes_CNN,X,Y=coord_CNN(boxes,size,288) #Funcion en archivo Functions.py
    boxes_yolo=convert(288,boxes_CNN) #Funcion en archivo Functions.py
    for i in range(len(grid)):
        for j in range(len(grid)):
            #coordenadas de las celdas:
            xmin=grid[i][j][0]
            xmax=grid[i][j][1]
            ymin=grid[i][j][2]
            ymax=grid[i][j][3]
            insert=[0,0,0,0,0,0,0,0,0,0,0]
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
y_test=np.reshape(y_test,(-1,(7*7),11))
y_test.shape
