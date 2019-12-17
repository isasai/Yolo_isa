# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:46:43 2019

@author: adrin
"""

# =============================================================================
# Guardar e importar variables
# =============================================================================

import pickle
with open('C:/Users/adrin/Desktop/isa/Yolo_isa-master/Yolo_isa-master/x2.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(x, f)
    
with open('train.pickle', 'rb') as f:
    y_pred = pickle.load(f)
    
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
