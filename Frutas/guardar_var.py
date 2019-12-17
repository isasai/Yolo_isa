# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:46:43 2019

@author: adrin
"""

# =============================================================================
# Guardar e importar variables
# =============================================================================

import pickle
with open('C:/Users/adrin/Desktop/isa/Yolo_isa-master/Yolo_isa-master/y_pred.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(y_pred, f)
    
with open('train.pickle', 'rb') as f:
    y_pred = pickle.load(f)
    
