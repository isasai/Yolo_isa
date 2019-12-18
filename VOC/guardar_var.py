# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:46:43 2019

@author: adrin
"""

# =============================================================================
# Importar variables
# =============================================================================


    
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


# =============================================================================
# Cargar modelo
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

#---------------------------------------------------------
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
