
import tensorflow as tf
import numpy as np

celcius_q = np.array([-40,-10,0,8,15,22,38],dtype=float)
far_q = np.array([-40,14,32,46,59,72,100],dtype=float)

layer1 = tf.keras.layers.Dense(units=1,input_shape=[1])

model = tf.keras.Sequential([layer1])

model.compile(loss='mean_squared_error',optimizer = tf.keras.optimizers.Adam(0.1))

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])

history = model.fit(celcius_q,far_q,epochs=500,verbose=False)

print(model.predict([100])) # sampe input for prediction 
