import tensorflow as tf
import numpy as np

X = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([0.0, 1.0, 2.0, 3.0])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, verbose=1)

model.save(r"C:\Crop_AI_System\models\model.h5")