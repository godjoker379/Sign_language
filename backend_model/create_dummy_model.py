import os
import tensorflow as tf
import numpy as np

# A very simple model to classify 21 landmarks * 3 coordinates = 63 features
# We assume 3 classes for now: A, B, C

num_classes = 3
input_shape = (63,)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train on some dummy data just to have some weights
dummy_x = np.random.rand(100, 63)
dummy_y = np.random.randint(0, num_classes, size=(100,))

model.fit(dummy_x, dummy_y, epochs=1)

# Save the model
model.save('model.h5')

print("Model saved to model.h5. Now run pip install tensorflowjs and use tensorflowjs_converter to convert it.")
