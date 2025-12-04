import numpy as np
import tensorflow as tf
 
x = np.array([[0.5, 1.2, -0.7]], dtype=np.float32)
 
W1 = np.array([
    [0.4, -0.2, 0.1],
    [-0.3, 0.8, 0.5]
], dtype=np.float32)
 
b1 = np.array([0.1, -0.2], dtype=np.float32)
 
W2 = np.array([[0.6],
               [-0.4]], dtype=np.float32)
 
b2 = np.array([0.2], dtype=np.float32)
 
W1_tf = W1.T
W2_tf = W2
 
inputs = tf.keras.Input(shape=(3,))
hidden = tf.keras.layers.Dense(2, activation='sigmoid', name="hidden")(inputs)
output = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(hidden)
model = tf.keras.Model(inputs, output)
 
model.get_layer("hidden").set_weights([W1_tf, b1])
model.get_layer("output").set_weights([W2_tf, b2])
 
print("TensorFlow output:", model.predict(x))
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
 
z1 = x.dot(W1_tf) + b1
a1 = sigmoid(z1)
 
z2 = a1.dot(W2_tf) + b2
a2 = sigmoid(z2)
 
print("Hidden z1:", z1)
print("Hidden a1:", a1)
print("Output z2:", z2)
print("Output a2:", a2)
 
if a2 > 0.5:
  print("The email is SPAM")
else:
  print("The email is NOT SPAM")
 
