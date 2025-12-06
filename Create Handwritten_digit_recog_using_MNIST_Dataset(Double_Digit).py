import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2

# 1. Load single-digit MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

x_train = np.expand_dims(x_train, -1)  # (60000, 28, 28, 1)
x_test  = np.expand_dims(x_test, -1)   # (10000, 28, 28, 1)

# 2. Build a synthetic "two-digit" dataset from MNIST
def make_two_digit_dataset(x, y, n_samples):
    # x: (N, 28, 28, 1), y: (N,)
    images = []
    labels = []
    n = x.shape[0]
    for _ in range(n_samples):
        idx1 = np.random.randint(0, n)
        idx2 = np.random.randint(0, n)
        img1 = x[idx1]      # left digit
        img2 = x[idx2]      # right digit
        d1 = int(y[idx1])
        d2 = int(y[idx2])
        # concatenate horizontally: result shape (28, 56, 1)
        two_digit_img = np.concatenate([img1, img2], axis=1)
        images.append(two_digit_img)
        # label is two-digit number: e.g. 2 and 7 -> 27
        labels.append(10 * d1 + d2)
    images = np.stack(images, axis=0)   # (n_samples, 28, 56, 1)
    labels = np.array(labels, dtype=np.int32)  # 0..99
    return images, labels

# you can adjust these sizes if training is slow
N_TRAIN = 30000
N_TEST  = 5000

X2_train, y2_train = make_two_digit_dataset(x_train, y_train, N_TRAIN)
X2_test,  y2_test  = make_two_digit_dataset(x_test,  y_test,  N_TEST)

# 3. Build CNN for two-digit images (input: 28x56x1, output: 100 classes)
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 56, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(100, activation='softmax')  # 100 classes: 00..99
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 4. Train on synthetic two-digit dataset
model.fit(
    X2_train, y2_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=2
)

# 5. Evaluate
test_loss, test_acc = model.evaluate(X2_test, y2_test, verbose=2)
print("\nTwo-digit test accuracy:", test_acc)

# 6. Predict on your own 2-digit image from disk
img_path = input("\nEnter 2-digit image path: ")

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError(f"Could not read image at: {img_path}")

# resize to match our two-digit MNIST shape: width=56, height=28
img = cv2.resize(img, (56, 28))  # (width=56, height=28)

# normalize and reshape
img = img.astype("float32") / 255.0
img = img.reshape(1, 28, 56, 1)

# predict class 0..99 and decode as two-digit number
pred = model.predict(img)
cls = int(np.argmax(pred))       # 0..99
tens = cls // 10
ones = cls % 10
print("\nPredicted number is:", f"{tens}{ones}")
