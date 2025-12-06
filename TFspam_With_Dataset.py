import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("spam_ham_dataset.csv")
df["label_num"] = df["label"].apply(lambda x: 1 if x == "spam" else 0)

X = df["text"].astype(str).values
y = df["label_num"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = keras.layers.TextVectorization(
    max_tokens=20000,
    output_mode="tf_idf"
)
vectorizer.adapt(X_train)

inputs = keras.Input(shape=(1,), dtype=tf.string)
x = vectorizer(inputs)
x = keras.layers.Dense(64, activation="relu")(x)
x = keras.layers.Dense(32, activation="relu")(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    verbose=2
)

y_prob = model.predict(X_test)
y_pred = (y_prob >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["ham","spam"]))

sample = ["Congratulations! You have won a free prize. Click now!"]
print("Sample prediction:", model.predict(tf.constant(sample)))
if model.predict(tf.constant(sample)) >= 0.5:
  print("The email is SPAM")
else:
  print("The email is NOT SPAM")

