import DataSource
import numpy as np
from tensorflow import keras
from keras import layers

dso: DataSource.DataSourceObject = DataSource.Load()

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])


model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


history = model.fit(
    dso.x, dso.y,
    batch_size=128,
    epochs=10,
    validation_split=0.2
)


test_loss, test_acc = model.evaluate(dso.xTest, dso.yTest)
print(f"Test Accuracy: {test_acc:.4f}")