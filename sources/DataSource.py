from typing import List

from tensorflow import keras
import tensorflow as tf

# Training-testing split data source object
class DataSourceObject:
    def __init__(self, x, y, xTest, yTest):
        self.x = x
        self.y = y
        self.xTest = xTest
        self.yTest = yTest

    def ToTensorflowObject(self) -> List[tf.data.Dataset]:
        trainer = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        tester = tf.data.Dataset.from_tensor_slices((self.xTest, self.yTest))

        # Shuffle data.
        trainer = trainer.shuffle(10000).batch(64)
        tester = tester.shuffle(2000).batch(64)

        return [trainer, tester]

def Load() -> DataSourceObject:
    # Load the dataset.
    (x, y), (xTest, yTest) = keras.datasets.mnist.load_data()

    # choose the top 1/10 data (I want to train faster)
    x = x[:6000]
    y = y[:6000]

    # Preprocess the dataset.
    x = x.reshape(6000, 28, 28, 1).astype("float32") / 255.0
    xTest = xTest.reshape(10000, 28, 28, 1).astype("float32") / 255.0
    # do not convert to tensor since we switch to SparseCategoricalCrossentropy
    # y = keras.utils.to_categorical(y, 10)
    # yTest = keras.utils.to_categorical(yTest, 10)

    return DataSourceObject(x, y, xTest, yTest)