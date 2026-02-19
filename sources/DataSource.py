from tensorflow import keras

# Training-testing split data source object
class DataSourceObject:
    def __init__(self, x, y, xTest, yTest):
        self.x = x
        self.y = y
        self.xTest = xTest
        self.yTest = yTest

def Load() -> DataSourceObject:
    # Load the dataset.
    (x, y), (xTest, yTest) = keras.datasets.mnist.load_data()

    # Preprocess the dataset.
    x = x.reshape(60000, 784).astype("float32") / 255
    xTest = xTest.reshape(10000, 784).astype("float32") / 255
    y = keras.utils.to_categorical(y, 10)
    yTest = keras.utils.to_categorical(yTest, 10)

    return DataSourceObject(x, y, xTest, yTest)