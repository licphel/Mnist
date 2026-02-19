import tensorflow as tf
from tensorflow import Tensor

class MAE(tf.keras.losses.Loss):
    def __init__(self, name='MAE'):
        super().__init__(name=name)
    
    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1)