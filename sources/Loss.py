import tensorflow as tf
from tensorflow import Tensor

# Mean |pred - true|
class AbsMean(tf.keras.losses.Loss):
    def __init__(self, name='AbsMean'):
        super().__init__(name=name)
    
    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1)
    
# Sum |pred - true|
class AbsSum(tf.keras.losses.Loss):
    def __init__(self, name='AbsSum'):
        super().__init__(name=name)
    
    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1)
    
# Mean |pred - true|^2
class AbsSqrMean(tf.keras.losses.Loss):
    def __init__(self, name='AbsMean'):
        super().__init__(name=name)
    
    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return tf.reduce_mean(tf.pow(y_pred - y_true, 2), axis=-1)
    
# Sum |pred - true|^2
class AbsSqrSum(tf.keras.losses.Loss):
    def __init__(self, name='AbsMean'):
        super().__init__(name=name)
    
    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return tf.reduce_sum(tf.pow(y_pred - y_true, 2), axis=-1)
    
# MEAN exp(|pred - true|)
class AbsExpMean(tf.keras.losses.Loss):
    def __init__(self, name='AbsExpMean'):
        super().__init__(name=name)
    
    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return tf.exp(tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1))
    
# SUM exp(|pred - true|)
class AbsExpSum(tf.keras.losses.Loss):
    def __init__(self, name='AbsExpSum'):
        super().__init__(name=name)
    
    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return tf.exp(tf.reduce_sum(tf.abs(y_pred - y_true), axis=-1))

SparseCategoricalCrossentropy = tf.keras.losses.SparseCategoricalCrossentropy
CategoricalCrossentropy = tf.keras.losses.CategoricalCrossentropy