import DataSource
import Model
import Loss
import Trainer

import tensorflow as tf

dso: DataSource.DataSourceObject = DataSource.Load()
model = Model.CreateSequential()
model.summary()

loss_fn = Loss.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

Trainer.TrainAsync(model, dso, loss_fn, optimizer).Visualize() 