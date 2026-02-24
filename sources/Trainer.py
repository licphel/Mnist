import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import DataSource
import Model
import Loss

class TrainingHistory:
    def __init__(self, hist):
        self.history = hist

    def Visualize(self):
        history = self.history
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy over Epochs')

        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss over Epochs')

        plt.tight_layout()
        plt.show()


def TrainAsync(model: models.Model, dso: DataSource.DataSourceObject, loss_fn, optimizer) -> TrainingHistory:
    model.compile(metrics=['accuracy'])

    (trd, ted) = dso.Packed()

    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_fn(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        correct_predictions = tf.equal(
            tf.argmax(predictions, axis=1), 
            tf.argmax(labels, axis=1)
        )
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        
        return loss, accuracy

    @tf.function
    def val_step(images, labels):
        predictions = model(images)
        loss = loss_fn(labels, predictions)
        
        correct_predictions = tf.equal(
            tf.argmax(predictions, axis=1), 
            tf.argmax(labels, axis=1)
        )
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return loss, accuracy

    for epoch in range(20):
        print(f'Epoch {epoch + 1}/20')
        
        epoch_loss = []
        epoch_acc = []
        for images, labels in trd:
            loss, acc = train_step(images, labels)
            epoch_loss.append(loss.numpy())
            epoch_acc.append(acc.numpy())
        
        avg_train_loss = np.mean(epoch_loss)
        avg_train_acc = np.mean(epoch_acc)
        
        val_losses = []
        val_accs = []
        for images, labels in ted:
            loss, acc = val_step(images, tf.cast(labels, tf.int64))
            val_losses.append(loss.numpy())
            val_accs.append(acc.numpy())
        
        avg_val_loss = np.mean(val_losses)
        avg_val_acc = np.mean(val_accs)

        history['loss'].append(avg_train_loss)
        history['accuracy'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(avg_val_acc)
        
        print(f'Training: Loss = {avg_train_loss:.4f}, Acc = {avg_train_acc:.4f}')
        print(f'Validation: Loss = {avg_val_loss:.4f}, Acc = {avg_val_acc:.4f}')
        print()

    return TrainingHistory(history)