import DataSource
import Model
import Loss

import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt

dso: DataSource.DataSourceObject = DataSource.Load()
model = Model.CreateSequential()
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

(trd, ted) = dso.ToTensorflowObject()

# 定义损失函数和优化器
loss_fn = Loss.Exp()
optimizer = tf.keras.optimizers.Adam()

# 创建字典来记录训练过程中的指标
history = {
    'loss': [],           # 训练损失
    'accuracy': [],       # 训练准确率
    'val_loss': [],       # 验证损失
    'val_accuracy': []    # 验证准确率
}

# 自定义训练步骤
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # 正确计算准确率：labels 是 one-hot，所以也要用 argmax
    correct_predictions = tf.equal(
        tf.argmax(predictions, axis=1), 
        tf.argmax(labels, axis=1)  # 注意：labels 是 one-hot，需要 argmax
    )
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    return loss, accuracy

# 验证步骤（不需要梯度）
@tf.function
def val_step(images, labels):
    predictions = model(images)
    loss = loss_fn(labels, predictions)
    
    # 同样，labels 是 one-hot
    correct_predictions = tf.equal(
        tf.argmax(predictions, axis=1), 
        tf.argmax(labels, axis=1)
    )
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return loss, accuracy

# 训练循环
for epoch in range(20):
    print(f'Epoch {epoch + 1}/20')
    
    # 训练阶段
    epoch_loss = []
    epoch_acc = []
    for images, labels in trd:
        loss, acc = train_step(images, labels)
        epoch_loss.append(loss.numpy())
        epoch_acc.append(acc.numpy())
    
    # 计算本epoch的平均训练指标
    avg_train_loss = np.mean(epoch_loss)
    avg_train_acc = np.mean(epoch_acc)
    
    # 验证阶段
    val_losses = []
    val_accs = []
    for images, labels in ted:
        loss, acc = val_step(images, tf.cast(labels, tf.int64))
        val_losses.append(loss.numpy())
        val_accs.append(acc.numpy())
    
    # 计算本epoch的平均验证指标
    avg_val_loss = np.mean(val_losses)
    avg_val_acc = np.mean(val_accs)
    
    # 记录到history字典
    history['loss'].append(avg_train_loss)
    history['accuracy'].append(avg_train_acc)
    history['val_loss'].append(avg_val_loss)
    history['val_accuracy'].append(avg_val_acc)
    
    # 打印进度
    print(f'训练 - 损失: {avg_train_loss:.4f}, 准确率: {avg_train_acc:.4f}')
    print(f'验证 - 损失: {avg_val_loss:.4f}, 准确率: {avg_val_acc:.4f}')
    print('---')

# 现在你可以像使用 model.fit() 返回的 history 一样使用它
plt.figure(figsize=(12, 4))

# 绘制准确率
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

# 绘制损失
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.tight_layout()
plt.show()