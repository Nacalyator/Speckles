from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import regularizers
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Create generators for data
base_dir = './'
data_dir = os.path.join(base_dir, 'data')

datagen = ImageDataGenerator(horizontal_flip=True,
                             rescale=1./255,
                             validation_split=0.2)

data_df = pd.read_csv(os.path.join(data_dir, 'df.csv'), dtype={'labels':np.float32})
data_df['labels'] = data_df['labels'] / data_df['labels'].abs().max()

train_gen = datagen.flow_from_dataframe(dataframe=data_df,
                                        directory=data_dir,
                                        x_col='ids',
                                        y_col='labels',
                                        target_size=(250, 500),
                                        color_mode='grayscale',
                                        class_mode='other',
                                        batch_size=10,
                                        shuffle=False,
                                        subset='training')

val_gen = datagen.flow_from_dataframe(dataframe=data_df,
                                      directory=data_dir,
                                      x_col='ids',
                                      y_col='labels',
                                      target_size=(250, 500),
                                      color_mode='grayscale',
                                      class_mode='other',
                                      batch_size=10,
                                      shuffle=False,
                                      subset='validation')

# Creating NN
from keras import Input, layers
from keras.models import Model


input_tensor = Input(shape=(250, 500, 1))
flat = layers.Flatten()(input_tensor)
fc_1 = layers.Dense(2048, activation='relu')(flat)
fc_2 = layers.Dense(1024, activation='relu')(fc_1)
fc_3 = layers.Dense(128, activation='relu')(fc_2)
fc_4 = layers.Dense(32, activation='relu')(fc_3)
output_tensor = layers.Dense(1, activation = 'relu')(fc_4)

model = Model(input_tensor, output_tensor)
model.summary()

model.compile(optimizer=optimizers.SGD(learning_rate=1e-3),
              loss='mse',
              metrics=['acc', 'mse'])
              #metrics=['mse', 'mae'])

# Train NN

history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=30,
                    batch_size=10,
                    steps_per_epoch=10)

# Save model
#model.save('./saved_models/conv_v1')

## Plots
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(loss) + 1)

plt.figure('Training and validation loss')
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure('Training and validation acc')
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()