import tensorflow as tf
from keras import Input, layers, optimizers, regularizers
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import csv
import cv2
import os
import matplotlib.pyplot as plt

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
#print(os.getenv('TF_GPU_ALLOCATOR'))

## Load data
# Data generators
base_dir = './train_v6'
train_dir = os.path.join(base_dir, 'train')
# Load dataframe from CSV
data_df_train = pd.read_csv(os.path.join(train_dir, 'df.csv'), delimiter=';', dtype={'vals':np.float32})
m = data_df_train['vals'].max()
data_df_train['vals'] = data_df_train['vals'] / m

datagen = ImageDataGenerator(rescale=1./255,
                             validation_split=0.2)

train_gen = datagen.flow_from_dataframe(dataframe=data_df_train,
                                        directory=train_dir,
                                        x_col='pics',
                                        y_col='vals',
                                        target_size=(125, 125),
                                        color_mode='grayscale',
                                        class_mode='raw',
                                        batch_size=32,
                                        shuffle=True,
                                        subset='training')

val_gen = datagen.flow_from_dataframe(dataframe=data_df_train,
                                        directory=train_dir,
                                        x_col='pics',
                                        y_col='vals',
                                        target_size=(125, 125),
                                        color_mode='grayscale',
                                        class_mode='raw',
                                        batch_size=32,
                                        shuffle=True,
                                        subset='validation')




## Creating NN

input_tensor = Input(shape=(125, 125, 1))
conv_1 = layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu')(input_tensor)
conv_2 = layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu')(conv_1)
conv_3 = layers.Conv2D(64, (4, 4), activation='relu')(conv_2)
conv_4 = layers.Conv2D(64, (3, 3), activation='relu')(conv_3)
flatten_1 = layers.Flatten()(conv_4)
fc_1 = layers.Dense(1024, activation='relu')(flatten_1)
fc_2 = layers.Dense(128, activation='relu')(fc_1)
output_tensor = layers.Dense(1)(fc_2)

model = Model(input_tensor, output_tensor)
model.summary()
model.compile(optimizer=optimizers.SGD(learning_rate=1e-4),
              loss='mse',
              metrics=['acc', 'mse', 'mae'])
cp_dir = './NN_states/conv_NN/'
model_dir = './saved_models/conv_v1'

cp_callback = ModelCheckpoint(filepath='./NN_states/conv_NN/cp-{epoch:04d}.ckpt',
                              monitor='val_mae',
                              verbose=1,
                              save_best_only=True,
                              mode='min',
                              save_weights_only=True)

## Train NN

history = model.fit(train_gen,
                    epochs=400,
                    batch_size=32,
                    validation_data=val_gen,
                    callbacks=[cp_callback])


## Redefine model
input_tensor = Input(shape=(125, 125, 1))
conv_1 = layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu')(input_tensor)
conv_2 = layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu')(conv_1)
conv_3 = layers.Conv2D(64, (4, 4), activation='relu')(conv_2)
conv_4 = layers.Conv2D(64, (3, 3), activation='relu')(conv_3)
flatten_1 = layers.Flatten()(conv_4)
fc_1 = layers.Dense(1024, activation='relu')(flatten_1)
fc_2 = layers.Dense(128, activation='relu')(fc_1)
output_tensor = layers.Dense(1)(fc_2)

model = Model(input_tensor, output_tensor)

## Load latest weight with the lowest MSE validation values
latest = tf.train.latest_checkpoint(cp_dir)
print(latest)
model.load_weights(latest)





## Test network
#loss, acc = model.evaluate(val_gen, verbose=2)
#print('The best loss is ' + str(loss) + ' and the accuracy is ' + str(acc))
for i in range(0, 10):
    buff = next(train_gen)
    x1 = buff[0][0]
    y1 = buff[1][0]
    x1 = np.expand_dims(x1, axis=0)
    y2 = model.predict(x1)
    print('Expectation: ', str(y1))
    print('Result: ', str(y2[0][0]))


## Save model
model.save(model_dir)

## Save history.history
pd.DataFrame.from_dict(history.history, orient='index').transpose().to_csv('dict_CONV.csv')

## Plots
mse = history.history['mse']
val_mse = history.history['val_mse']
mae = history.history['mae']
val_mae = history.history['val_mae']
epochs = range(1, len(mse) + 1)

plt.figure('Training and validation MSE')
plt.plot(epochs, mse, 'bo', label='Training MSE')
plt.plot(epochs, val_mse, 'b', label='Validation MSE')
plt.title('Training and validation MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()

plt.figure('Training and validation MAE')
plt.plot(epochs, mae, 'bo', label='Training MAE')
plt.plot(epochs, val_mae, 'b', label='Validation MAE')
plt.title('Training and validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

# =============================================
# Keract visualizations
# =============================================

from keract import get_activations, display_activations
buff = next(train_gen)
x1 = buff[0][0]
x1 = np.expand_dims(x1, axis=0)
y1 = buff[1][0]
keract_inputs = x1
keract_targets = y1
activations = get_activations(model, keract_inputs)
display_activations(activations, cmap="gray", save=False)

'''
conv_1 = layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu')(input_tensor)
conv_2 = layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu')(conv_1)
conv_3 = layers.Conv2D(64, (4, 4), activation='relu')(conv_2)
conv_4 = layers.Conv2D(64, (3, 3), activation='relu')(conv_3)
271 (mae 0.01876)
'''
