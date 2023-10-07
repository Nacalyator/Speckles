import tensorflow as tf
from keras import Input, layers, optimizers, regularizers
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
#print(os.getenv('TF_GPU_ALLOCATOR'))


## Create generators for data
base_dir = './'
#train_dir = os.path.join(base_dir, 'train')
train_dir = os.path.join(base_dir, 'train_v2')
# Load dataframe from CSV
data_df = pd.read_csv(os.path.join(train_dir, 'df.csv'), delimiter=';', dtype={'vals':np.float32})
data_df['vals'] = data_df['vals'] / data_df['vals'].max()

datagen = ImageDataGenerator(horizontal_flip=True,
                             rescale=1./255,
                             validation_split=0.2)

train_gen = datagen.flow_from_dataframe(dataframe=data_df,
                                        directory=train_dir,
                                        x_col='pics',
                                        y_col='vals',
                                        target_size=(250, 500),
                                        color_mode='grayscale',
                                        class_mode='raw',
                                        batch_size=1,
                                        shuffle=True,
                                        subset='training')

val_gen = datagen.flow_from_dataframe(dataframe=data_df,
                                      directory=train_dir,
                                      x_col='pics',
                                      y_col='vals',
                                      target_size=(250, 500),
                                      color_mode='grayscale',
                                      class_mode='raw',
                                      batch_size=1,
                                      shuffle=True,
                                      subset='validation')
#buff = train_gen.next()
#b1 = buff[0][0]
#b2 = buff[1][0]

## Creating NN

input_tensor = Input(shape=(250, 500, 1))
conv_1 = layers.Conv2D(16, (10, 10), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))(input_tensor)
conv_2 = layers.Conv2D(24, (5, 5), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))(conv_1)
conv_3 = layers.Conv2D(24, (3, 3), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))(conv_2)
flatten_1 = layers.Flatten()(conv_3)
fc_1 = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001))(flatten_1)
fc_2 = layers.Dense(32, activation='relu',  kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001))(fc_1)
output_tensor = layers.Dense(1)(fc_2)

model = Model(input_tensor, output_tensor)
model.summary()
model.compile(optimizer=optimizers.SGD(learning_rate=1e-4),
              loss='mse',
              metrics=['acc', 'mse', 'mae'])
cp_dir = './conv_NN_v1_states/'
model_dir = './saved_models/conv_v1'

cp_callback = ModelCheckpoint(filepath='./conv_NN_v1_states/cp-{epoch:04d}.ckpt',
                              monitor='val_mae',
                              verbose=1,
                              save_best_only=True,
                              mode='min',
                              save_weights_only=True)

## Train NN
history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=30,
                    steps_per_epoch=50,
                    callbacks=[cp_callback])


## Load latest weight with the lowest MSE validation values
latest = tf.train.latest_checkpoint(cp_dir)
print(latest)
#model.load_weights(latest)

# =============================================
# Keract visualizations
# =============================================
from keract import get_activations, display_activations
buff = train_gen.next()
b1 = buff[0]
b2 = buff[1]
keract_inputs = b1
keract_targets = b2
activations = get_activations(model, keract_inputs)
display_activations(activations, cmap="gray", save=False)



## Test network
#loss, acc = model.evaluate(val_gen, verbose=2)
#print('The best loss is ' + str(loss) + ' and the accuracy is ' + str(acc))
# v1
b = train_gen.next()
b1 = b[0]
b2 = b[1]
b2_t = model.predict(b1)
print('Expectation: ', str(b2))
print('Result: ', str(b2_t))
#2
b = train_gen.next()
b1 = b[0]
b2 = b[1]
b2_t = model.predict(b1)
print('Expectation: ', str(b2), ' Result: ', str(b2_t))
#3
b = train_gen.next()
b1 = b[0]
b2 = b[1]
b2_t = model.predict(b1)
print('Expectation: ', str(b2), ' Result: ', str(b2_t))

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