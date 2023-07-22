import tensorflow as tf
from keras import Input, layers, optimizers, regularizers
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
#print(os.getenv('TF_GPU_ALLOCATOR'))

def generate_matrix(line_num):
    matrix = np.zeros((125, 125), dtype=np.float32)
    row_to_fill = (line_num // 4) % 125
    matrix[row_to_fill] = np.linspace(0.1, 12.5, 125)
    return matrix

def generate_coefficient(line_num):
    row_to_fill = (line_num // 4) % 125
    if row_to_fill == 0: return 1/1
    return 1/row_to_fill

x = np.array([generate_matrix(x) for x in range(5000)])
y = np.array([generate_coefficient(x) for x in range(5000)])

# Normalize x
x = x / 12.5

## Creating NN


##input_tensor = Input(shape=(125, 125, 1))
##conv_1 = layers.Conv2D(8, (5, 5), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))(input_tensor)
##conv_2 = layers.Conv2D(16, (5, 5), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))(conv_1)
##conv_3 = layers.Conv2D(24, (2, 2), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))(conv_2)
##flatten_1 = layers.Flatten()(conv_3)
##fc_1 = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001))(flatten_1)
##fc_2 = layers.Dense(32, activation='relu',  kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001))(fc_1)
##output_tensor = layers.Dense(1)(fc_2)
input_tensor = Input(shape=(125, 125, 1))
conv_1 = layers.Conv2D(8, (5, 5), activation='relu')(input_tensor)
conv_2 = layers.Conv2D(16, (5, 5), activation='relu')(conv_1)
conv_3 = layers.Conv2D(24, (2, 2), activation='relu')(conv_2)
flatten_1 = layers.Flatten()(conv_3)
fc_1 = layers.Dense(128, activation='relu')(flatten_1)
fc_2 = layers.Dense(32, activation='relu')(fc_1)
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

history = model.fit(x=x, y=y,
                    epochs=5,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[cp_callback])


## Redefine model

## '''
## input_tensor = Input(shape=(125, 125, 1))
## conv_1 = layers.Conv2D(8, (5, 5), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))(input_tensor)
## conv_2 = layers.Conv2D(16, (5, 5), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))(conv_1)
## conv_3 = layers.Conv2D(24, (2, 2), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))(conv_2)
## flatten_1 = layers.Flatten()(conv_3)
## fc_1 = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001))(flatten_1)
## fc_2 = layers.Dense(32, activation='relu',  kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001))(fc_1)
## output_tensor = layers.Dense(1)(fc_2)
## Model(input_tensor, output_tensor)
## '''
## Load latest weight with the lowest MSE validation values
latest = tf.train.latest_checkpoint(cp_dir)
print(latest)
model.load_weights(latest)





## Test network
#loss, acc = model.evaluate(val_gen, verbose=2)
#print('The best loss is ' + str(loss) + ' and the accuracy is ' + str(acc))
# v1
x1 = x[1]
y1 = y[1]
x1 = np.expand_dims(x1, axis=0)
y2 = model.predict(x1)
print('Expectation: ', str(y1))
print('Result: ', str(y2[0][0]))
#2
x1 = x[100]
y1 = y[100]
x1 = np.expand_dims(x1, axis=0)
y2 = model.predict(x1)
print('Expectation: ', str(y1))
print('Result: ', str(y2[0][0]))
#3
x1 = x[300]
y1 = y[300]
x1 = np.expand_dims(x1, axis=0)
y2 = model.predict(x1)
print('Expectation: ', str(y1))
print('Result: ', str(y2[0][0]))
#4
x1 = x[400]
y1 = y[400]
x1 = np.expand_dims(x1, axis=0)
y2 = model.predict(x1)
print('Expectation: ', str(y1))
print('Result: ', str(y2[0][0]))
#5
x1 = x[500]
y1 = y[500]
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
x1 = np.expand_dims(x[1], axis=0)
y1 = y[1]
keract_inputs = x1
keract_targets = y1
activations = get_activations(model, keract_inputs)
display_activations(activations, cmap="gray", save=False)
