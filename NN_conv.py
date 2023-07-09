from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras import regularizers
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Create generators for data
base_dir = './'
train_dir = os.path.join(base_dir, 'train')

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
                                        batch_size=10,
                                        shuffle=True,
                                        subset='training')

val_gen = datagen.flow_from_dataframe(dataframe=data_df,
                                      directory=train_dir,
                                      x_col='pics',
                                      y_col='vals',
                                      target_size=(250, 500),
                                      color_mode='grayscale',
                                      class_mode='raw',
                                      batch_size=10,
                                      shuffle=True,
                                      subset='validation')
#buff = train_gen.next()
#b1 = buff[0][0]
#b2 = buff[1][0]

# Creating NN
from keras import Input, layers
from keras.models import Model

#1
# 14 epochs is optimal
input_tensor = Input(shape=(250, 500, 1))
conv_1 = layers.Conv2D(16, (4, 4), activation='relu')(input_tensor)
d1 = layers.Dropout(0.1)(conv_1)
conv_2 = layers.Conv2D(24, (3, 3), activation='relu')(d1)
d2 = layers.Dropout(0.1)(conv_2)
conv_3 = layers.Conv2D(32, (3, 3), activation='relu')(conv_2)
flatten_1 = layers.Flatten()(conv_3)
fc_1 = layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.L1L2(l1=0.01, l2=0.01))(flatten_1)
fc_2 = layers.Dense(32, activation='relu')(fc_1)
output_tensor = layers.Dense(1, activation = 'linear')(fc_2)


#2
'''
input_tensor = Input(shape=(250, 500, 1))
conv_1 = layers.Conv2D(24, (5, 5), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))(input_tensor)
conv_2 = layers.Conv2D(24, (5, 5), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))(conv_1)
conv_3 = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))(conv_2)
flatten_1 = layers.Flatten()(conv_3)
fc_1 = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001))(flatten_1)
fc_2 = layers.Dense(32, activation='relu',  kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001))(fc_1)
output_tensor = layers.Dense(1)(fc_2)
'''

model = Model(input_tensor, output_tensor)
model.summary()

model.compile(optimizer=optimizers.SGD(learning_rate=1e-4),
              loss='mse',
              metrics=['acc', 'mse', 'mae'])


# Train NN

cp_callback = ModelCheckpoint(filepath='./conv_NN_v1_states/cp-{epoch:04d}.ckpt',
                              monitor='val_mse',
                              verbose=1,
                              save_best_only=True,
                              mode='min',
                              save_weights_only=True,
                              save_freq=50)

history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=50,
                    steps_per_epoch=100,
                    callbacks=[cp_callback])

# Test network
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


# Save model
#model.save('./saved_models/conv_v1')

# Save history.history
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