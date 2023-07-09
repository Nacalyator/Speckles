from keras.preprocessing.image import ImageDataGenerator
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

history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=14,
                    steps_per_epoch=100)

# Test network
#1
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
model.save('./saved_models/conv_v1')

# Save history.history
#pd.DataFrame.from_dict(history.history, orient='index').transpose().to_csv('dict_CONV.csv')

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