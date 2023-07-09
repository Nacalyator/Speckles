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

input_tensor = Input(shape=(250, 500, 1))
flat = layers.Flatten()(input_tensor)
fc_1 = layers.Dense(2048, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))(flat)
fc_2 = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))(fc_1)
fc_3 = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))(fc_2)
fc_4 = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))(fc_3)
fc_5 = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))(fc_4)
fc_6 = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))(fc_5)
output_tensor = layers.Dense(1)(fc_6)


model = Model(input_tensor, output_tensor)
model.summary()

model.compile(optimizer=optimizers.SGD(learning_rate=1e-3),
              loss='mse',
              metrics=['acc', 'mse', 'mae'])

# Train NN

history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=100,
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
#model.save('./saved_models/conv_v1')

# Save history.history
pd.DataFrame.from_dict(history.history, orient='index').transpose().to_csv('dict_CONV.csv')

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