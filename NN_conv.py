from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import regularizers
import pandas as pd
import os
import matplotlib.pyplot as plt

# Create generators for data
base_dir = './'
data_dir = os.path.join(base_dir, 'data')

datagen = ImageDataGenerator(horizontal_flip=True,
                             rescale=1./255,
                             validation_split=0.2)

data_df = pd.read_csv(os.path.join(data_dir, 'df.csv'))
data_df['labels'] = data_df['labels'] / data_df['labels'].abs().max()

train_gen = datagen.flow_from_dataframe(dataframe=data_df,
                                        directory=data_dir,
                                        x_col='ids',
                                        y_col='labels',
                                        target_size=(250, 500),
                                        color_mode='grayscale',
                                        class_mode='raw',
                                        batch_size=10,
                                        shuffle=True,
                                        subset='training')

val_gen = datagen.flow_from_dataframe(dataframe=data_df,
                                      directory=data_dir,
                                      x_col='ids',
                                      y_col='labels',
                                      target_size=(250, 500),
                                      color_mode='grayscale',
                                      class_mode='raw',
                                      batch_size=10,
                                      shuffle=True,
                                      subset='validation')

# Creating NN
from keras import Input, layers
from keras.models import Model


input_tensor = Input(shape=(250, 500, 1))
conv_1 = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
d1 = layers.Dropout(0.1)(conv_1)
conv_2 = layers.Conv2D(32, (3, 3), activation='relu')(d1)
d2 = layers.Dropout(0.1)(conv_2)
conv_3 = layers.Conv2D(48, (3, 3), activation='relu')(conv_2)
flatten_1 = layers.Flatten()(conv_3)
fc_1 = layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.L1L2(l1=0.01, l2=0.01))(flatten_1)
fc_2 = layers.Dense(32, activation='relu')(fc_1)
output_tensor = layers.Dense(1, activation = 'sigmoid')(fc_2)

model = Model(input_tensor, output_tensor)
model.summary()

model.compile(optimizer=optimizers.SGD(learning_rate=1e-5),
              loss='mse',
              metrics=['mse'])


# Train NN

history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=100,
                    batch_size=10,
                    steps_per_epoch=100)

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