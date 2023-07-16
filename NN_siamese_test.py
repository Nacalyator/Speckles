import tensorflow as tf
import cv2
from keras import Input, layers, optimizers, regularizers, utils
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Can it solve some erros with the GPU use?
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

## Data generators
base_dir = './train_v3'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
# Load dataframe from CSV
data_df_train = pd.read_csv(os.path.join(train_dir, 'df.csv'), delimiter=';', dtype={'vals':np.float32})
data_df_val = pd.read_csv(os.path.join(val_dir, 'df.csv'), delimiter=';', dtype={'vals':np.float32})
m1 = data_df_train['vals'].max()
m2 = data_df_val['vals'].max()
m = 0
if m1 >= m2:
    m = m1
else:
    m = m2
data_df_train['vals'] = data_df_train['vals'] / m
data_df_val['vals'] = data_df_val['vals'] / m

datagen = ImageDataGenerator(rescale=1./255)

train_gen_1 = datagen.flow_from_dataframe(dataframe=data_df_train,
                                          directory=os.path.join(train_dir, '1'),
                                          x_col='pics',
                                          y_col='vals',
                                          target_size=(250, 250),
                                          color_mode='grayscale',
                                          class_mode='raw',
                                          batch_size=1,
                                          shuffle=True,
                                          seed=7)
train_gen_2 = datagen.flow_from_dataframe(dataframe=data_df_train,
                                          directory=os.path.join(train_dir, '2'),
                                          x_col='pics',
                                          y_col='vals',
                                          target_size=(250, 250),
                                          color_mode='grayscale',
                                          class_mode='raw',
                                          batch_size=1,
                                          shuffle=True,
                                          seed=7)
val_gen_1 = datagen.flow_from_dataframe(dataframe=data_df_val,
                                        directory=os.path.join(val_dir, '1'),
                                        x_col='pics',
                                        y_col='vals',
                                        target_size=(250, 250),
                                        color_mode='grayscale',
                                        class_mode='raw',
                                        batch_size=1,
                                        shuffle=True,
                                        seed=7)
val_gen_2 = datagen.flow_from_dataframe(dataframe=data_df_val,
                                        directory=os.path.join(val_dir, '1'),
                                        x_col='pics',
                                        y_col='vals',
                                        target_size=(250, 250),
                                        color_mode='grayscale',
                                        class_mode='raw',
                                        batch_size=1,
                                        shuffle=True,
                                        seed=7)

# Custom data generator (2 images are input, 1 float is output)
def generate_data_generator_for_two_images(gen_1, gen_2):
    while True:
        X1 = gen_1.next()
        X2 = gen_2.next()
        yield [X1[0], X2[0]], X1[1]

train_gen = generate_data_generator_for_two_images(train_gen_1, train_gen_2)
val_gen = generate_data_generator_for_two_images(val_gen_1, val_gen_2)
'''
buff = next(train_gen)
b1 = buff[0]
b2 = buff[1]
'''

## Creating NN
def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors
	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)
	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))

#input_1
input_1 = Input(shape=(250, 250, 1))
conv_1_1 = layers.Conv2D(64, (3, 3), activation='relu') (input_1)
maxpool_1_1 = layers.MaxPooling2D(pool_size=(2, 2))(conv_1_1)
conv_1_2 = layers.Conv2D(64, (3, 3), activation='relu') (maxpool_1_1)
maxpool_1_2 = layers.MaxPooling2D(pool_size=(2, 2))(conv_1_2)
conv_1_3 = layers.Conv2D(64, (3, 3), activation='relu') (maxpool_1_2)
maxpool_1_3 = layers.MaxPooling2D(pool_size=(2, 2))(conv_1_3)
GAP_1 = layers.GlobalAveragePooling2D()(maxpool_1_3)
future_extractor = Model(input_1, GAP_1)

pic_1 = Input(shape=(250, 250, 1))
pic_2 = Input(shape=(250, 250, 1))
feats_1 = future_extractor(pic_1)
feats_2 = future_extractor(pic_2)
distance = layers.Lambda(euclidean_distance)([feats_1, feats_2])
outputs = layers.Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[pic_1, pic_2], outputs=outputs)

model.summary()
model.compile(optimizer=optimizers.SGD(learning_rate=1e-4),
              loss='mse',
              metrics=['mse', 'mae', 'acc'])
cp_dir = './siamese_NN_v1_states/'
model_dir = './saved_models/siamese_v1'

cp_callback = ModelCheckpoint(filepath='./siamese_NN_v1_states/cp-{epoch:04d}.ckpt',
                              monitor='val_mae',
                              verbose=1,
                              save_best_only=True,
                              mode='min',
                              save_weights_only=True)

## Load latest weight with the lowest MSE validation values
latest = tf.train.latest_checkpoint(cp_dir)
#print(latest)
model.load_weights(latest)


## Test network


for j in range(20):
     buff = next(val_gen)
     inp = buff[0]
     outp = buff[1]
     res = model.predict(inp)
     print('Expected: ', str(outp), ', Result: ', str(res), ', Diff: ', str(abs(res-outp)))
