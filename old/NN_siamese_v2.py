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
    
## Creating NN
def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors
	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def attention_feature_map(feature_map):
    # Query, key, and value
    query = layers.Conv2D(64, (1, 1), activation='relu')(feature_map)
    key = layers.Conv2D(64, (1, 1), activation='relu')(feature_map)
    value = layers.Conv2D(128, (1, 1), activation='relu')(feature_map)  # Change depth to 128 to match conv_4
    
    # Pooling to reduce dimensions
    pooled_query = layers.AveragePooling2D(pool_size=(2, 2))(query)
    pooled_key = layers.AveragePooling2D(pool_size=(2, 2))(key)
    pooled_value = layers.AveragePooling2D(pool_size=(2, 2))(value)
    
    # Reshape for matrix multiplication
    pooled_query_reshape = layers.Reshape((-1, 64))(pooled_query)
    pooled_key_reshape = layers.Reshape((-1, 64))(pooled_key)
    pooled_value_reshape = layers.Reshape((-1, 128))(pooled_value)  # Change depth to 128 to match conv_4
    
    # Attention scores
    score = layers.Dot(axes=[2, 2])([pooled_query_reshape, pooled_key_reshape])
    score = layers.Activation('softmax')(score)
    
    # Weighted feature map
    attended = layers.Dot(axes=[2, 1])([score, pooled_value_reshape])

    # Compute reshape dimensions
    attended_shape = K.int_shape(attended)
    reshape_dims = int(np.sqrt(attended_shape[1]))

    # Reshape back to spatial dimensions
    attended_reshape = layers.Reshape((reshape_dims, reshape_dims, 128))(attended)
    
    
    
    # Upsample
    upsampled_attended = layers.UpSampling2D(size=(2, 2))(attended_reshape)

    # Crop the original feature map
    cropped_original = layers.Cropping2D(cropping=((0, 1), (0, 1)))(feature_map)

    # Use a 1x1 convolution to change the number of filters in `cropped_original` to 128
    resized_original = layers.Conv2D(128, (1, 1), activation='relu')(cropped_original)

    # Crop the upsampled feature map
    upsampled_attended_cropped = layers.Cropping2D(cropping=((0, 1), (0, 1)))(upsampled_attended)

    # Combine
    combined = layers.Add()([upsampled_attended_cropped, resized_original])

    return combined




#input_stack
input = Input(shape=(125, 125, 1))
conv_1 = layers.Conv2D(32, (8, 8), activation='relu') (input)
conv_2 = layers.Conv2D(48, (7, 7), activation='relu') (conv_1)
conv_3 = layers.Conv2D(64, (3, 3), activation='relu') (conv_2)
conv_4 = layers.Conv2D(164, (2, 2), activation='relu') (conv_3)
attention_map = attention_feature_map(conv_4)
flatt_1 = layers.Flatten()(attention_map)
dense_1 = layers.Dense(448, activation='relu')(flatt_1)
dense_2 = layers.Dense(192, activation='relu')(dense_1)
embedding_network = Model(input, dense_2)

'''
Best MAE is 0.02392
input = Input(shape=(125, 125, 1))
conv_1 = layers.Conv2D(128, (4, 4), strides=(3, 3), activation='relu') (input)
conv_2 = layers.Conv2D(64, (6, 6), activation='relu') (conv_1)
conv_3 = layers.Conv2D(16, (4, 4), activation='relu') (conv_2)
conv_4 = layers.Conv2D(128, (6, 6), activation='relu') (conv_3)
attention_map = attention_feature_map(conv_4)
flatt_1 = layers.Flatten()(attention_map)
dense_1 = layers.Dense(2048, activation='relu')(flatt_1)
dense_2 = layers.Dense(768, activation='relu')(dense_1)
embedding_network = Model(input, dense_2)
'''

'''
Best MAE is 0.02393
input = Input(shape=(125, 125, 1))
conv_1 = layers.Conv2D(48, (5, 5), activation='relu') (input)
conv_2 = layers.Conv2D(64, (5, 5), strides=(2, 2), activation='relu') (conv_1)
conv_3 = layers.Conv2D(48, (4, 4), activation='relu') (conv_2)
conv_4 = layers.Conv2D(96, (3, 3), activation='relu') (conv_3)
attention_map = attention_feature_map(conv_4)
flatt_1 = layers.Flatten()(attention_map)
dense_1 = layers.Dense(1536, activation='relu')(flatt_1)
dense_2 = layers.Dense(512, activation='relu')(dense_1)
embedding_network = Model(input, dense_2)
'''



input_1 = Input(shape=(125, 125, 1))
input_2 = Input(shape=(125, 125, 1))

tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

merged = layers.Concatenate(axis=-1)([tower_1, tower_2])
dense__1 = layers.Dense(8192, activation='relu')(merged)
dense__2 = layers.Dense(2048, activation='relu')(dense__1)
dense__3 = layers.Dense(512, activation='relu')(dense__2)
output_layer = layers.Dense(1, activation='sigmoid')(dense__3)


#merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
#output_layer = layers.Dense(1, activation='sigmoid')(norm_layer)

model = Model(inputs=[input_1, input_2], outputs=output_layer)
model.summary()

model.compile(optimizer=optimizers.SGD(learning_rate=1e-2),
              loss='mse',
              metrics=['mse', 'mae', 'acc'])

cp_dir = './NN_states/siamese_v2/'
model_dir = './saved_models/siamese_v2'

cp_callback = ModelCheckpoint(filepath='./NN_states/siamese_v2/cp-{epoch:04d}.ckpt',
                              monitor='val_mae',
                              verbose=1,
                              save_best_only=True,
                              mode='min',
                              save_weights_only=True)





## Data generators
base_dir = './train_v7'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
# Load dataframe from CSV
data_df_train = pd.read_csv(os.path.join(train_dir, 'df.csv'), delimiter=';', dtype={'vals':np.float32})
data_df_val = pd.read_csv(os.path.join(val_dir, 'df.csv'), delimiter=';', dtype={'vals':np.float32})


datagen = ImageDataGenerator(rescale=1./255)
#datagen = ImageDataGenerator()

train_gen_1 = datagen.flow_from_dataframe(dataframe=data_df_train,
                                          directory=os.path.join(train_dir, '1'),
                                          x_col='pics',
                                          y_col='vals',
                                          target_size=(125, 125),
                                          color_mode='grayscale',
                                          class_mode='raw',
                                          batch_size=1,
                                          shuffle=True,
                                          seed=7)
train_gen_2 = datagen.flow_from_dataframe(dataframe=data_df_train,
                                          directory=os.path.join(train_dir, '2'),
                                          x_col='pics',
                                          y_col='vals',
                                          target_size=(125, 125),
                                          color_mode='grayscale',
                                          class_mode='raw',
                                          batch_size=1,
                                          shuffle=True,
                                          seed=7)
val_gen_1 = datagen.flow_from_dataframe(dataframe=data_df_val,
                                        directory=os.path.join(val_dir, '1'),
                                        x_col='pics',
                                        y_col='vals',
                                        target_size=(125, 125),
                                        color_mode='grayscale',
                                        class_mode='raw',
                                        batch_size=1,
                                        shuffle=True,
                                        seed=7)
val_gen_2 = datagen.flow_from_dataframe(dataframe=data_df_val,
                                        directory=os.path.join(val_dir, '2'),
                                        x_col='pics',
                                        y_col='vals',
                                        target_size=(125, 125),
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
for i in range(20):
    buff = next(train_gen)
    b1 = buff[0]
    b2 = buff[1]
    plt.imshow(b1[0][0])
    plt.show()
    plt.imshow(b1[1][0])
    plt.show()

    test = 1
'''









## Train NN
history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=1500,
                    steps_per_epoch=500,
                    validation_steps=200,
                    callbacks=[cp_callback],
                    verbose=1)


## Load latest weight with the lowest MSE validation values
latest = tf.train.latest_checkpoint(cp_dir)
print(latest)
model.load_weights(latest)

# =============================================
# Keract visualizations
# =============================================
'''
from keract import get_activations, display_activations
buff = train_gen_1.next()
b1 = buff[0]
b2 = buff[1]
keract_inputs = b1
keract_targets = b2
activations = get_activations(model, keract_inputs)
display_activations(activations, cmap="gray", save=False)
'''


## Test network
#loss, acc = model.evaluate(val_gen, verbose=2)
#print('The best loss is ' + str(loss) + ' and the accuracy is ' + str(acc))
# v1

for j in range(20):
     buff = next(val_gen)
     inp = buff[0]
     outp = buff[1]
     res = model.predict(inp)
     print('Expected: ', str(outp), ', Result: ', str(res), ', Diff: ', str(abs(res-outp)))

## Save model
model.save(model_dir)

## Save history.history
pd.DataFrame.from_dict(history.history, orient='index').transpose().to_csv('dict_siamese_v2.csv')

## Plots
mse = history.history['mse']
val_mse = history.history['val_mse']
mae = history.history['mae']
val_mae = history.history['val_mae']
epochs = range(1, len(mse) + 1)

def plt_metric(history, metric, title, has_valid=True):
    plt.plot(history[metric], 'bo')
    if has_valid:
        plt.plot(history["val_" + metric], 'b')
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()

plt_metric(history.history, 'mse', 'Training and validation MSE', True)
plt_metric(history.history, 'mae', 'Training and validation MAE', True)
