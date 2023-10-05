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
import csv
import matplotlib.pyplot as plt

# Can it solve some erros with the GPU use? 
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    
## Creating NN

csv_test = './rb_1_test_55.csv'
csv_tests = ['./br_3_test_51.csv',
            './br_3_test_52.csv',
            './br_3_test_53.csv',
            './br_3_test_54.csv',
            './br_3_test_55.csv']

cp_dir = './Data/trained_NN_BLUE/v3/siamese_v1'
## Data generators
base_dir = './train_v8_55'
base_dirs = ['./train_v8_51r',
             './train_v8_52r',
             './train_v8_53r',
             './train_v8_54r',
             './train_v8_55r']


for i in range(1, 5):
    train_dir = os.path.join(base_dirs[i], 'train')
    # Load dataframe from CSV
    data_df_train = pd.read_csv(os.path.join(train_dir, 'df.csv'), delimiter=';', dtype={'vals':np.float32})
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
                                            shuffle=False,
                                            seed=7)
    train_gen_2 = datagen.flow_from_dataframe(dataframe=data_df_train,
                                            directory=os.path.join(train_dir, '2'),
                                            x_col='pics',
                                            y_col='vals',
                                            target_size=(125, 125),
                                            color_mode='grayscale',
                                            class_mode='raw',
                                            batch_size=1,
                                            shuffle=False,
                                            seed=7)
    # Custom data generator (2 images are input, 1 float is output)
    def generate_data_generator_for_two_images(gen_1, gen_2):
        while True:
            X1 = gen_1.next()
            X2 = gen_2.next()
            yield [X1[0], X2[0]], X1[1]
    train_gen = generate_data_generator_for_two_images(train_gen_1, train_gen_2)

    ## Redefine model
    input = Input(shape=(125, 125, 1))
    conv_1 = layers.Conv2D(32, (8, 8), activation='relu') (input)
    conv_2 = layers.Conv2D(48, (7, 7), activation='relu') (conv_1)
    conv_3 = layers.Conv2D(64, (3, 3), activation='relu') (conv_2)
    conv_4 = layers.Conv2D(164, (2, 2), activation='relu') (conv_3)
    flatt_1 = layers.Flatten()(conv_4)
    dense_1 = layers.Dense(448, activation='relu')(flatt_1)
    dense_2 = layers.Dense(192, activation='relu')(dense_1)
    embedding_network = Model(input, dense_2)

    input_1 = Input(shape=(125, 125, 1))
    input_2 = Input(shape=(125, 125, 1))
    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)
    merged = layers.Concatenate(axis=-1)([tower_1, tower_2])
    dense__1 = layers.Dense(8192, activation='relu')(merged)
    dense__2 = layers.Dense(2048, activation='relu')(dense__1)
    dense__3 = layers.Dense(512, activation='relu')(dense__2)
    output_layer = layers.Dense(1, activation='sigmoid')(dense__3)
    model = Model(inputs=[input_1, input_2], outputs=output_layer)

    model.compile(optimizer=optimizers.SGD(learning_rate=1e-2),
                loss='mse',
                metrics=['mse', 'mae', 'acc'])

    ## Load latest weight with the lowest MSE validation values

    latest = tf.train.latest_checkpoint(cp_dir)
    print(latest)
    model.load_weights(latest)

    output = []
    measured = []
    trained = []

    for d in range(11000):
        h = next(train_gen)
        inp = h[0]
        outp = h[1][0]
        res = model.predict(inp, verbose=0)
        res = res[0][0]
        output.append([outp, res])
        measured.append(outp)
        trained.append(res)

    #pd.DataFrame.from_dict(output, orient='index').transpose().to_csv('test_siamese_v1_1.csv')



    csv_file_train = open(csv_tests[i], 'w', newline='')
    csv_header = ['pics', 'vals']
    df_writer_train = csv.writer(csv_file_train, delimiter=';')
    df_writer_train.writerow(csv_header)

    for i in range(len(output)):
        df_writer_train.writerow([output[i][0], output[i][1]])

    csv_file_train.close()

    ## PLOTS
'''
    plt.figure('Mesuared and predicted data')
    plt.plot([i for i in range(1, 11000)], measured[1:11000], 'b', label='Measured data')
    plt.plot([i for i in range(1, 11000)], trained[1:11000], 'bo', label='Predicted data')
    plt.title('Mesuared and predicted data')
    plt.xlabel('Frame')
    plt.ylabel('Normalized deformation')
    plt.legend()
    plt.show()
    '''