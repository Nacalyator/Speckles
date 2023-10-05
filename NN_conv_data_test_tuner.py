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
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
#print(os.getenv('TF_GPU_ALLOCATOR'))

## Load data
# Data generators
base_dir = './train_v6'
train_dir = os.path.join(base_dir, 'train')

def load_images_from_folder(folder_path):
    image_list = []

    # List all files in the directory
    for filename in os.listdir(folder_path):
        # Check if the file is an image (by its extension)
        if filename.endswith(".png"):
            # Read the image
            img_path = os.path.join(folder_path, filename)
            img = tf.keras.preprocessing.image.load_img(img_path, color_mode='grayscale', target_size=(125, 125))
            
            # Convert image to array and normalize to [0,1]
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            image_list.append(img_array)

    return np.array(image_list)

def load_targets_from_csv(csv_path):
    # Read the CSV file using pandas
    df = pd.read_csv(csv_path, header=None, delimiter=';')  # Assumes no header
    
    # Get the values from the second column
    targets = df.iloc[:, 1].values

    # Check for NaN or missing values
    if pd.isnull(targets).any():
        print("Warning: NaN or missing values detected. They will be replaced with zeros.")
        targets = np.nan_to_num(targets)

    # Cast to float32
    targets = targets.astype(np.float32)

    return targets


    return targets

targets = load_targets_from_csv(os.path.join(train_dir, 'df copy.csv'))

images = load_images_from_folder(train_dir)

assert len(images) == len(targets), "Mismatch between number of images and targets."


validation_size = 0.2

X_train, X_val, y_train, y_val = train_test_split(images, targets, test_size=validation_size, random_state=42)

print("Training images shape:", X_train.shape)
print("Validation images shape:", X_val.shape)
print("Training targets shape:", y_train.shape)
print("Validation targets shape:", y_val.shape)


## Creating NN

class MyHyperModel(HyperModel):

    def build(self, hp):
        input_tensor = Input(shape=(125, 125, 1))

        conv_1 = layers.Conv2D(
            filters=hp.Int('filters_1', 116, 128, step=4),
            kernel_size=hp.Choice('kernel_size_1', values=[3, 4, 5, 6]),
            strides=hp.Choice('strides_1', values=[2, 3, 4]),
            activation='relu'
        )(input_tensor)

        conv_2 = layers.Conv2D(
            filters=hp.Int('filters_2', 56, 72, step=4),
            kernel_size=hp.Choice('kernel_size_2', values=[4, 5, 6, 7]),
            strides=hp.Choice('strides_2', values=[1]),
            activation='relu'
        )(conv_1)

        conv_3 = layers.Conv2D(
            filters=hp.Int('filters_3', 16, 24, step=4),
            kernel_size=hp.Choice('kernel_size_3', values=[3, 4, 5]),
            activation='relu'
        )(conv_2)

        conv_4 = layers.Conv2D(
            filters=hp.Int('filters_4', 120, 132, step=4),
            kernel_size=hp.Choice('kernel_size_4', values=[4, 5, 6, 7]),
            activation='relu'
        )(conv_3)

        flatten_1 = layers.Flatten()(conv_4)

        fc_1 = layers.Dense(
            units=hp.Int('dense_units_1', 1536, 2048, step=64),
            activation='relu'
        )(flatten_1)

        fc_2 = layers.Dense(
            units=hp.Int('dense_units_2', 256, 768, step=64),
            activation='relu'
        )(fc_1)

        output_tensor = layers.Dense(1)(fc_2)
        
        model = Model(inputs=input_tensor, outputs=output_tensor)
        model.compile(optimizer='SGD', loss='MAE')

        return model




tuner = RandomSearch(
    MyHyperModel(),
    objective='val_loss',
    max_trials=60,  # number of different hyperparameter configurations to try
    executions_per_trial=3,  # number of models to train per trial (to reduce variance)
    directory='output_dir',
    project_name='my_project'
)


tuner.search(X_train, y_train,
             epochs=25,
             validation_data=(X_val, y_val))



best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Display the best hyperparameters
for hp_name in best_hps.values:
    print(f"{hp_name}: {best_hps.get(hp_name)}")

'''
filters_1: 48
kernel_size_1: 5
strides_1: 1
filters_2: 56
kernel_size_2: 5
strides_2: 2
filters_3: 36
kernel_size_3: 4
filters_4: 76
kernel_size_4: 3
dense_units_1: 1216
dense_units_2: 512
0.004053967539221048
'''

'''
Best Value So Far |Hyperparameter
120               |filters_1
4                 |kernel_size_1
3                 |strides_1
64                |filters_2
6                 |kernel_size_2
1                 |strides_2
16                |filters_3
4                 |kernel_size_3
128               |filters_4
6                 |kernel_size_4
1984              |dense_units_1
640               |dense_units_2
0.0036606204230338335
'''