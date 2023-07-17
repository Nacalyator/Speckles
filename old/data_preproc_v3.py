import csv
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Paths to data files
video_path = './RED.mp4'
csv_path = './RED.csv'

# Read data from csv
load_times = []
load_deforms = []
csv_file = open(csv_path, 'r', newline='', encoding='latin-1')
csv_reader = csv.reader(csv_file, delimiter=';')
csv_reader.__next__()
csv_reader.__next__()
csv_reader.__next__()
csv_reader.__next__()
for row in csv_reader:
	load_times.append(float(row[0].replace(',', '.')))
	load_deforms.append(float(row[3].replace(',', '.')))
print('CSV file has been read')

# Read video frames and imfo
timestamps = []
frames = []
cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)
video_total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
prev = []
while (cap.isOpened()):
	ret, frame = cap.read()
	if ret == True:
		timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame = cv2.resize(frame, [125, 125])
		#filt = cv2.fastNlMeansDenoising(frame,None, 20, 7, 20) 
		#filt = cv2.bilateralFilter(frame, 5, 500, 500)
		filt = cv2.medianBlur(frame, 5)
		#pic = np.concatenate((frame, filt), axis=1)
		#cv2.imshow('Result', pic)
		#cv2.waitKey(0)
		frames.append(filt)
	else:
		break
print('Frames have been read')

# Synchronizing timestamps with the deformation timings
data_deform = np.interp(timestamps, load_times, load_deforms)

# Creating data for NN training
base_dir = './'
data_dir = os.path.join(base_dir, 'train_v3')
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

train_dir_1 = os.path.join(train_dir, '1')
train_dir_2 = os.path.join(train_dir, '2')

val_dir_1 = os.path.join(val_dir, '1')
val_dir_2 = os.path.join(val_dir, '2')


df_csv_train = os.path.join(train_dir, 'df.csv')
df_csv_val = os.path.join(val_dir, 'df.csv')

if not os.path.isdir(data_dir):
	os.mkdir(data_dir)

if not os.path.isdir(train_dir):
	os.mkdir(train_dir)
if not os.path.isdir(train_dir_1):
	os.mkdir(train_dir_1)
if not os.path.isdir(train_dir_2):
	os.mkdir(train_dir_2)

if not os.path.isdir(val_dir):
	os.mkdir(val_dir)
if not os.path.isdir(val_dir_1):
	os.mkdir(val_dir_1)
if not os.path.isdir(val_dir_2):
	os.mkdir(val_dir_2)
#input_data = []
#output_data = []
csv_file_train = open(df_csv_train, 'w', newline='')
csv_file_val = open(df_csv_val, 'w', newline='')

csv_header = ['pics', 'vals']
df_writer_train = csv.writer(csv_file_train, delimiter=';')
df_writer_val = csv.writer(csv_file_val, delimiter=';') 
df_writer_train.writerow(csv_header)
df_writer_val.writerow(csv_header)
count = 0
# each count_val picture goes to validation data
count_val = 5
for j in range(1, 15):
	for i in range(len(frames)-j):
		if (data_deform[i] - data_deform[i+j]) > 1e-6:
			count += 1
			filename = str(count).zfill(5) + '.png'
			if (count % count_val) == 0:
				cv2.imwrite(os.path.join(os.path.join(val_dir, '1'), filename), frames[i])
				cv2.imwrite(os.path.join(os.path.join(val_dir, '2'), filename), frames[i + j])
				df_writer_val.writerow([filename, str(abs(data_deform[i] - data_deform[i+j]))])
			else:
				cv2.imwrite(os.path.join(os.path.join(train_dir, '1'), filename), frames[i])
				cv2.imwrite(os.path.join(os.path.join(train_dir, '2'), filename), frames[i + j])
				df_writer_train.writerow([filename, str(abs(data_deform[i] - data_deform[i+j]))])
		
csv_file.close()
print('Done!')