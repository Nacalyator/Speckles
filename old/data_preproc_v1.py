import csv
from datetime import time
from genericpath import isdir
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
		frame = cv2.resize(frame, [250, 250])
		#filt = cv2.bilateralFilter(frame, 7, 500, 500)
		filt = cv2.medianBlur(frame, 9)
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
data_dir = os.path.join(base_dir, 'train')
df_csv = os.path.join(data_dir, 'df.csv')
if not os.path.isdir(data_dir):
	os.mkdir(data_dir)

#input_data = []
#output_data = []
csv_file = open(df_csv, 'w', newline='')
csv_header = ['pics', 'vals']
df_writer = csv.writer(csv_file, delimiter=';') 
df_writer.writerow(csv_header)
count = 0
for j in range(1, 15):
	for i in range(len(frames)-j):
		filename = str(count).zfill(5) + '.png'
		#buff = np.concatenate((frames[i], frames[i + j]), axis=1)
		cv2.imwrite(os.path.join(data_dir, filename), np.concatenate((frames[i], frames[i + j]), axis=1))
		df_writer.writerow([filename, str(abs(data_deform[i] - data_deform[i+j]))])
		count += 1
		#input_data.append(np.concatenate((frames[i], frames[i + j]), axis=1))
		#output_data.append(abs(data_deform[i] - data_deform[i+j]))
csv_file.close()