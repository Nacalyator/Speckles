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
		filt = cv2.fastNlMeansDenoising(frame,None, 8, 11, 21)
		#pic = np.concatenate((cv2.resize(frame, [500, 500]), cv2.resize(filt, [500, 500])), axis=1)
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
data_dir = os.path.join(base_dir, 'train_v6')
train_dir = os.path.join(data_dir, 'train')

df_csv_train = os.path.join(train_dir, 'df.csv')

if not os.path.isdir(data_dir):
	os.mkdir(data_dir)

if not os.path.isdir(train_dir):
	os.mkdir(train_dir)
#input_data = []
#output_data = []
csv_file_train = open(df_csv_train, 'w', newline='')

csv_header = ['pics', 'vals']
df_writer_train = csv.writer(csv_file_train, delimiter=';')
df_writer_train.writerow(csv_header)
count = 0

for i in range(len(frames)-1):
	for j in range(1, 7):
		frame_1 = frames[i]
		frame_2 = frames[0]
		if j == 1:
			frame_1 = frames[i]
			frame_2 = frames[0]
		elif j == 2:
			frame_1 = cv2.rotate(frames[i], cv2.ROTATE_90_CLOCKWISE)
		elif j == 3:
			frame_1 = cv2.rotate(frames[i], cv2.ROTATE_180)
		elif j == 4:
			frame_1 = cv2.rotate(frames[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
		elif j == 5:
			frame_1 = cv2.flip(frames[i], 0)
		elif j == 6:
			frame_1 = cv2.flip(frames[i], 1)
		elif j == 7:
			frame_1 = cv2.flip(frames[i], -1)

		count += 1
		filename = str(count).zfill(5) + '.png'
		
		cv2.imwrite(os.path.join(train_dir, filename), frame_1)
		df_writer_train.writerow([filename, str(abs(data_deform[i]))])

csv_file.close()
print('Done!')