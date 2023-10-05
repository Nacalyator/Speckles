import csv
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Paths to data files
video_path = './Data/SM_RED/Test/RED_SM_55.mp4'
csv_path = './Data/SM_RED/Test/RED_SM_55.csv'

base_dir = './'
data_dir = os.path.join(base_dir, 'train_v8_55r')

# Read data from csv and video functions
def extract_data_csv(fname):
	load_times = []
	load_deforms = []
	csv_file = open(fname, 'r', newline='', encoding='latin-1')
	csv_reader = csv.reader(csv_file, delimiter=';')
	csv_reader.__next__()
	for row in csv_reader:
		load_times.append(float(row[0].replace(',', '.')))
		load_deforms.append(float(row[3].replace(',', '.')))
	print('CSV file has been read: ', fname)
	return load_times, load_deforms

def extract_data_mp4(fname):
	timestamps = []
	frames = []
	cap = cv2.VideoCapture(fname)
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
			frames.append(filt)
		else:
			break
	print('Frames have been read: ', fname)
	return timestamps, frames

# Read data

train_dir = os.path.join(data_dir, 'train')

train_dir_1 = os.path.join(train_dir, '1')
train_dir_2 = os.path.join(train_dir, '2')

if not os.path.isdir(data_dir):
	os.mkdir(data_dir)
if not os.path.isdir(train_dir):
	os.mkdir(train_dir)
if not os.path.isdir(train_dir_1):
	os.mkdir(train_dir_1)
if not os.path.isdir(train_dir_2):
	os.mkdir(train_dir_2)

deforms_train = []
pic_name_train = []
count = 0

load_times, load_deforms = extract_data_csv(csv_path)
timestamps, frames = extract_data_mp4(video_path)
data_deform = np.interp(timestamps, load_times, load_deforms)
	
for j in range(1, (len(frames)-1)):
	count += 1
	filename = str(count).zfill(6) + '.png'
	frame_1 = frames[j]
	frame_2 = frames[0]
	cv2.imwrite(os.path.join(os.path.join(train_dir, '1'), filename), frame_1)
	cv2.imwrite(os.path.join(os.path.join(train_dir, '2'), filename), frame_2)
	deforms_train.append(data_deform[j])
	pic_name_train.append(filename)
	


m1 = max(deforms_train)
deforms_train /= m1



df_csv_train = os.path.join(train_dir, 'df.csv')

csv_file_train = open(df_csv_train, 'w', newline='')
csv_header = ['pics', 'vals']
df_writer_train = csv.writer(csv_file_train, delimiter=';')
df_writer_train.writerow(csv_header)

for i in range(len(deforms_train)):
	df_writer_train.writerow([pic_name_train[i], str(abs(deforms_train[i]))])

csv_file_train.close()
print('Done!')