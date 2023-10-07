import csv
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Paths to data files
data_path = './Data/SM_BLUE_v2'
# Get lists of files
CSVs = [f for f in os.listdir(data_path) if f[-4:] == '.csv']
CSVs.sort()
videos = [f for f in os.listdir(data_path) if f[-4:] == '.mp4']
videos.sort()
assert len(CSVs) == len(videos), 'Oops! There is a difference in the files amount'
files_amount = len(CSVs)


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
base_dir = './'
data_dir = os.path.join(base_dir, 'train_v7')
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

train_dir_1 = os.path.join(train_dir, '1')
train_dir_2 = os.path.join(train_dir, '2')

val_dir_1 = os.path.join(val_dir, '1')
val_dir_2 = os.path.join(val_dir, '2')

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

deforms_train = []
deforms_val = []
pic_name_train = []
pic_name_val = []
count = 0
count_val = 5

for i in range(files_amount):
	load_times, load_deforms = extract_data_csv(os.path.join(data_path, CSVs[i]))
	timestamps, frames = extract_data_mp4(os.path.join(data_path, videos[i]))
	data_deform = np.interp(timestamps, load_times, load_deforms)
	
	for j in range(1, (len(frames)-1)):
		for k in range(1, 7):
			frame_1 = []
			frame_2 = []
			if k == 1:
				frame_1 = frames[j]
				frame_2 = frames[0]
			elif k == 2:
				frame_1 = cv2.rotate(frames[j], cv2.ROTATE_90_CLOCKWISE)
				frame_2 = cv2.rotate(frames[0], cv2.ROTATE_90_CLOCKWISE)
			elif k == 3:
				frame_1 = cv2.rotate(frames[j], cv2.ROTATE_180)
				frame_2 = cv2.rotate(frames[0], cv2.ROTATE_180)
			elif k == 4:
				frame_1 = cv2.rotate(frames[j], cv2.ROTATE_90_COUNTERCLOCKWISE)
				frame_2 = cv2.rotate(frames[0], cv2.ROTATE_90_COUNTERCLOCKWISE)
			elif k == 5:
				frame_1 = cv2.flip(frames[j], 0)
				frame_2 = cv2.flip(frames[0], 0)
			elif k == 6:
				frame_1 = cv2.flip(frames[j], 1)
				frame_2 = cv2.flip(frames[0], 1)
			elif k == 7:
				frame_1 = cv2.flip(frames[j], -1)
				frame_2 = cv2.flip(frames[0], -1)

			count += 1
			filename = str(count).zfill(7) + '.png'
			if (count % count_val) == 0:
				cv2.imwrite(os.path.join(os.path.join(val_dir, '1'), filename), frame_1)
				cv2.imwrite(os.path.join(os.path.join(val_dir, '2'), filename), frame_2)
				deforms_val.append(data_deform[j])
				pic_name_val.append(filename)
			else:
				cv2.imwrite(os.path.join(os.path.join(train_dir, '1'), filename), frame_1)
				cv2.imwrite(os.path.join(os.path.join(train_dir, '2'), filename), frame_2)
				deforms_train.append(data_deform[j])
				pic_name_train.append(filename)


m1 = max(deforms_train)
m2 = max(deforms_val)
m = 0
if m1 >= m2:
	m = m1
else:
	m = m2

deforms_train /= m
deforms_val /= m



df_csv_train = os.path.join(train_dir, 'df.csv')
df_csv_val = os.path.join(val_dir, 'df.csv')

csv_file_train = open(df_csv_train, 'w', newline='')
csv_file_val = open(df_csv_val, 'w', newline='')
csv_header = ['pics', 'vals']
df_writer_train = csv.writer(csv_file_train, delimiter=';')
df_writer_val = csv.writer(csv_file_val, delimiter=';') 
df_writer_train.writerow(csv_header)
df_writer_val.writerow(csv_header)

for i in range(len(deforms_train)):
	df_writer_train.writerow([pic_name_train[i], str(abs(deforms_train[i]))])
for i in range(len(deforms_val)):
	df_writer_val.writerow([pic_name_val[i], str(abs(deforms_val[i]))])


csv_file_train.close()
csv_file_val.close()
print('Done!')