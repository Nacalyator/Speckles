import csv
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Paths to data files
video_path = './Data/SM_BLUE_v2/Test/BLUE_SM_54.mp4'
csv_path = './Data/SM_BLUE_v2/Test/BLUE_SM_54.csv'
output_file = './intensity_data_54b'

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
			intensity = np.mean(frame)
			frames.append(intensity)
		else:
			break
	print('Frames have been read: ', fname)
	return timestamps, frames



# Read data

pic_name_train = []
count = 0

load_times, load_deforms = extract_data_csv(csv_path)
timestamps, frames = extract_data_mp4(video_path)
data_deform = np.interp(timestamps, load_times, load_deforms)

m1 = max(data_deform)
data_deform /= m1

csv_output = open(output_file, 'w', newline='')
csv_header = ['deforms', 'intensity']
csv_writer = csv.writer(csv_output, delimiter=';')
csv_writer.writerow(csv_header)

for i in range(len(data_deform)):
	csv_writer.writerow([str(abs(data_deform[i])), str(frames[i])])

#csv_writer.close()
print('Done!')
frames = frames - np.min(frames)
frames = frames / np.max(frames)


plt.figure('Mesuared and estimated data')
plt.plot([i for i in range(len(data_deform))], data_deform, 'b', label='Measured data')
plt.plot([i for i in range(len(data_deform))], frames, 'bo', label='Estimated data')
plt.title('Mesuared and estimated data')
plt.xlabel('Frame')
plt.ylabel('Normalized deformation')
plt.legend()
plt.show()