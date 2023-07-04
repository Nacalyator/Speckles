import csv
import cv2

# Data files
video_path = './RED.mp4'
csv_path = './RED.csv'

# Read data from csv
load_times = []
load_deforms = []
csv_file = open(csv_path, 'r', newline='')
csv_reader = csv.reader(csv_file, delimiter=';')
csv_reader.__next__()
csv_reader.__next__()
csv_reader.__next__()
csv_reader.__next__()
for row in csv_reader:
	load_times.append(float(row[0].replace(',', '.')))
	load_deforms.append(float(row[3].replace(',', '.')))

# Read video frames and imfo
timestamps = []
frames = []
cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)
video_total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
while (cap.isOpened()):
	ret, frame = cap.read()
	if ret == True:
		timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame = cv2.resize(frame, [250, 250])
		frame = cv2.bilateralFilter(frame,9,75,75)
		cv2.imshow('work', frame)
		test = 1
	else:
		break
	