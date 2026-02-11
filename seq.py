import cv2
import numpy as np
import psutil
import time
import os
import pandas as pd
from openpyxl import load_workbook, Workbook
import cProfile
import pstats
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

import json

# Load configuration from JSON file
with open("user_input_data.json", "r") as file:
    config = json.load(file)

# Extract parameters
video_path = config["video"]
color_channel = config["color_channel"]
rows, cols = config["grids"]["rows"], config["grids"]["cols"]

roi1_x, roi1_y, roi1_width, roi1_height = 47, 202, 441, 141
# roi2_x, roi2_y, roi2_width, roi2_height = 360, 250, 200, 180

num_rows, num_cols = rows, cols

grid_width1 = roi1_width // num_cols
grid_height1 = roi1_height // num_rows
# grid_width2 = roi2_width // num_cols
# grid_height2 = roi2_height // num_rows

frame_count = 0
start_time = time.time()

result_matrix1 = np.zeros((num_rows, num_cols), dtype=int)
result_matrix2 = np.zeros((num_rows, num_cols), dtype=int)

cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('seq_output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

if not os.path.exists('output_MVI_40771'):
    os.makedirs('output_MVI_40771')


ret, frame1 = cap.read()
ret, frame2 = cap.read()

excel_file_path = 'result_matrix1.xlsx'

def append_to_excel(result_matrix):
    try:
        workbook = load_workbook(excel_file_path)
        sheet = workbook.active
    except Exception as e:
        print(f"Error loading workbook: {e}. Creating a new one.")
        workbook = Workbook()
        sheet = workbook.active

    result_df = pd.DataFrame(result_matrix)

    next_row = sheet.max_row + 2 if sheet.max_row > 1 else 1 

    for row_index, row in enumerate(result_df.values):
        for col_index, value in enumerate(row):
            sheet.cell(row=next_row + row_index, column=col_index + 1, value=value)

    workbook.save(excel_file_path)

def process_channel(channel):
    blur = cv2.GaussianBlur(channel, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def apply_histogram_equalization(frame, roi_x, roi_y, roi_width, roi_height, channel_choice):
    roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    if channel_choice == 'gray':
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_equalized = cv2.equalizeHist(roi_gray)
        frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width] = cv2.cvtColor(roi_equalized, cv2.COLOR_GRAY2BGR)
    else:

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        if channel_choice == 'H':
            h = cv2.equalizeHist(h)

        elif channel_choice == 'S':
            s = cv2.equalizeHist(s)
        elif channel_choice == 'V':
            v = cv2.equalizeHist(v)
        hsv_equalized = cv2.merge([h, s, v])
        frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width] = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)
    return frame

def process_hsv(frame1, frame2, channels, channel_choice):
    frame1 = apply_histogram_equalization(frame1, roi1_x, roi1_y, roi1_width, roi1_height, channel_choice)
    frame2 = apply_histogram_equalization(frame2, roi1_x, roi1_y, roi1_width, roi1_height, channel_choice)
    # frame1 = apply_histogram_equalization(frame1, roi2_x, roi2_y, roi2_width, roi2_height, channel_choice)
    # frame2 = apply_histogram_equalization(frame2, roi2_x, roi2_y, roi2_width, roi2_height, channel_choice)
    
    diff = cv2.absdiff(frame1, frame2)
    hsv = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)
    channels_data = [cv2.split(hsv)[i] for i in channels]
    return channels_data


def process_grayscale(frame1, frame2, channel_choice):
    frame1 = apply_histogram_equalization(frame1, roi1_x, roi1_y, roi1_width, roi1_height, channel_choice)
    frame2 = apply_histogram_equalization(frame2, roi1_x, roi1_y, roi1_width, roi1_height, channel_choice)
    # frame1 = apply_histogram_equalization(frame1, roi2_x, roi2_y, roi2_width, roi2_height, channel_choice)
    # frame2 = apply_histogram_equalization(frame2, roi2_x, roi2_y, roi2_width, roi2_height, channel_choice)
    
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return [gray]

def process_grid_cell(args):
    roi_x, roi_y, grid_width, grid_height, frame, channels_data = args
    result = 0
    detection_flags = []
    for channel in channels_data:
        grid_channel = channel[roi_y:roi_y + grid_height, roi_x:roi_x + grid_width]
        contours = process_channel(grid_channel)
        detection_flags.append(any(cv2.contourArea(contour) >= 100 for contour in contours))
    if all(detection_flags):
        result = 1
    return result

def process_grid(roi_x, roi_y, grid_width, grid_height, result_matrix, channels_data):
    for row in range(num_rows):
        for col in range(num_cols):
            grid_x = roi_x + col * grid_width
            grid_y = roi_y + row * grid_height
            grid_frame = frame1[grid_y:grid_y + grid_height, grid_x:grid_x + grid_width]

            if grid_frame.size == 0:
                continue

            detection_flags = []
            for channel in channels_data:
                grid_channel = channel[grid_y:grid_y + grid_height, grid_x:grid_x + grid_width]
                contours = process_channel(grid_channel)
                detection_flags.append(any(cv2.contourArea(contour) >= 100 for contour in contours))
            # AND
            if all(detection_flags):
                result_matrix[row, col] = 1

vechicles = {}
next_vehicle_id = 0

def process_roi(frame1,roi_x, roi_y,roi_width,roi_height,channels_data,frame_count):
    centroid_votes = {}
    global vechicle_count
    global vehicle_count_history
    global next_vehicle_id
    threshold = max(50,1.5*grid_height1)
    matched_vechicle_ids = set()
    keys_to_remove = []

    roi_frame = frame1[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    for channel in channels_data:
        roi_channel = channel[roi_y:roi_y+roi_height , roi_x: roi_x +roi_width]
        countours_roi = process_channel(roi_channel)
        for contour in countours_roi:
            if cv2.contourArea(contour) > 500:  # vehicle-sized
                cv2.drawContours(roi_frame, [contour], -1, (255,0,0), 2)
                centroid = get_centroid(contour)
                if centroid:
                    cx,cy = centroid 
                    centroid =( cx + roi_x , cy +roi_y)
                    centroid_votes[centroid] = centroid_votes.get(centroid,0) +1

    for centroid , count in centroid_votes.items():
        if (count == len(channels_data)):
            best_match = None
            min_distance = threshold
            for key in list(vechicles.keys()):
                if key in matched_vechicle_ids:
                    continue
                value = vechicles[key]
                distance = np.sqrt((centroid[0] - value['centroid'][0])**2 + (centroid[1] - value['centroid'][1])**2)
                if distance < min_distance and (0<= abs(centroid[1]-value['prev_centroid'][1]) <threshold):
                    min_distance = distance
                    best_match = key
            if best_match is not None:
                v = vechicles[best_match]
                v['prev_centroid'] = v['centroid']
                v['centroid'] = centroid
                v['last_seen'] = frame_count
                v['age'] = v.get('age', 0) + 1
                matched_vechicle_ids.add(best_match)
                cv2.putText(frame1,f"ID:{best_match}",(centroid[0],centroid[1]-5),cv2.FONT_HERSHEY_SIMPLEX,1,(32,224,255),2)

            else:
                vechicles[next_vehicle_id] = {'centroid': centroid, 'counted': False, 'prev_centroid':centroid, 'last_seen':frame_count, 'age': 1}
                cv2.putText(frame1,f"ID:{next_vehicle_id}",(centroid[0],centroid[1]-5),cv2.FONT_HERSHEY_SIMPLEX,1,(32,224,255),2)
                next_vehicle_id += 1


    temp_count =0
    for key in list(vechicles.keys()):
        value = vechicles[key]
        if not value['counted'] and value['age'] >= 3 and value['prev_centroid'][1] < COUNT_LINE_Y and value['centroid'][1] >= COUNT_LINE_Y:
            vechicle_count += 1
            temp_count += 1
            vechicles[key]['counted'] = True
            vechicles[key]['last_seen'] = frame_count 

    vehicle_count_history.append(temp_count)    
    cv2.putText(frame1,f"Vehicles Passed: {vechicle_count}",(10, 100),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 255),3)
    
    for key in list(vechicles.keys()):
        value = vechicles[key]
        if frame_count - value['last_seen'] >FPS * 2: 
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del vechicles[key]

def density_calculation(result_matrix):
    density = np.sum(result_matrix==1) / (num_rows * num_cols)
    density_values.append(density)

    FLOW_WINDOW = 2 *FPS  # 1 second
    vehicles_last_1s = sum(vehicle_count_history[-FLOW_WINDOW:])
    WINDOW = FPS
    smoothed_density = np.mean(density_values[-WINDOW:])
    MAX_FLOW = 8  # max vehicles/sec expected for this ROI
    flow_score = min(vehicles_last_1s / MAX_FLOW, 1.0)

    combined_density = 0.7 * smoothed_density + 0.3 * flow_score
    combined_density_values.append(combined_density)
    if len(combined_density_values) > 50:
        p33 = np.percentile(combined_density_values, 33)
        p66 = np.percentile(combined_density_values, 66)

        delta_low = p33 - BASE_LOW
        delta_high = p66 - BASE_HIGH

        delta_low = np.clip(delta_low,-0.1,0.1)
        delta_high = np.clip(delta_high,-0.1,0.1)

        low_th = BASE_LOW + delta_low
        high_th = BASE_HIGH + delta_high    
        if low_th >= high_th:
            high_th = low_th + 0.05            
    else:
        low_th, high_th = 0.3, 0.6

    if combined_density > high_th:
        state = "High"
    elif combined_density > low_th:
        state = "Medium"
    else:
        state = "Low"
    cv2.putText(frame1, f"Density: {combined_density:.2f} ({state})", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

user_choice = color_channel  # This can be 'H', 'S', 'V', or 'gray'
he_choice = 'V'  # User's choice for histogram equalization

choices = {
    'H': [0],
    'S': [1],
    'V': [2],
    'H+S': [0, 1],
    'H+V': [0, 2],
    'S+V': [1, 2],
    'H+S+V': [0, 1, 2],
    'gray': 'gray'
}

channels = choices[user_choice]
frame_times = []
memory_usages = []
density_values = []
vehicle_count_history =[]
combined_density_values = []

## find the number of vechicles 
COUNT_LINE_Y = roi1_y + roi1_height // 2
vechicle_count = 0
BASE_LOW =0.25
BASE_HIGH =0.55
FPS = 30

def get_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return int(M['m10']/M['m00']),int(M['m01']/M['m00'])


def main():
    global frame1, frame2, ret, frame_count
    while cap.isOpened():
        if not ret : 
            break
        frame_count += 1

        frame_start_time = time.time()

        if frame1.shape[:2] == frame2.shape[:2]:
            if channels == 'gray':
                channels_data = process_grayscale(frame1, frame2, he_choice)
            else:
                channels_data = process_hsv(frame1, frame2, channels, he_choice)

            result_matrix1.fill(0)
            # result_matrix2.fill(0)

            process_grid(roi1_x, roi1_y, grid_width1, grid_height1, result_matrix1, channels_data)
            # process_grid(roi2_x, roi2_y, grid_width2, grid_height2, result_matrix2, channels_data, frame_count)
            
            process_roi(frame1,roi1_x,roi1_y,roi1_width,roi1_height,channels_data,frame_count)

            # append_to_excel(result_matrix1)

            for row in range(num_rows):
                for col in range(num_cols):
                    if result_matrix1[row, col] == 0:
                        grid_x = roi1_x + col * grid_width1
                        grid_y = roi1_y + row * grid_height1
                        cv2.rectangle(frame1, (grid_x, grid_y), (grid_x + grid_width1, grid_y + grid_height1), (0, 0, 255), 2)

            # for row in range(num_rows):
            #     for col in range(num_cols):
            #         if result_matrix2[row, col] == 0:
            #             grid_x = roi2_x + col * grid_width2
            #             grid_y = roi2_y + row * grid_height2
            #             cv2.rectangle(frame1, (grid_x, grid_y), (grid_x + grid_width2, grid_y + grid_height2), (0, 0, 255), 2)

            for row in range(num_rows):
                for col in range(num_cols):
                    if result_matrix1[row, col] == 1:
                        grid_x = roi1_x + col * grid_width1
                        grid_y = roi1_y + row * grid_height1
                        cv2.rectangle(frame1, (grid_x, grid_y), (grid_x + grid_width1, grid_y + grid_height1), (0, 255, 0), 2)


            #density calculation and display
            density_calculation(result_matrix1)

            cv2.line(frame1,(roi1_x,COUNT_LINE_Y),(roi1_x+roi1_width,COUNT_LINE_Y),(255,255,255),2)

            # for row in range(num_rows):
            #     for col in range(num_cols):
            #         if result_matrix2[row, col] == 1:
            #             grid_x = roi2_x + col * grid_width2
            #             grid_y = roi2_y + row * grid_height2
            #             cv2.rectangle(frame1, (grid_x, grid_y), (grid_x + grid_width2, grid_y + grid_height2), (0, 255, 0), 2)

            frame_end_time = time.time()
            frame_time = frame_end_time - frame_start_time
            frame_times.append(frame_time)

            memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
            memory_usages.append(memory_usage)

            cv2.putText(frame1, "Frame: {}".format(frame_count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            out.write(frame1)

            output_filename = f'output_seq/frame_{frame_count:04d}.jpg'
            cv2.imwrite(output_filename, frame1)
            cv2.imshow('Frame', frame1)
            frame1 = frame2
            ret, frame2 = cap.read()

        if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.waitKey(1) & 0xFF == ord('Q')):  # Reduce delay
            break

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution Time: {:.2f} seconds".format(execution_time))

    total_memory_usage = psutil.Process().memory_info().rss
    print("Total Memory Usage: {:.2f} MB".format(total_memory_usage / (1024 * 1024)))

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("frames: "f"{frame_count}")

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(10)  # Print the top 10 functions by cumulative time

#
