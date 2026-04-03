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
from collections import Counter


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

def process_grid_channel(channel):
    blur = cv2.GaussianBlur(channel, (5,5), 1) 
    _, thresh = cv2.threshold(blur,50, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=5)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def process_channel(channel):
    blur = cv2.GaussianBlur(channel, (11,11),1) 
    _, thresh = cv2.threshold(blur,120, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=20)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    frame1_blur = cv2.GaussianBlur(frame1, (7,7), 0)
    frame2_blur = cv2.GaussianBlur(frame2, (7,7), 0)

    diff = cv2.absdiff(frame1_blur, frame2_blur)
    hsv = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)
    channels_data = [cv2.split(hsv)[i] for i in channels]
    return channels_data


def process_grayscale(frame1, frame2, channel_choice):
    frame1 = apply_histogram_equalization(frame1, roi1_x, roi1_y, roi1_width, roi1_height, channel_choice)
    frame2 = apply_histogram_equalization(frame2, roi1_x, roi1_y, roi1_width, roi1_height, channel_choice)
    # frame1 = apply_histogram_equalization(frame1, roi2_x, roi2_y, roi2_width, roi2_height, channel_choice)
    # frame2 = apply_histogram_equalization(frame2, roi2_x, roi2_y, roi2_width, roi2_height, channel_choice)
    frame1_blur = cv2.GaussianBlur(frame1, (7,7), 0)
    frame2_blur = cv2.GaussianBlur(frame2, (7,7), 0)
    diff = cv2.absdiff(frame1_blur, frame2_blur)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return [gray]

'''
def process_grid_cell(args):
    roi_x, roi_y, grid_width, grid_height, frame, channels_data = args
    result = 0
    detection_flags = []
    for channel in channels_data:
        grid_channel = channel[roi_y:roi_y + grid_height, roi_x:roi_x + grid_width]
        contours = process_grid_channel(grid_channel)
        detection_flags.append(any(cv2.contourArea(contour) >= 100 for contour in contours))
    if all(detection_flags):
        result = 1
    return result
'''

'''
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
                contours = process_grid_channel(grid_channel)
                detection_flags.append(any(cv2.contourArea(contour) >= 100 for contour in contours))
            # AND
            if all(detection_flags):
                result_matrix[row, col] = 1
'''

def process_grid(roi_x, roi_y, grid_width, grid_height, result_matrix, channels_data):
    combined_contours = []

    for row in range(num_rows):
        for col in range(num_cols):
            grid_x = roi_x + col * grid_width
            grid_y = roi_y + row * grid_height

            grid_frame = frame1[grid_y:grid_y + grid_height, grid_x:grid_x + grid_width]
            if grid_frame.size == 0:
                continue

            detection_flags = []
            cell_contours = []

            for channel in channels_data:
                grid_channel = channel[grid_y:grid_y + grid_height, grid_x:grid_x + grid_width]
                contours = process_grid_channel(grid_channel)

                # filter valid contours
                valid_contour = [c for c in contours if cv2.contourArea(c) > 250]

                detection_flags.append(len(valid_contour) > 0)
                cell_contours.extend(valid_contour)

            if all(detection_flags):
                result_matrix[row, col] = 1
                combined_contours.append((row, col, cell_contours))

    return combined_contours

vechicles = {}
next_vehicle_id = 0

def Classify_vechicle(contour):

    area = cv2.contourArea(contour)
    if area < 800:
        return None, "Unknown"

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h) if h > 0 else 0

    label = "Unknown"
    # Bike
    if w < 40 and area < 3500:
        label = "Bike"
    # Car
    elif 40 <= w < 150:
        label = "Car"
    # Bus
    elif w >= 200 and aspect_ratio > 3:
        label = "Bus"
    # Truck
    elif w >= 150:
        label = "TRUCK"
    return label, (x, y, w, h)
    
'''
def process_tracking_counting(frame1,roi_x, roi_y,roi_width,roi_height,channels_data,frame_count):
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
                    if centroid not in centroid_votes:
                        centroid_votes[centroid] = {
                            "count": 1,
                            "contour": contour
                        }
                    else:
                        centroid_votes[centroid]["count"] += 1

    for centroid , values in centroid_votes.items():
        count = values["count"]
        if (count <= len(channels_data)):
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
                v['contour'] = values['contour']
                label, bbox = Classify_vechicle(v['contour'])
                v.setdefault('labels', [])
                if bbox is not None and label is not None:
                    x, y, w, h = bbox
                    v['bbox'] = (x + roi_x, y + roi_y, w, h)
                    v['labels'].append(label)
                    if len(v['labels']) > 10:
                        v['labels'].pop(0)
                    votes = Counter(v['labels'])
                    v['final_label'] = votes.most_common(1)[0][0]
                v['prev_centroid'] = v['centroid']
                v['centroid'] = centroid
                v['last_seen'] = frame_count
                v['age'] = v.get('age', 0) + 1
                matched_vechicle_ids.add(best_match)
                cv2.putText(frame1,f"ID:{best_match}",(centroid[0],centroid[1]-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2)
                if v.get('bbox') is not None:
                    x,y,w,h = v['bbox']
                    cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,0,0),2)
                    cv2.putText(frame1,f"{v['final_label']}",(centroid[0],centroid[1]+20),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 255),2)

            else:
                vechicles[next_vehicle_id] = {'centroid': centroid, 'counted': False, 'prev_centroid':centroid, 'last_seen':frame_count, 'age': 1 , 'contour': values['contour']}
                cv2.putText(frame1,f"ID:{next_vehicle_id}",(centroid[0],centroid[1]-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2)
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
'''
TOTAL_CELLS = num_rows * num_cols

def classify_by_grid(vehicle_cells):
    normalized = vehicle_cells / TOTAL_CELLS
    if normalized < 0.08:      # ~1-2 cells → Bike
        return "Bike"
    elif normalized < 0.20:    # ~3-5 cells → Car
        return "Car"
    else:                      # 6+ cells → Truck
        return "Truck"

def process_tracking_counting(frame1, roi_x, roi_y, frame_count, cell_contours):
    global vechicle_count, vehicle_count_history, next_vehicle_id

    matched_vechicle_ids = set()
    keys_to_remove = []

    #max pixel the vehicle can move in 2 frames 
    MATCH_THRESHOLD = roi1_height // 3

    cv2.putText(frame1, f"Vehicles Passed: {vechicle_count}",(10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)


    if not cell_contours:
        vehicle_count_history.append(0)
        for key in list(vechicles.keys()):
            if frame_count - vechicles[key]['last_seen'] > FPS * 2:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del vechicles[key]
        return


    raw_detections = []
    for (row, col, contours) in cell_contours:
        grid_x = roi_x + col * grid_width1
        grid_y = roi_y + row * grid_height1
        for contour in contours:
            centroid = get_centroid(contour)
            if centroid is None:
                continue
            cx = centroid[0] + grid_x
            cy = centroid[1] + grid_y
            raw_detections.append((cx, cy, contour))


    # No of vehicle can be present in the same lane in a frame
    MAX_VEHICLES_IN_LANE = 2
    # more the vehicle less the merge radius 
    MERGE_RADIUS = roi1_width // (MAX_VEHICLES_IN_LANE * 2)
    merged = []
    used = [False] * len(raw_detections)

    for i, (cx, cy, contour) in enumerate(raw_detections):
        if used[i]:
            continue
        cluster_cx = [cx]
        cluster_cy = [cy]
        cluster_size =1
        best_contour = contour
        best_area = cv2.contourArea(contour)
        used[i] = True

        for j, (cx2, cy2, contour2) in enumerate(raw_detections):
            if used[j]:
                continue
            dist = np.sqrt((cx - cx2)**2 + (cy - cy2)**2)
            if dist < MERGE_RADIUS:
                cluster_cx.append(cx2)
                cluster_cy.append(cy2)
                cluster_size +=1
                area2 = cv2.contourArea(contour2)
                if area2 > best_area:
                    best_area = area2
                    best_contour = contour2
                used[j] = True

        # representative centroid = mean of cluster
        merged.append((
            int(np.mean(cluster_cx)),
            int(np.mean(cluster_cy)),
            best_contour,
            cluster_size
        ))

    # tracking same logic, now operates on merged detections
    vehicle_cell_counts = {}
    # --- tracking ---
    for cx, cy, contour, cluster_size in merged:
        best_match = None
        min_distance = MATCH_THRESHOLD

        for key in list(vechicles.keys()):
            if key in matched_vechicle_ids:
                continue
            value = vechicles[key]
            px, py = value['centroid']
            distance = np.sqrt((cx - px)**2 + (cy - py)**2)
            if distance < min_distance and abs(cy - py) < 2 * MATCH_THRESHOLD:
                min_distance = distance
                best_match = key

        if best_match is not None:
            v = vechicles[best_match]
            v['prev_centroid'] = v['centroid']
            v['centroid'] = (cx, cy)
            v['contour'] = contour
            v['last_seen'] = frame_count
            v['age'] = v.get('age', 0) + 1
            matched_vechicle_ids.add(best_match)
        else:
            vechicles[next_vehicle_id] = {
                'centroid': (cx, cy),
                'prev_centroid': (cx, cy),
                'last_seen': frame_count,
                'age': 1,
                'counted': False,
                'contour': contour,
                'cell_count': 1,
                'final_label': 'Unknown',
                'bbox': None
            }
            next_vehicle_id += 1
            matched_vechicle_ids.add(next_vehicle_id)


    # Reset cell counts for all active vehicles this frame
    for key in vechicles:
        vechicles[key]['frame_cell_count'] = 0
        vechicles[key]['cell_pts'] = []   

    for (row, col, contours) in cell_contours:
        grid_x = roi_x + col * grid_width1
        grid_y = roi_y + row * grid_height1
        cell_cx = grid_x + grid_width1 // 2
        cell_cy = grid_y + grid_height1 // 2

        best_vid = None
        best_dist = MATCH_THRESHOLD * 2
        for key, v in vechicles.items():
            if frame_count - v['last_seen'] > 1: 
                continue
            px, py = v['centroid']
            d = np.sqrt((cell_cx - px)**2 + (cell_cy - py)**2)
            if d < best_dist:
                best_dist = d
                best_vid = key

        if best_vid is not None:
            vechicles[best_vid]['frame_cell_count'] += 1
    
            vechicles[best_vid]['cell_pts'].extend([
                (grid_x, grid_y),
                (grid_x + grid_width1, grid_y + grid_height1)
            ])


    for key, v in vechicles.items():
        if frame_count - v['last_seen'] > 1:
            continue   # skip stale vehicles

        fc = v.get('frame_cell_count', 0)
        if fc > 0:
            prev = v.get('cell_count', fc)
            v['cell_count'] = int(0.6 * prev + 0.4 * fc) 

           
            pts = v.get('cell_pts', [])
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bx, by = min(xs), min(ys)
                bw = max(xs) - bx
                bh = max(ys) - by
                v['bbox'] = (bx, by, bw, bh)

        v['final_label'] = classify_by_grid(v.get('cell_count', 1))

        # draw bbox + label
        if v.get('bbox') is not None:
            bx, by, bw, bh = v['bbox']
            cx, cy = v['centroid']
            cv2.rectangle(frame1, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
            text_y = max(by - 5, 15)
            cv2.putText(frame1, f"ID:{key} | {v['final_label']} | cells:{v['cell_count']}",(bx, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    
    temp_count = 0
    for key in list(vechicles.keys()):
        value = vechicles[key]
        if (not value['counted'] and value['age'] >= 2 and
                value['prev_centroid'][1] < COUNT_LINE_Y and
                value['centroid'][1] >= COUNT_LINE_Y):
            vechicle_count += 1
            temp_count += 1
            vechicles[key]['counted'] = True
            vechicles[key]['last_seen'] = frame_count

    vehicle_count_history.append(temp_count)

    # cleanup  
    for key in list(vechicles.keys()):
        if frame_count - vechicles[key]['last_seen'] > FPS * 2:
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
all_frames_data = []

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

            cell_contours = process_grid(roi1_x, roi1_y, grid_width1, grid_height1, result_matrix1, channels_data)
            # process_grid(roi2_x, roi2_y, grid_width2, grid_height2, result_matrix2, channels_data, frame_count)
            
            process_tracking_counting(frame1,roi1_x,roi1_y,frame_count,cell_contours)

            all_frames_data.append((frame_count, result_matrix1.copy()))

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

def save_all_to_excel():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Grid Data"

    for frame_number, matrix in all_frames_data:
        sheet.append([f"Frame {frame_number}"])
        for row in matrix:
            sheet.append(list(row))
        sheet.append([])

    workbook.save(excel_file_path)

save_all_to_excel()
