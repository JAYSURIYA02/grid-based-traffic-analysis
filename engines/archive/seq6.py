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
import seaborn as sns
import json
import xml.etree.ElementTree as ET
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
'''
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

CALIBRATION_FRAMES = 60
col_span_observations = []
recent_col_spans = []
RECALIBRATION_INTERVAL = 300
calibrated = False
CAR_THRESH = 2
VAN_THRESH = 4
active_classes = 1  # how many classes actually detected

def detect_num_classes(observations):
    """
    Use gap analysis to find how many natural clusters exist.
    Much simpler than K-means but equally effective for 1-3 classes.
    """
    if len(observations) < 10:
        return 1

    unique = sorted(set(observations))
    if len(unique) == 1:
        return 3

    # find gaps between consecutive unique values
    gaps = [(unique[i+1] - unique[i], i) for i in range(len(unique)-1)]
    gaps.sort(reverse=True)  # largest gap first

    std = np.std(observations)
    span = max(observations) - min(observations)

    # if spread is very small → all same type
    if std < 0.5 or span <= 1:
        return 3

    # if two clear groups exist (1 big gap)
    if gaps[0][0] >= 1:
        # check if there's a second meaningful gap
        if len(gaps) > 1 and gaps[1][0] >= 2:
            return 3
        return 3
    return 3

def update_calibration(size_score):
    global calibrated, CAR_THRESH, VAN_THRESH, active_classes

    if size_score > 0:
        col_span_observations.append(size_score)
        recent_col_spans.append(size_score)
        if len(recent_col_spans) > 100:
            recent_col_spans.pop(0)

    if not calibrated:
        if frame_count >= CALIBRATION_FRAMES and len(col_span_observations) >= 10:

            active_classes = 3
            print(f"[Calibration] Detected {active_classes} vehicle class(es)")

            if active_classes == 1:
                # ── Case 1 & 3: Only one type exists ──────────────
                # Don't split — label everything as the dominant class
                # Use median to identify what type it is
                median_span = np.median(col_span_observations)
                # We still need thresholds but make them irrelevant
                # by setting them so everything falls in one bucket
                CAR_THRESH = int(max(col_span_observations)) + 1
                VAN_THRESH = CAR_THRESH + 1
                print(f"[Calibration] Single class → all labelled by size")

            elif active_classes == 2:
                # ── Case 2: Two types exist ────────────────────────
                # Use only 1 split point, collapse 3rd class
                CAR_THRESH = max(1, int(np.percentile(col_span_observations, 50)))
                VAN_THRESH = int(max(col_span_observations)) + 1  # no Bus
                print(f"[Calibration] Two classes → CAR≤{CAR_THRESH} VAN>{CAR_THRESH}")

            else:
                # ── Case 3: All three types exist ─────────────────
                CAR_THRESH = max(1, int(np.percentile(col_span_observations, 33)))
                VAN_THRESH = max(CAR_THRESH + 1,
                                 int(np.percentile(col_span_observations, 66)))
                # Safety: collapse if thresholds too close
                if VAN_THRESH - CAR_THRESH <= 1:
                    VAN_THRESH = CAR_THRESH + 2
                print(f"[Calibration] Three classes → "
                      f"CAR≤{CAR_THRESH} VAN≤{VAN_THRESH} BUS>{VAN_THRESH}")

            calibrated = True

        return

    # ── Periodic recalibration ─────────────────────────────────────
    if frame_count % RECALIBRATION_INTERVAL == 0 and len(recent_col_spans) >= 20:
        new_classes = detect_num_classes(recent_col_spans)

        if new_classes != active_classes:
            # scene changed — full recalibrate
            active_classes = new_classes
            calibrated = False
            col_span_observations.clear()
            print(f"[Recalibration] Class count changed to {new_classes} — resetting")
            return

        if active_classes == 1:
            CAR_THRESH = int(max(recent_col_spans)) + 1
            VAN_THRESH = CAR_THRESH + 1

        elif active_classes == 2:
            new_car = max(1, int(np.percentile(recent_col_spans, 50)))
            CAR_THRESH = int(round(0.7 * CAR_THRESH + 0.3 * new_car))
            VAN_THRESH = int(max(recent_col_spans)) + 1

        else:
            new_car = max(1, int(np.percentile(recent_col_spans, 33)))
            new_van = max(new_car + 1, int(np.percentile(recent_col_spans, 66)))
            CAR_THRESH = int(round(0.7 * CAR_THRESH + 0.3 * new_car))
            VAN_THRESH = int(round(0.7 * VAN_THRESH + 0.3 * new_van))
            if VAN_THRESH - CAR_THRESH <= 1:
                VAN_THRESH = CAR_THRESH + 2

        print(f"[Recalibration] CAR≤{CAR_THRESH} VAN≤{VAN_THRESH} "
              f"BUS>{VAN_THRESH} classes={active_classes}")

def classify_by_grid(size_score):
    """Classification respects how many classes were actually detected."""
    if active_classes == 1:
        # identify the single class by where the median falls
        median_span = np.median(col_span_observations) if col_span_observations else 2
        if median_span <= 3:
            return "Car"
        elif median_span <= 5:
            return "Van"
        else:
            return "Bus"

    elif active_classes == 2:
        # only Car and Van exist
        if size_score <= CAR_THRESH:
            return "Car"
        else:
            return "Van"     # never returns Bus

    else:
        # full 3-class
        if size_score <= CAR_THRESH:
            return "Car"
        elif size_score <= VAN_THRESH:
            return "Van"
        else:
            return "Bus"
        
        
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
                'col_span': 1,
                'row_span': 1,
                'size_score': 1,
                'final_label': 'Unknown',
                'bbox': None
            }
            matched_vechicle_ids.add(next_vehicle_id)
            next_vehicle_id += 1


    # Reset cell counts for all active vehicles this frame
    for key in vechicles:
        vechicles[key]['frame_cell_count'] = 0
        vechicles[key]['cell_pts'] = []
        vechicles[key]['occupied_cols'] = set()
        vechicles[key]['occupied_rows'] = set()

    for (row, col, contours) in cell_contours:
        grid_x = roi_x + col * grid_width1
        grid_y = roi_y + row * grid_height1
        cell_cx = grid_x + grid_width1 // 2
        cell_cy = grid_y + grid_height1 // 2

        best_vid = None
        best_dist = MATCH_THRESHOLD * 2
        for key, v in vechicles.items():
            if v['last_seen'] != frame_count:
                continue
            px, py = v['centroid']
            d = np.sqrt((cell_cx - px)**2 + (cell_cy - py)**2)
            if d < best_dist:
                best_dist = d
                best_vid = key

        if best_vid is not None:
            vechicles[best_vid]['frame_cell_count'] += 1
            vechicles[best_vid]['occupied_cols'].add(col)
            vechicles[best_vid]['occupied_rows'].add(row)

            vechicles[best_vid]['cell_pts'].extend([
                (grid_x, grid_y),
                (grid_x + grid_width1, grid_y + grid_height1)
            ])


    for key, v in vechicles.items():
        if v['last_seen'] != frame_count:
            continue

        fc = v.get('frame_cell_count', 0)
        if fc > 0:
            prev = v.get('cell_count', fc)
            v['cell_count'] = int(0.6 * prev + 0.4 * fc)

            curr_col_span = len(v['occupied_cols'])
            curr_row_span = len(v['occupied_rows'])
            prev_col_span = v.get('col_span', curr_col_span)
            prev_row_span = v.get('row_span', curr_row_span)
            v['col_span'] = int(round(0.6 * prev_col_span + 0.4 * curr_col_span))
            v['row_span'] = int(round(0.6 * prev_row_span + 0.4 * curr_row_span))

            pts = v.get('cell_pts', [])
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bx, by = min(xs), min(ys)
                bw = max(xs) - bx
                bh = max(ys) - by
                v['bbox'] = (bx, by, bw, bh)

        v['size_score'] = v.get('col_span', 1) * v.get('row_span', 1)
        update_calibration(v.get('size_score', 1))

        v['final_label'] = classify_by_grid(v.get('size_score', 1))

        if v.get('bbox') is not None:
            bx, by, bw, bh = v['bbox']
            status = "CAL" if not calibrated else v['final_label']
            cv2.rectangle(frame1, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
            text_y = max(by - 5, 15)
            cv2.putText(
                frame1,
                f"ID:{key} | {status} | cols:{v.get('col_span', 1)}",
                (bx, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

    
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


def map_detrac_label(vehicle_type):
    mapping = {
        'car': 'Car',
        'van': 'Van',
        'bus': 'Bus',
        'others': 'Unknown'
    }
    return mapping.get(vehicle_type.lower(), 'Unknown')


def load_detrac(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    gt_data = {}

    for frame in root.findall('frame'):
        frame_id = int(frame.get('num'))
        gt_data[frame_id] = []

        target_list = frame.find('target_list')
        if target_list is None:
            continue

        for target in target_list.findall('target'):
            box = target.find('box')
            attr = target.find('attribute')

            if box is None or attr is None:
                continue

            x = float(box.get('left'))
            y = float(box.get('top'))
            w = float(box.get('width'))
            h = float(box.get('height'))

            label = map_detrac_label(attr.get('vehicle_type', 'others'))

            gt_data[frame_id].append({
                "bbox": (x, y, w, h),
                "label": label
            })

    return gt_data


def create_gt_label_grid(gt_boxes):
    grid = np.full((num_rows, num_cols), "Empty", dtype=object)

    for obj in gt_boxes:
        x, y, w, h = obj["bbox"]
        label = obj["label"]

        for r in range(num_rows):
            for c in range(num_cols):
                gx = roi1_x + c * grid_width1
                gy = roi1_y + r * grid_height1

                if not (x + w < gx or x > gx + grid_width1 or
                        y + h < gy or y > gy + grid_height1):
                    grid[r][c] = label

    return grid


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

            label_matrix = np.full((num_rows, num_cols), "Empty", dtype=object)
            for r in range(num_rows):
                for c in range(num_cols):
                    if result_matrix1[r, c] == 0:
                        continue

                    gx = roi1_x + c * grid_width1
                    gy = roi1_y + r * grid_height1

                    best_label = "Empty"
                    max_overlap = 0

                    for v in vechicles.values():
                        if v.get('bbox') is None:
                            continue

                        bx, by, bw, bh = v['bbox']
                        label = v.get('final_label', "Unknown")

                        overlap_x = max(0, min(bx + bw, gx + grid_width1) - max(bx, gx))
                        overlap_y = max(0, min(by + bh, gy + grid_height1) - max(by, gy))
                        overlap_area = overlap_x * overlap_y

                        if overlap_area > max_overlap:
                            max_overlap = overlap_area
                            best_label = label

                    if max_overlap > 0:
                        label_matrix[r][c] = best_label

            all_frames_data.append((frame_count, result_matrix1.copy(), label_matrix.copy()))

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

    # Evaluation runs only after processing has completed.
    XML_PATH = r"C:\Users\jaysu\OneDrive\Desktop\PROJECTS\BTP\Traffic_video_dataset\Detrac\Kaggledetracimgarchive\DETRAC-Train-Annotations-XML\DETRAC-Train-Annotations-XML\MVI_40191.xml" # CHANGE THIS

    if os.path.exists(XML_PATH):
        gt_data = load_detrac(XML_PATH)

        classes = ["Empty", "Car", "Van", "Bus"]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        conf_matrix = np.zeros((4, 4), dtype=int)

        for frame_id, _binary_grid, pred_grid in all_frames_data:
            gt_boxes = gt_data.get(frame_id, [])
            gt_grid = create_gt_label_grid(gt_boxes)

            for i in range(num_rows):
                for j in range(num_cols):
                    gt_label = gt_grid[i][j]
                    pred_label = pred_grid[i][j]

                    gt_idx = class_to_idx.get(gt_label, 0)
                    pred_idx = class_to_idx.get(pred_label, 0)

                    conf_matrix[gt_idx][pred_idx] += 1

        print("\n-- Multi-Class Grid Detection Metrics --")
        print("Confusion Matrix (rows=GT, cols=Pred):")
        print(conf_matrix)

        for i, cls in enumerate(classes):
            TP = conf_matrix[i][i]
            FP = conf_matrix[:, i].sum() - TP
            FN = conf_matrix[i, :].sum() - TP

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0)

            print(f"\nClass: {cls}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall   : {recall:.3f}")
            print(f"F1 Score : {f1:.3f}")

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes
        )
        plt.title("Multi-Class Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
    else:
        print(f"Skipping evaluation: XML file not found at {XML_PATH}")

def save_all_to_excel():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Grid Data"

    for frame_number, binary_matrix, label_matrix in all_frames_data:
        sheet.append([f"Frame {frame_number} - Binary"])
        for row in binary_matrix:
            sheet.append(list(row))
        sheet.append([])

        sheet.append([f"Frame {frame_number} - Label"])
        for row in label_matrix:
            sheet.append(list(row))
        sheet.append([])

    workbook.save(excel_file_path)

save_all_to_excel()
