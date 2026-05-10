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
xml_path = config.get("xml_path", "")

roi1_x, roi1_y, roi1_width, roi1_height = 47, 202, 441, 141
#roi1_x, roi1_y, roi1_width, roi1_height = 360, 250, 400, 200

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
    _, thresh = cv2.threshold(blur,60, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (grid_width1, grid_height1 // 2))
    dilated = cv2.dilate(thresh, kernel, iterations=5)
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
                valid_contour = [c for c in contours if cv2.contourArea(c) > 100]

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

def group_cells(cell_contours):
    """
    4-connected flood-fill grouping of hot grid cells.
    Returns a list of clusters, each cluster = list of (row, col) tuples.
    This replaces the old distance-based merge and gives accurate col_span/row_span.
    """
    grid = np.zeros((num_rows, num_cols), dtype=int)
    for (r, c, _) in cell_contours:
        grid[r][c] = 1

    visited = np.zeros_like(grid)
    clusters = []

    for r in range(num_rows):
        for c in range(num_cols):
            if grid[r][c] == 1 and visited[r][c] == 0:
                stack = [(r, c)]
                cluster = []
                while stack:
                    rr, cc = stack.pop()
                    if rr < 0 or rr >= num_rows or cc < 0 or cc >= num_cols:
                        continue
                    if visited[rr][cc] or grid[rr][cc] == 0:
                        continue
                    visited[rr][cc] = 1
                    cluster.append((rr, cc))
                    # Only allow horizontal OR vertical continuity, not zig-zag.
                    if (rr, cc) in cluster:
                        stack.extend([
                            (rr + 1, cc),
                            (rr - 1, cc),
                            (rr, cc + 1),
                            (rr, cc - 1)
                        ])
                clusters.append(cluster)

    return clusters


def split_cluster_by_columns(cluster):
    from collections import defaultdict

    col_groups = defaultdict(list)

    for r, c in cluster:
        col_groups[c].append((r, c))

    sorted_cols = sorted(col_groups.keys())

    clusters = []
    current_cluster = []

    prev_col = None
    for col in sorted_cols:
        if prev_col is None or col - prev_col <= 1:
            current_cluster.extend(col_groups[col])
        else:
            clusters.append(current_cluster)
            current_cluster = col_groups[col]

        prev_col = col

    if current_cluster:
        clusters.append(current_cluster)

    return clusters


def cluster_to_object(cluster, roi_x, roi_y):
    """Convert a cell cluster into a vehicle object dict with bbox, centroid, col/row span."""
    cols = [c for (_, c) in cluster]
    rows = [r for (r, _) in cluster]

    col_span = len(set(cols))
    row_span = len(set(rows))

    xs, ys = [], []
    for r, c in cluster:
        gx = roi_x + c * grid_width1
        gy = roi_y + r * grid_height1
        xs.extend([gx, gx + grid_width1])
        ys.extend([gy, gy + grid_height1])

    bx, by = min(xs), min(ys)
    bw = max(xs) - bx
    bh = max(ys) - by
    cx = bx + bw // 2
    cy = by + bh // 2

    return {
        "centroid": (cx, cy),
        "bbox": (bx, by, bw, bh),
        "col_span": col_span,
        "row_span": row_span,
        "cell_count": len(cluster),
    }

'''
def classify_by_grid(col_span, row_span):
    norm_col = col_span / num_cols
    norm_row = row_span / num_rows

    size_score = 0.6 * norm_col + 0.4 * norm_row

    if size_score > 0.60:
        return "Bus"
    elif size_score > 0.35:
        return "Van"
    else:
        return "Car"
'''

def classify_by_grid(col_span, row_span):
    norm_col = col_span / num_cols
    norm_row = row_span / num_rows

    # Bus: wide OR very large overall
    if norm_col > 0.75 or (norm_col > 0.55 and norm_row > 0.55):
        return "Bus"

    # Van: long but not too wide
    if norm_row > 0.45 :
        return "Van"

    return "Car"

def is_duplicate_crossing(cx, frame_count):
    """Return True if a nearby crossing was already counted recently."""
    for past_frame, past_cx in recent_crossings:
        if (frame_count - past_frame) < CROSS_COOLDOWN_FRAMES and abs(cx - past_cx) < CROSS_COOLDOWN_PX:
            return True
    return False


def prune_crossings(frame_count):
    """Keep only recent crossings within cooldown window."""
    recent_crossings[:] = [
        (f, cx) for (f, cx) in recent_crossings
        if (frame_count - f) < CROSS_COOLDOWN_FRAMES
    ]


def suppress_duplicate_tracks(frame_count):
    """Suppress duplicate active tracks that overlap heavily in the same frame."""
    active = {
        k: v for k, v in vechicles.items()
        if v.get('last_seen') == frame_count and v.get('bbox') is not None
    }
    keys = list(active.keys())
    suppressed = set()

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ka, kb = keys[i], keys[j]
            if ka in suppressed or kb in suppressed:
                continue

            ax, ay, aw, ah = active[ka]['bbox']
            bx, by, bw, bh = active[kb]['bbox']

            ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
            iy = max(0, min(ay + ah, by + bh) - max(ay, by))
            inter = ix * iy
            union = aw * ah + bw * bh - inter
            iou = inter / union if union > 0 else 0

            if iou > 0.40:
                age_a = active[ka].get('age', 0)
                age_b = active[kb].get('age', 0)
                if age_a > age_b:
                    loser = kb
                elif age_b > age_a:
                    loser = ka
                else:
                    loser = max(ka, kb)

                suppressed.add(loser)
                vechicles[loser]['counted'] = True
                vechicles[loser]['last_seen'] = -1

    return suppressed


def process_tracking_counting(frame1, roi_x, roi_y, frame_count, cell_contours):
    global vechicle_count, vehicle_count_history, next_vehicle_id

    keys_to_remove = []
    MATCH_THRESHOLD = roi1_height // 2

    cv2.putText(frame1, f"Vehicles Passed: {vechicle_count}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    prune_crossings(frame_count)

    if not cell_contours:
        vehicle_count_history.append(0)
        for key in list(vechicles.keys()):
            if frame_count - vechicles[key]['last_seen'] > FPS * 2:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del vechicles[key]
        return

    # ── STEP 1: group hot cells into spatially-connected clusters ──────────────
    clusters = group_cells(cell_contours)

    # Split each connected component by column continuity to avoid lane merges.
    new_clusters = []
    for cl in clusters:
        split = split_cluster_by_columns(cl)
        new_clusters.extend(split)
    clusters = new_clusters

    # ── STEP 2: convert each cluster → object dict ─────────────────────────────
    objects = [cluster_to_object(cl, roi_x, roi_y) for cl in clusters]

    # ── STEP 3: match each object to an existing track (nearest centroid) ──────
    matched_track_ids = set()

    for obj in objects:
        cx, cy = obj["centroid"]
        best_match = None
        min_dist = MATCH_THRESHOLD

        for vid, v in vechicles.items():
            if vid in matched_track_ids:
                continue
            px, py = v["centroid"]
            d = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            if d < min_dist:
                min_dist = d
                best_match = vid

        if best_match is not None:
            v = vechicles[best_match]
            v["prev_centroid"] = v["centroid"]
            v["centroid"]  = obj["centroid"]
            v["bbox"]      = obj["bbox"]
            v["col_span"]  = obj["col_span"]
            v["row_span"]  = obj["row_span"]
            v["cell_count"] = obj["cell_count"]
            v["last_seen"] = frame_count
            v["age"]       = v.get("age", 0) + 1
            matched_track_ids.add(best_match)
        else:
            vechicles[next_vehicle_id] = {
                "centroid":      obj["centroid"],
                "prev_centroid": obj["centroid"],
                "bbox":          obj["bbox"],
                "col_span":      obj["col_span"],
                "row_span":      obj["row_span"],
                "cell_count":    obj["cell_count"],
                "last_seen":     frame_count,
                "age":           1,
                "counted":       False,
                "frame_age":     0,
                "recent_labels": [],
                "label_votes":   Counter(),
                "weighted_votes": {"Car": 0.0, "Van": 0.0, "Bus": 0.0},
                "final_label":   "Unknown",
                "display_label": "Unknown",
            }
            matched_track_ids.add(next_vehicle_id)
            next_vehicle_id += 1

    # ── STEP 4: classify each active track ─────────────────────────────────────
    for key, v in vechicles.items():
        if v["last_seen"] != frame_count:
            continue

        v["frame_age"] = v.get("frame_age", 0) + 1
        raw_label = classify_by_grid(v["col_span"], v["row_span"])

        # Combine vertical position confidence with temporal exponential weighting.
        wv = v.setdefault("weighted_votes", {"Car": 0.0, "Van": 0.0, "Bus": 0.0})
        cy = v["centroid"][1]
        pos_weight = cy / max(1, frame1.shape[0])
        time_weight = (1 + LABEL_EWA_ALPHA) ** v["frame_age"]
        weight = pos_weight * time_weight
        wv[raw_label] = wv.get(raw_label, 0.0) + weight
        v["final_label"] = max(wv, key=wv.get)

        labels = v.setdefault("recent_labels", [])
        labels.append(raw_label)
        if len(labels) > 10:
            labels.pop(0)
        v["display_label"] = max(set(labels), key=labels.count)

        # draw bbox + label
        if v.get("bbox") is not None:
            bx, by, bw, bh = v["bbox"]
            status = v["display_label"]
            cv2.rectangle(frame1, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
            label_y = max(by - 5 - (key % 3) * 16, 12)
            cv2.putText(
                frame1,
                f"ID:{key} | {status} | cols:{v['col_span']}",
                (bx, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,
            )

    suppress_duplicate_tracks(frame_count)

    # ── STEP 5: count vehicles crossing the line ────────────────────────────────
    temp_count = 0
    for key in list(vechicles.keys()):
        value = vechicles[key]
        if (not value["counted"]
                and value["age"] >= 2
                and value["prev_centroid"][1] < COUNT_LINE_Y
                and value["centroid"][1] >= COUNT_LINE_Y):

            cx = value["centroid"][0]
            if is_duplicate_crossing(cx, frame_count):
                vechicles[key]["counted"] = True
                continue

            vechicle_count += 1
            temp_count += 1
            vechicles[key]["counted"] = True
            recent_crossings.append((frame_count, cx))
            vechicles[key]["last_seen"] = frame_count

    vehicle_count_history.append(temp_count)

    # ── cleanup stale tracks ────────────────────────────────────────────────────
    for key in list(vechicles.keys()):
        if frame_count - vechicles[key]["last_seen"] > FPS * 2:
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
CROSS_COOLDOWN_FRAMES = 30
CROSS_COOLDOWN_PX = 60
LABEL_EWA_ALPHA = 0.15
MIN_FRAMES_FOR_LABEL = 5
recent_crossings = []
all_frames_data = []
all_frames_vehicle_data = []

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
            obj_id = int(target.get('id', -1))

            gt_data[frame_id].append({
                "id": obj_id,
                "bbox": (x, y, w, h),
                "label": label
            })

    return gt_data


def create_gt_label_grid(gt_boxes):
    grid = np.full((num_rows, num_cols), "Empty", dtype=object)

    for obj in gt_boxes:
        x, y, w, h = obj["bbox"]
        label = obj["label"]

        # ignore unknown classes
        if label not in ["Car", "Van", "Bus"]:
            continue

        # 👉 center of bounding box
        cx = x + w / 2
        cy = y + h / 2

        # 👉 map center to grid cell
        col = int((cx - roi1_x) / grid_width1)
        row = int((cy - roi1_y) / grid_height1)

        # check bounds
        if 0 <= row < num_rows and 0 <= col < num_cols:
            grid[row][col] = label

    return grid


def match_vehicle(pred_bbox, gt_boxes):
    best_id = None
    max_iou = 0

    px, py, pw, ph = pred_bbox

    for obj in gt_boxes:
        gx, gy, gw, gh = obj["bbox"]

        ix = max(0, min(px + pw, gx + gw) - max(px, gx))
        iy = max(0, min(py + ph, gy + gh) - max(py, gy))
        inter = ix * iy

        union = pw * ph + gw * gh - inter
        iou = inter / union if union > 0 else 0

        if iou > max_iou:
            max_iou = iou
            best_id = obj.get("id")

    return best_id if max_iou > 0.1 else None


def compute_vehicle_metrics(cm):
    metrics = {}
    classes = ["Car", "Van", "Bus"]

    for i, cls in enumerate(classes):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        metrics[cls] = (precision, recall, f1)

    return metrics


def filter_gt_boxes_to_roi(gt_boxes):
    return [
        obj for obj in gt_boxes
        if not (
            obj["bbox"][0] + obj["bbox"][2] < roi1_x or
            obj["bbox"][0] > roi1_x + roi1_width or
            obj["bbox"][1] + obj["bbox"][3] < roi1_y or
            obj["bbox"][1] > roi1_y + roi1_height
        )
    ]


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

            frame_vehicle_preds = {}
            for vid, v in vechicles.items():
                if v.get('last_seen') != frame_count or v.get('bbox') is None:
                    continue
                pred_label = v.get('final_label') if v.get('frame_age', 0) >= MIN_FRAMES_FOR_LABEL else None
                if pred_label not in ["Car", "Van", "Bus"]:
                    pred_label = None
                frame_vehicle_preds[vid] = {
                    "bbox": v['bbox'],
                    "label": pred_label
                }
            all_frames_vehicle_data.append((frame_count, frame_vehicle_preds))

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
    XML_PATH = xml_path.strip()
    if XML_PATH and not os.path.isabs(XML_PATH):
        XML_PATH = os.path.abspath(XML_PATH)

    if XML_PATH and os.path.exists(XML_PATH):
        gt_data = load_detrac(XML_PATH)
        DEBUG_GRID_METRICS = False
        SHOW_GRID_HEATMAP = True
        conf_matrix = None

        # Vehicle counts must be ID-based (grid-independent).
        unique_gt_ids = set()
        for _frame_id, objs in gt_data.items():
            roi_gt_boxes = filter_gt_boxes_to_roi(objs)
            for obj in roi_gt_boxes:
                obj_id = obj.get("id", -1)
                if obj_id >= 0:
                    unique_gt_ids.add(obj_id)

        unique_pred_track_ids = set()
        for _frame_id, pred_vehicles in all_frames_vehicle_data:
            unique_pred_track_ids.update(pred_vehicles.keys())

        print("\n-- Vehicle Count Summary (Grid-Independent) --")
        print(f"Actual vehicles (unique GT IDs): {len(unique_gt_ids)}")
        print(f"Predicted vehicles (unique tracker IDs): {len(unique_pred_track_ids)}")

        gt_vehicle_labels = {}
        for frame_id, objs in gt_data.items():
            roi_gt_boxes = filter_gt_boxes_to_roi(objs)
            for obj in roi_gt_boxes:
                vid = obj.get("id", -1)
                label = obj.get("label", "Car")
                if label not in ["Car", "Van", "Bus"]:
                    continue
                gt_vehicle_labels[vid] = label

        if DEBUG_GRID_METRICS or SHOW_GRID_HEATMAP:
            classes = ["Empty", "Car", "Van", "Bus"]
            class_to_idx = {c: i for i, c in enumerate(classes)}
            conf_matrix = np.zeros((4, 4), dtype=int)

            for frame_id, _binary_grid, pred_grid in all_frames_data:
                gt_boxes = gt_data.get(frame_id + 1, [])
                gt_boxes = filter_gt_boxes_to_roi(gt_boxes)
                gt_grid = create_gt_label_grid(gt_boxes)

                for i in range(num_rows):
                    for j in range(num_cols):
                        gt_label = gt_grid[i][j]
                        pred_label = pred_grid[i][j]

                        gt_idx = class_to_idx.get(gt_label, 0)
                        pred_idx = class_to_idx.get(pred_label, 0)

                        conf_matrix[gt_idx][pred_idx] += 1


        # 2) Vehicle-level evaluation (bbox vs XML only; not grid-based)
        vehicle_classes = ["Car", "Van", "Bus"]
        vehicle_idx = {c: i for i, c in enumerate(vehicle_classes)}
        conf_matrix_vehicle = np.zeros((3, 3), dtype=int)
        pred_vehicle_labels = {}
        matched_pred_track_ids = set()

        for frame_id, pred_vehicles in all_frames_vehicle_data:
            gt_boxes = gt_data.get(frame_id + 1, [])
            gt_boxes = filter_gt_boxes_to_roi(gt_boxes)
            # print(f"Frame {frame_id} -> GT objects: {len(gt_boxes)}")

            for vid, pred_obj in pred_vehicles.items():
                pred_label = pred_obj.get("label")
                if pred_label is None or pred_label not in vehicle_idx:
                    continue

                gt_id = match_vehicle(pred_obj["bbox"], gt_boxes)
                if gt_id is None:
                    continue

                pred_vehicle_labels.setdefault(gt_id, []).append(pred_label)
                matched_pred_track_ids.add(vid)

        for gt_id, pred_list in pred_vehicle_labels.items():
            pred_label = Counter(pred_list).most_common(1)[0][0]
            gt_label = gt_vehicle_labels.get(gt_id, "Car")
            if gt_label not in vehicle_idx:
                gt_label = "Car"

            i = vehicle_idx[gt_label]
            j = vehicle_idx[pred_label]
            conf_matrix_vehicle[i][j] += 1

        matched_gt_ids = set(pred_vehicle_labels.keys())
        unmatched_gt_ids = unique_gt_ids - matched_gt_ids
        unmatched_pred_track_ids = unique_pred_track_ids - matched_pred_track_ids

        print("\n-- Vehicle Matching Summary (Unique IDs) --")
        print(f"Matched GT vehicles: {len(matched_gt_ids)}")
        print(f"Unmatched GT vehicles: {len(unmatched_gt_ids)}")
        print(f"Matched predicted tracks: {len(matched_pred_track_ids)}")
        print(f"Unmatched predicted tracks: {len(unmatched_pred_track_ids)}")

        print("\n-- DEBUG INFO --")
        print(f"Total GT vehicles: {len(unique_gt_ids)}")
        print(f"Total predicted vehicles: {len(unique_pred_track_ids)}")
        print(f"Matched GT vehicles: {len(matched_gt_ids)}")
        print(f"Unmatched GT vehicles: {len(unmatched_gt_ids)}")
        print(f"Unmatched predicted vehicles: {len(unmatched_pred_track_ids)}")
        if DEBUG_GRID_METRICS and conf_matrix is not None:
            print("\n-- DEBUG: Grid Confusion Matrix (Empty-grid heatmap) --")
            print(conf_matrix)

        print("\n-- Vehicle-Level Confusion Matrix --")
        print(conf_matrix_vehicle)

        metrics = compute_vehicle_metrics(conf_matrix_vehicle)
        print("\n-- Vehicle-Level Metrics --")
        for cls in ["Car", "Van", "Bus"]:
            precision, recall, f1 = metrics[cls]
            print(f"\nClass: {cls}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall   : {recall:.3f}")
            print(f"F1 Score : {f1:.3f}")

        if SHOW_GRID_HEATMAP and conf_matrix is not None:
            plt.figure(1, figsize=(8, 6))
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

        classes_vehicle = ["Car", "Van", "Bus"]

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            conf_matrix_vehicle,
            annot=True,
            fmt="d",
            cmap="viridis",
            xticklabels=classes_vehicle,
            yticklabels=classes_vehicle,
            ax=ax,
        )
        ax.set_title("Vehicle-Level Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        fig.subplots_adjust(bottom=0.28)
        metrics_text = "\n".join([
            f"{cls}: P={metrics[cls][0]:.3f}  R={metrics[cls][1]:.3f}  F1={metrics[cls][2]:.3f}"
            for cls in classes_vehicle
        ])
        fig.text(0.5, 0.02, metrics_text, ha="center", va="bottom", fontsize=9)

        plt.show()
    elif XML_PATH:
        print(f"Skipping evaluation: XML file not found at {XML_PATH}")
    else:
        print("Skipping evaluation: XML path not provided in user_input_data.json")

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
