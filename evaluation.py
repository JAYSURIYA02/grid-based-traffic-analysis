import cv2
import numpy as np
from collections import Counter

from annotation_parser import load_detrac_annotations
from utils import CLASSES, compute_metrics

CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}


def map_detrac_label(vehicle_type):
    label = (vehicle_type or "").lower()
    if label == "car":
        return "Car"
    if label == "van":
        return "Van"
    if label == "bus":
        return "Bus"
    return None


def filter_gt_boxes_to_roi(gt_boxes, roi_x, roi_y, roi_w, roi_h):
    filtered = []
    for obj in gt_boxes:
        x, y, w, h = obj["bbox"]
        if (x + w < roi_x or x > roi_x + roi_w or
                y + h < roi_y or y > roi_y + roi_h):
            continue
        filtered.append(obj)
    return filtered


def create_gt_label_grid(gt_boxes, rows, cols, roi_x, roi_y, grid_w, grid_h):
    grid = np.full((rows, cols), "Empty", dtype=object)

    for obj in gt_boxes:
        label = map_detrac_label(obj.get("type"))
        if label not in ("Car", "Van", "Bus"):
            continue

        x, y, w, h = obj["bbox"]
        for r in range(rows):
            for c in range(cols):
                gx = roi_x + c * grid_w
                gy = roi_y + r * grid_h

                if not (x + w < gx or x > gx + grid_w or
                        y + h < gy or y > gy + grid_h):
                    grid[r][c] = label

    return grid


def match_vehicle(pred_bbox, gt_boxes):
    best_id = None
    max_iou = 0.0

    px, py, pw, ph = pred_bbox
    for obj in gt_boxes:
        gx, gy, gw, gh = obj["bbox"]
        gt_id = obj.get("id", -1)
        if gt_id < 0:
            continue

        ix = max(0, min(px + pw, gx + gw) - max(px, gx))
        iy = max(0, min(py + ph, gy + gh) - max(py, gy))
        inter = ix * iy
        union = pw * ph + gw * gh - inter
        iou = inter / union if union > 0 else 0.0

        if iou > max_iou:
            max_iou = iou
            best_id = gt_id

    return best_id if max_iou > 0.1 else None


class GridDetector:
    def __init__(self, rows, cols, roi_x=47, roi_y=202, roi_w=441, roi_h=141,
                 color_channel="V", he_choice="V", fps=30, min_frames_for_label=5):
        self.rows = rows
        self.cols = cols
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_w = roi_w
        self.roi_h = roi_h
        self.grid_w = max(1, roi_w // cols)
        self.grid_h = max(1, roi_h // rows)
        self.channels = self._resolve_channels(color_channel)
        self.he_choice = he_choice
        self.fps = fps
        self.min_frames_for_label = min_frames_for_label

        self.vehicles = {}
        self.next_vehicle_id = 0
        self.label_ewa_alpha = 0.15

    def _resolve_channels(self, color_channel):
        choices = {
            "H": [0],
            "S": [1],
            "V": [2],
            "H+S": [0, 1],
            "H+V": [0, 2],
            "S+V": [1, 2],
            "H+S+V": [0, 1, 2],
            "gray": "gray",
        }
        return choices.get(color_channel, [2])

    def process_grid_channel(self, channel):
        blur = cv2.GaussianBlur(channel, (5, 5), 1)
        _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.grid_w, max(1, self.grid_h // 2))
        )
        dilated = cv2.dilate(thresh, kernel, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def apply_histogram_equalization(self, frame, channel_choice):
        frame = frame.copy()
        roi = frame[self.roi_y:self.roi_y + self.roi_h, self.roi_x:self.roi_x + self.roi_w]
        if roi.size == 0:
            return frame

        if channel_choice == "gray":
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_eq = cv2.equalizeHist(roi_gray)
            frame[self.roi_y:self.roi_y + self.roi_h,
                  self.roi_x:self.roi_x + self.roi_w] = cv2.cvtColor(roi_eq, cv2.COLOR_GRAY2BGR)
            return frame

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if channel_choice == "H":
            h = cv2.equalizeHist(h)
        elif channel_choice == "S":
            s = cv2.equalizeHist(s)
        elif channel_choice == "V":
            v = cv2.equalizeHist(v)

        hsv_eq = cv2.merge([h, s, v])
        frame[self.roi_y:self.roi_y + self.roi_h,
              self.roi_x:self.roi_x + self.roi_w] = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
        return frame

    def process_hsv(self, frame1, frame2, channels, channel_choice):
        frame1 = self.apply_histogram_equalization(frame1, channel_choice)
        frame2 = self.apply_histogram_equalization(frame2, channel_choice)
        frame1_blur = cv2.GaussianBlur(frame1, (7, 7), 0)
        frame2_blur = cv2.GaussianBlur(frame2, (7, 7), 0)
        diff = cv2.absdiff(frame1_blur, frame2_blur)
        hsv = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)
        return [cv2.split(hsv)[i] for i in channels]

    def process_grayscale(self, frame1, frame2, channel_choice):
        frame1 = self.apply_histogram_equalization(frame1, channel_choice)
        frame2 = self.apply_histogram_equalization(frame2, channel_choice)
        frame1_blur = cv2.GaussianBlur(frame1, (7, 7), 0)
        frame2_blur = cv2.GaussianBlur(frame2, (7, 7), 0)
        diff = cv2.absdiff(frame1_blur, frame2_blur)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        return [gray]

    def compute_channels(self, frame1, frame2):
        if self.channels == "gray":
            return self.process_grayscale(frame1, frame2, self.he_choice)
        return self.process_hsv(frame1, frame2, self.channels, self.he_choice)

    def process_grid(self, frame, channels_data):
        result_matrix = np.zeros((self.rows, self.cols), dtype=int)
        cell_contours = []
        frame_h, frame_w = frame.shape[:2]

        for row in range(self.rows):
            for col in range(self.cols):
                grid_x = self.roi_x + col * self.grid_w
                grid_y = self.roi_y + row * self.grid_h

                if (grid_x < 0 or grid_y < 0 or
                        grid_x + self.grid_w > frame_w or
                        grid_y + self.grid_h > frame_h):
                    continue

                detection_flags = []
                cell_contours_local = []
                for channel in channels_data:
                    grid_channel = channel[grid_y:grid_y + self.grid_h,
                                           grid_x:grid_x + self.grid_w]
                    if grid_channel.size == 0:
                        continue
                    contours = self.process_grid_channel(grid_channel)
                    valid = [c for c in contours if cv2.contourArea(c) > 100]
                    detection_flags.append(len(valid) > 0)
                    cell_contours_local.extend(valid)

                if detection_flags and all(detection_flags):
                    result_matrix[row, col] = 1
                    cell_contours.append((row, col, cell_contours_local))

        return result_matrix, cell_contours

    def group_cells(self, cell_contours):
        grid = np.zeros((self.rows, self.cols), dtype=int)
        for r, c, _ in cell_contours:
            grid[r][c] = 1

        visited = np.zeros_like(grid)
        clusters = []

        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r][c] == 1 and visited[r][c] == 0:
                    stack = [(r, c)]
                    cluster = []
                    while stack:
                        rr, cc = stack.pop()
                        if rr < 0 or rr >= self.rows or cc < 0 or cc >= self.cols:
                            continue
                        if visited[rr][cc] or grid[rr][cc] == 0:
                            continue
                        visited[rr][cc] = 1
                        cluster.append((rr, cc))
                        stack.extend([
                            (rr + 1, cc),
                            (rr - 1, cc),
                            (rr, cc + 1),
                            (rr, cc - 1),
                        ])
                    clusters.append(cluster)

        return clusters

    def split_cluster_by_columns(self, cluster):
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
                current_cluster = list(col_groups[col])
            prev_col = col

        if current_cluster:
            clusters.append(current_cluster)

        return clusters

    def cluster_to_object(self, cluster):
        cols = [c for (_, c) in cluster]
        rows = [r for (r, _) in cluster]
        col_span = len(set(cols))
        row_span = len(set(rows))

        xs, ys = [], []
        for r, c in cluster:
            gx = self.roi_x + c * self.grid_w
            gy = self.roi_y + r * self.grid_h
            xs.extend([gx, gx + self.grid_w])
            ys.extend([gy, gy + self.grid_h])

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

    def classify_by_grid(self, col_span, row_span):
        norm_col = col_span / self.cols
        norm_row = row_span / self.rows

        if norm_col > 0.75 or (norm_col > 0.55 and norm_row > 0.55):
            return "Bus"
        if norm_row > 0.45:
            return "Van"
        return "Car"

    def suppress_duplicate_tracks(self, frame_id):
        active = {
            k: v for k, v in self.vehicles.items()
            if v.get("last_seen") == frame_id and v.get("bbox") is not None
        }
        keys = list(active.keys())
        suppressed = set()

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ka, kb = keys[i], keys[j]
                if ka in suppressed or kb in suppressed:
                    continue

                ax, ay, aw, ah = active[ka]["bbox"]
                bx, by, bw, bh = active[kb]["bbox"]

                ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
                iy = max(0, min(ay + ah, by + bh) - max(ay, by))
                inter = ix * iy
                union = aw * ah + bw * bh - inter
                iou = inter / union if union > 0 else 0

                if iou > 0.40:
                    age_a = active[ka].get("age", 0)
                    age_b = active[kb].get("age", 0)
                    if age_a > age_b:
                        loser = kb
                    elif age_b > age_a:
                        loser = ka
                    else:
                        loser = max(ka, kb)

                    suppressed.add(loser)
                    self.vehicles[loser]["last_seen"] = -1

        return suppressed

    def _prune_tracks(self, frame_id):
        keys_to_remove = []
        for key, value in self.vehicles.items():
            last_seen = value.get("last_seen", -1)
            if last_seen < 0 or frame_id - last_seen > self.fps * 2:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.vehicles[key]

    def update_tracks(self, frame_shape, frame_id, cell_contours):
        if not cell_contours:
            self._prune_tracks(frame_id)
            return

        clusters = self.group_cells(cell_contours)
        new_clusters = []
        for cl in clusters:
            new_clusters.extend(self.split_cluster_by_columns(cl))
        clusters = new_clusters

        objects = [self.cluster_to_object(cl) for cl in clusters]
        matched_track_ids = set()
        match_threshold = self.roi_h // 2

        for obj in objects:
            cx, cy = obj["centroid"]
            best_match = None
            min_dist = match_threshold

            for vid, v in self.vehicles.items():
                if vid in matched_track_ids:
                    continue
                px, py = v["centroid"]
                dist = np.hypot(cx - px, cy - py)
                if dist < min_dist:
                    min_dist = dist
                    best_match = vid

            if best_match is not None:
                v = self.vehicles[best_match]
                v["prev_centroid"] = v["centroid"]
                v["centroid"] = obj["centroid"]
                v["bbox"] = obj["bbox"]
                v["col_span"] = obj["col_span"]
                v["row_span"] = obj["row_span"]
                v["cell_count"] = obj["cell_count"]
                v["last_seen"] = frame_id
                v["age"] = v.get("age", 0) + 1
                matched_track_ids.add(best_match)
            else:
                self.vehicles[self.next_vehicle_id] = {
                    "centroid": obj["centroid"],
                    "prev_centroid": obj["centroid"],
                    "bbox": obj["bbox"],
                    "col_span": obj["col_span"],
                    "row_span": obj["row_span"],
                    "cell_count": obj["cell_count"],
                    "last_seen": frame_id,
                    "age": 1,
                    "frame_age": 0,
                    "recent_labels": [],
                    "weighted_votes": {"Car": 0.0, "Van": 0.0, "Bus": 0.0},
                    "final_label": "Unknown",
                    "display_label": "Unknown",
                }
                matched_track_ids.add(self.next_vehicle_id)
                self.next_vehicle_id += 1

        frame_h = frame_shape[0]
        for key, v in self.vehicles.items():
            if v.get("last_seen") != frame_id:
                continue

            v["frame_age"] = v.get("frame_age", 0) + 1
            raw_label = self.classify_by_grid(v["col_span"], v["row_span"])

            wv = v.setdefault("weighted_votes", {"Car": 0.0, "Van": 0.0, "Bus": 0.0})
            pos_weight = v["centroid"][1] / max(1, frame_h)
            time_weight = (1 + self.label_ewa_alpha) ** v["frame_age"]
            weight = pos_weight * time_weight
            wv[raw_label] = wv.get(raw_label, 0.0) + weight
            v["final_label"] = max(wv, key=wv.get)

            labels = v.setdefault("recent_labels", [])
            labels.append(raw_label)
            if len(labels) > 10:
                labels.pop(0)
            v["display_label"] = max(set(labels), key=labels.count)

        self.suppress_duplicate_tracks(frame_id)
        self._prune_tracks(frame_id)

    def build_label_matrix(self, result_matrix, frame_id):
        label_matrix = np.full((self.rows, self.cols), "Empty", dtype=object)

        for r in range(self.rows):
            for c in range(self.cols):
                if result_matrix[r, c] == 0:
                    continue

                gx = self.roi_x + c * self.grid_w
                gy = self.roi_y + r * self.grid_h

                best_label = "Empty"
                max_overlap = 0
                for v in self.vehicles.values():
                    if v.get("last_seen") != frame_id:
                        continue
                    if v.get("bbox") is None:
                        continue

                    label = v.get("final_label", "Unknown")
                    if label not in ("Car", "Van", "Bus"):
                        continue

                    bx, by, bw, bh = v["bbox"]
                    overlap_x = max(0, min(bx + bw, gx + self.grid_w) - max(bx, gx))
                    overlap_y = max(0, min(by + bh, gy + self.grid_h) - max(by, gy))
                    overlap_area = overlap_x * overlap_y

                    if overlap_area > max_overlap:
                        max_overlap = overlap_area
                        best_label = label

                if max_overlap > 0:
                    label_matrix[r][c] = best_label

        return label_matrix


def process_video(video_path, xml_path, rows, cols):
    """Run grid-based detection and evaluation for a single video."""
    gt_data = load_detrac_annotations(xml_path)
    detector = GridDetector(rows, cols)

    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    ret2, frame2 = cap.read()

    conf_matrix = np.zeros((4, 4), dtype=int)
    frame_id = 1

    gt_vehicle_labels = {}
    pred_vehicle_labels = {}
    matched_pred_track_ids = set()
    unique_pred_track_ids = set()

    while ret and ret2:
        if frame1.shape[:2] != frame2.shape[:2]:
            frame1 = frame2
            ret2, frame2 = cap.read()
            frame_id += 1
            continue

        channels_data = detector.compute_channels(frame1, frame2)
        result_matrix, cell_contours = detector.process_grid(frame1, channels_data)
        detector.update_tracks(frame1.shape, frame_id, cell_contours)
        pred_grid = detector.build_label_matrix(result_matrix, frame_id)

        gt_boxes = gt_data.get(frame_id + 1, [])
        gt_boxes = filter_gt_boxes_to_roi(gt_boxes, detector.roi_x, detector.roi_y,
                                          detector.roi_w, detector.roi_h)

        for obj in gt_boxes:
            gt_id = obj.get("id", -1)
            label = map_detrac_label(obj.get("type"))
            if gt_id >= 0 and label in ("Car", "Van", "Bus"):
                gt_vehicle_labels[gt_id] = label
        gt_grid = create_gt_label_grid(gt_boxes, rows, cols,
                                       detector.roi_x, detector.roi_y,
                                       detector.grid_w, detector.grid_h)

        for i in range(rows):
            for j in range(cols):
                gt_label = gt_grid[i][j]
                pred_label = pred_grid[i][j]
                gt_idx = CLASS_TO_IDX.get(gt_label, 0)
                pred_idx = CLASS_TO_IDX.get(pred_label, 0)
                conf_matrix[gt_idx][pred_idx] += 1

        frame_vehicle_preds = {}
        for vid, v in detector.vehicles.items():
            if v.get("last_seen") != frame_id:
                continue
            if v.get("bbox") is None:
                continue

            unique_pred_track_ids.add(vid)
            if v.get("frame_age", 0) < detector.min_frames_for_label:
                continue

            pred_label = v.get("final_label")
            if pred_label not in ("Car", "Van", "Bus"):
                continue

            frame_vehicle_preds[vid] = {
                "bbox": v["bbox"],
                "label": pred_label,
            }

        for vid, pred_obj in frame_vehicle_preds.items():
            gt_id = match_vehicle(pred_obj["bbox"], gt_boxes)
            if gt_id is None:
                continue
            pred_vehicle_labels.setdefault(gt_id, []).append(pred_obj["label"])
            matched_pred_track_ids.add(vid)

        frame1 = frame2
        ret2, frame2 = cap.read()
        frame_id += 1

    cap.release()

    precision, recall, f1 = compute_metrics(conf_matrix)

    vehicle_classes = ["Car", "Van", "Bus"]
    vehicle_idx = {c: i for i, c in enumerate(vehicle_classes)}
    conf_matrix_vehicle = np.zeros((3, 3), dtype=int)

    for gt_id, pred_list in pred_vehicle_labels.items():
        if not pred_list:
            continue
        pred_label = Counter(pred_list).most_common(1)[0][0]
        gt_label = gt_vehicle_labels.get(gt_id)
        if gt_label not in vehicle_idx:
            continue
        i = vehicle_idx[gt_label]
        j = vehicle_idx[pred_label]
        conf_matrix_vehicle[i][j] += 1

    return {
        "conf_matrix": conf_matrix,
        "vehicle_conf_matrix": conf_matrix_vehicle,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "frame_count": frame_id - 1,
        "pred_tracks": len(unique_pred_track_ids),
        "matched_pred_tracks": len(matched_pred_track_ids),
    }
