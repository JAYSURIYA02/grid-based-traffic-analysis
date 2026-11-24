import os
import xml.etree.ElementTree as ET


def load_detrac_annotations(xml_path):
    """Load DETRAC XML annotations.

    Returns:
        dict[int, list[dict]]: frame_id -> list of {"bbox": [x, y, w, h], "type": str}
    """
    if not xml_path or not os.path.exists(xml_path):
        return {}

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return {}

    root = tree.getroot()
    frames = {}

    for frame in root.findall("frame"):
        frame_id = frame.get("num")
        if frame_id is None:
            continue
        try:
            frame_id = int(frame_id)
        except ValueError:
            continue

        frames.setdefault(frame_id, [])
        target_list = frame.find("target_list")
        if target_list is None:
            continue

        for target in target_list.findall("target"):
            box = target.find("box")
            attr = target.find("attribute")
            if box is None or attr is None:
                continue

            try:
                x = float(box.get("left"))
                y = float(box.get("top"))
                w = float(box.get("width"))
                h = float(box.get("height"))
            except (TypeError, ValueError):
                continue

            vehicle_type = (attr.get("vehicle_type") or "").lower()
            try:
                target_id = int(target.get("id", -1))
            except (TypeError, ValueError):
                target_id = -1
            frames[frame_id].append({
                "bbox": [x, y, w, h],
                "type": vehicle_type,
                "id": target_id,
            })

    return frames
