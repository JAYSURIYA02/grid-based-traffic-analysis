"""
Calibrate PIXELS_PER_METER by clicking two points on a RAW video frame
(the same pixel coordinate system seq.py actually processes).

Usage:
    python calibrate_pixels_per_meter.py <video_path> [frame_number] [json_path]

    video_path  - path to the video file
    frame_number - which frame to display (default: 0)
    json_path   - path to user_input_data.json (default: user_input_data.json in cwd)

Workflow:
    1. The script opens the specified frame and displays it.
    2. Click TWO points spanning a known real-world distance
       (lane width, dash+gap, car width — whatever you're confident about).
    3. Press 'c' to confirm the two points — you'll be prompted in the
       terminal to enter the real-world distance in meters.
    4. Press 'n' to add another sample (click two new points).
    5. Press 'r' to reset the current (unconfirmed) pair and re-click.
    6. Press 'q' to finish — the script averages all confirmed samples,
       warns if they disagree by >15%, and saves pixels_per_meter into
       the JSON config file.
"""
import sys
import os
import json
import numpy as np
import cv2


# ── Global state for mouse callback ──────────────────────────────────────────
points = []
frame_clean = None   # pristine copy of the frame (no drawings)
frame_display = None  # the copy we draw on


def click_event(event, x, y, flags, param):
    """Mouse callback: collect up to 2 points, draw circles + connecting line."""
    global points, frame_display
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))
        cv2.circle(frame_display, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(frame_display, f"P{len(points)} ({x},{y})",
                    (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)
        if len(points) == 2:
            cv2.line(frame_display, points[0], points[1], (0, 255, 0), 2)
            # Show pixel distance on the frame
            px_dist = np.hypot(points[1][0] - points[0][0],
                               points[1][1] - points[0][1])
            mid = ((points[0][0] + points[1][0]) // 2,
                   (points[0][1] + points[1][1]) // 2)
            cv2.putText(frame_display, f"{px_dist:.1f} px",
                        (mid[0] + 5, mid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
                        cv2.LINE_AA)
            print(f"\n  Two points placed — pixel distance: {px_dist:.1f} px")
            print("  Press 'c' to confirm & enter real-world distance,")
            print("        'r' to reset and re-click.")
        cv2.imshow("Calibration Frame", frame_display)


def reset_frame():
    """Clear current points and redraw the clean frame."""
    global points, frame_display
    points.clear()
    frame_display = frame_clean.copy()
    cv2.imshow("Calibration Frame", frame_display)
    print("\n  Points reset — click two new points.")


def main():
    global frame_clean, frame_display

    # ── Parse arguments ──────────────────────────────────────────────────────
    if len(sys.argv) < 2:
        print("Usage: python calibrate_pixels_per_meter.py "
              "<video_path> [frame_number] [json_path]")
        return

    video_path = sys.argv[1]
    frame_number = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    json_path = sys.argv[3] if len(sys.argv) > 3 else "user_input_data.json"

    # ── Read the video frame ─────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"ERROR: Could not read frame {frame_number} from video.")
        return

    frame_clean = frame.copy()
    frame_display = frame.copy()

    print(f"\nVideo: {video_path}")
    print(f"Frame: {frame_number}")
    print(f"Native frame size: {frame.shape[1]} x {frame.shape[0]}")
    print(f"JSON config: {json_path}")
    print("\n" + "=" * 60)
    print("CALIBRATION — click TWO points spanning a known distance")
    print("=" * 60)
    print("\nControls:")
    print("  Click  — place a point (2 points per sample)")
    print("  'c'    — confirm current pair & enter real-world distance")
    print("  'n'    — start a new sample (after confirming the current one)")
    print("  'r'    — reset current pair and re-click")
    print("  'q'    — finish, average all samples, and save to JSON")
    print()

    cv2.namedWindow("Calibration Frame", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration Frame", click_event)
    cv2.imshow("Calibration Frame", frame_display)

    # ── Collect samples ──────────────────────────────────────────────────────
    samples = []  # list of computed pixels_per_meter values

    while True:
        key = cv2.waitKey(50) & 0xFF

        if key == ord('r'):
            reset_frame()

        elif key == ord('c'):
            if len(points) != 2:
                print("  ⚠ Need exactly 2 points before confirming.")
                continue

            (x1, y1), (x2, y2) = points
            pixel_dist = np.hypot(x2 - x1, y2 - y1)

            # Prompt for real-world distance in the terminal
            print()
            real_dist_str = input(
                "  Enter the real-world distance (in meters) between "
                "the two points\n"
                "  (e.g. 3.75 for lane width, 15 for dash+gap, "
                "1.7 for car width): "
            )
            try:
                real_dist = float(real_dist_str)
            except ValueError:
                print("  ⚠ Invalid number — sample discarded. "
                      "Press 'r' to re-click or 'c' to retry.")
                continue

            if real_dist <= 0:
                print("  ⚠ Distance must be positive — sample discarded.")
                continue

            ppm = pixel_dist / real_dist
            samples.append(ppm)

            print(f"\n  ✓ Sample #{len(samples)}")
            print(f"    Point 1:           {points[0]}")
            print(f"    Point 2:           {points[1]}")
            print(f"    Pixel distance:    {pixel_dist:.2f} px")
            print(f"    Real distance:     {real_dist} m")
            print(f"    pixels_per_meter:  {ppm:.3f}")
            print(f"\n  Press 'n' to add another sample, "
                  f"or 'q' to finish and save.")

        elif key == ord('n'):
            # Start a new sample
            reset_frame()
            print("  Ready for next sample — click two new points.")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

    # ── Compute final value ──────────────────────────────────────────────────
    if not samples:
        print("\n  No samples were confirmed — nothing saved.")
        return

    avg_ppm = float(np.mean(samples))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Samples taken: {len(samples)}")
    for i, s in enumerate(samples, 1):
        print(f"    Sample #{i}: {s:.3f} px/m")

    if len(samples) > 1:
        std = float(np.std(samples))
        coeff_var = std / avg_ppm if avg_ppm > 0 else 0
        if coeff_var > 0.15:
            print(f"\n  ⚠ WARNING: Samples disagree by "
                  f"{coeff_var * 100:.1f}% (> 15%)!")
            print("    Consider re-calibrating with more consistent points.")
        else:
            print(f"\n  ✓ Samples agree well "
                  f"(coefficient of variation: {coeff_var * 100:.1f}%)")

    print(f"\n  ➜ Final pixels_per_meter = {avg_ppm:.3f}")

    # ── Update JSON config ───────────────────────────────────────────────────
    config = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            config = json.load(f)

    config["pixels_per_meter"] = round(avg_ppm, 3)

    with open(json_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\n  ✓ Saved pixels_per_meter = {avg_ppm:.3f} "
          f"to {json_path}")
    print(f"    (all other keys preserved)")
    print()


if __name__ == "__main__":
    main()