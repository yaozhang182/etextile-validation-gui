#!/usr/bin/env python3
import csv
import time
import cv2

# ========= CONFIG =========
VIDEO_OUT = 'train.mp4'   # will fall back to AVI if MP4 fails
FRAMES_CSV = 'train_video.csv'
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 40                # target FPS; actual may vary by camera
CSV_DELIM_OUT = ';'
DRAW_TIMESTAMP = True     # overlay epoch on the video frames
CAM_INDEX = 0             # change if you have multiple cameras
# ==========================

def open_video_writer(path, width, height, fps):
    # Try MP4 first
    fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc_mp4, fps, (width, height))
    if out.isOpened():
        return out, path

    # Fall back to AVI (MJPG)
    alt_path = 'video.avi'
    fourcc_avi = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(alt_path, fourcc_avi, fps, (width, height))
    if out.isOpened():
        print(f"[Warn] MP4 open failed. Falling back to AVI: {alt_path}")
        return out, alt_path

    return None, None

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[Error] Could not open camera index {CAM_INDEX}.")
        print("Tip: On macOS, grant Camera permission in System Settings → Privacy & Security → Camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    out, path = open_video_writer(VIDEO_OUT, FRAME_WIDTH, FRAME_HEIGHT, FPS)
    if out is None:
        print("[Error] Could not create any video writer (MP4 or AVI).")
        cap.release()
        return

    frames_file = open(FRAMES_CSV, 'w', newline='')
    frames_writer = csv.writer(frames_file, delimiter=CSV_DELIM_OUT)
    frames_writer.writerow(['FrameIndex', 'EpochTime', 'Marker'])
    frames_file.flush()

    print("Recording... Press 'm' to insert marker, 'q' to quit.")
    print(f"Writing video to: {path}")
    print(f"Writing frame timestamps to: {FRAMES_CSV}")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Error] Failed to grab frame.")
                break

            epoch = time.time()

            if DRAW_TIMESTAMP:
                cv2.putText(frame, f"{epoch:.6f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            out.write(frame)

            marker = ''
            key = cv2.waitKey(1) & 0xFF
            if key == ord('m'):
                marker = 'M'
                print(f"[Marker] Inserted at {epoch:.6f}")
            elif key == ord('q'):
                frames_writer.writerow([frame_idx, epoch, marker])
                frames_file.flush()
                break

            frames_writer.writerow([frame_idx, epoch, marker])

            # Flush periodically
            if frame_idx % 30 == 0:
                frames_file.flush()

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[Info] Stopped by user (Ctrl+C).")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        frames_file.flush()
        frames_file.close()

    print(f"[Done] Saved video to {path} and frame times to {FRAMES_CSV}")

if __name__ == "__main__":
    main()