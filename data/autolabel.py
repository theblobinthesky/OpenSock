import argparse
import os, shutil, json
os.environ['TQDM_DISABLE'] = '1'
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autolabel.log'),
        logging.StreamHandler()
    ]
)

from stabilizer import Stabilizer
from image_tracker import ImageTracker
from video_tracker import VideoTracker

TEMP_DIR = "temp"
IOU_THRESH = 0.9
TRACK_SKIP = 20
MAX_INTERESTING_FRAMES = 180

def visualize_tracking(dir: str, video_name: str, video_path: str, video_tracker: VideoTracker, stabilizer: Stabilizer) -> None:
    data = video_tracker.import_master_track(dir, video_name)
    master_track = data['important_frames']
    
    if data.get('variant_frames') is not None:
        logging.debug(f"Skipping {video_name} - variant frames already marked")
        return
    
    variant_frames = []
    deleted_obj_ids = set()
    current_idx = 0
    masks = {}
    playing = False
    play_speed = 100  # milliseconds between frames
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.debug(f"Error: Could not open video {video_path}")
        return

    window_name = "Sock Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for obj_id, mask in masks.items():
                if mask[y, x]:
                    deleted_obj_ids.add(obj_id)
                    logging.debug(f"Deleted object {obj_id}")
                    display_frame()
                    break
    
    cv2.setMouseCallback(window_name, on_mouse)
    
    def display_frame():
        frame_data = master_track[current_idx]
        frame_idx, obj_data = frame_data['index'], frame_data['data']
        
        masks.clear()
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return False

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, stabilizer.output_warped_size, interpolation=cv2.INTER_AREA)
        stabilized_frame, _, _, _ = stabilizer.stabilize_frame(frame_rgb)
        display = cv2.cvtColor(stabilized_frame, cv2.COLOR_RGB2BGR)
        
        overlay = display.copy()
        h, w = overlay.shape[:2]
        mask_img = np.zeros((h, w), dtype=np.uint8)
        
        for obj_id, info in obj_data.items():
            if obj_id in deleted_obj_ids:
                continue
                
            segmentation = info.get('segmentation', [])
            bbox = info.get('bbox', [0, 0, 0, 0])
            
            color = plt.cm.tab20(int(obj_id) % 20)
            color = [int(c * 255) for c in color]
            
            if segmentation and len(segmentation[0]) > 0:
                points = np.array(segmentation[0]).reshape(-1, 2).astype(np.int32)
                if len(points) > 0:
                    cv2.fillPoly(overlay, [points], color)
                    cv2.polylines(overlay, [points], True, color, 4)
                    
                    obj_mask = np.zeros_like(mask_img)
                    cv2.fillPoly(obj_mask, [points], 255)
                    masks[obj_id] = obj_mask > 0
                    
                    x, y, w, h = bbox
                    cv2.putText(overlay, f"ID: {obj_id}", (int(x), int(y - 10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        
        playback_status = "PLAYING" if playing else ""
        variant_status = "VARIANT" if frame_idx in variant_frames else ""
        cv2.putText(display, f"Frame: {frame_idx} {playback_status} {variant_status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "j: mark variant | click: delete | right: play | space: pause | q: quit", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, display)
        return True
    
    display_frame()
    
    last_time = cv2.getTickCount() / cv2.getTickFrequency()
    
    while True:
        wait_time = 1 if playing else 100
        key = cv2.waitKey(wait_time) & 0xFF
        
        # Handle playback
        if playing:
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if current_time - last_time >= play_speed / 1000.0:
                last_time = current_time
                if current_idx < len(master_track) - 1:
                    current_idx += 1
                    display_frame()
                else:
                    playing = False  # Stop at the end
        
        # Handle key presses
        if key == ord('j'):
            frame_idx = master_track[current_idx]['index']
            if frame_idx in variant_frames:
                variant_frames.remove(frame_idx)
                logging.debug(f"Removed frame {frame_idx} from variants")
            else:
                variant_frames.append(frame_idx)
                logging.debug(f"Added frame {frame_idx} to variants")
            display_frame()
            
        elif key == ord('q'):
            break
            
        elif key == 83:  # Right arrow - start playback
            playing = True
            last_time = cv2.getTickCount() / cv2.getTickFrequency()
            
        elif key == 32:  # Space - pause playback
            playing = False
            display_frame()
            
        elif key == 81:  # Left arrow
            playing = False
            current_idx = max(current_idx - 1, 0)
            display_frame()
            
    cap.release()
    cv2.destroyAllWindows()

    # Apply changes to the data
    if deleted_obj_ids:
        for frame in master_track:
            for obj_id in deleted_obj_ids:
                if str(obj_id) in frame['data']:
                    del frame['data'][str(obj_id)]
        data['num_objects'] -= len(deleted_obj_ids)
        logging.debug(f"Removed {len(deleted_obj_ids)} objects from all frames")

    # Save changes
    if variant_frames or deleted_obj_ids:
        data['variant_frames'] = variant_frames
        json_path = video_tracker._get_json_file_path(dir, video_name)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        logging.debug(f"Saved changes to {json_path}")
               
            
def extract_frames(temp_output_dir: str, cap, frame_indices: list[int], output_size: tuple[int, int]):
    shutil.rmtree(temp_output_dir, ignore_errors=True)
    os.mkdir(temp_output_dir)
    
    for i, frame_idx in enumerate(frame_indices): 
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            logging.debug("Failed to retrieve frame.")
            return []
        
        frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{temp_output_dir}/{i}.jpeg", frame)

def process_video(
    video_path: str,
    output_dir: str,
    video_tracker: VideoTracker,
    diff_threshold: float,
    skip_frames: int,
) -> None:
    image_tracker = video_tracker.image_tracker
    stabilizer = image_tracker.stabilizer 
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.debug(f"Error: Could not open video file {video_path}")
        return
    
    interesting_frames = stabilizer.get_interesting_frames(cap, skip_frames, image_tracker.target_size, diff_threshold, MAX_INTERESTING_FRAMES)
    track_frames = interesting_frames[::TRACK_SKIP]
    logging.debug(f"Found {len(interesting_frames)} interesting frames and {len(track_frames)} track frames.")
    
    tracks = []
    obj_id_start = 0
    
    def _get_obj_ids():
        return list(range(obj_id_start, obj_id_start + len(interesting_frames)))
    
    for i, frame_idx in enumerate(track_frames):
        logging.debug(f"Tracking frame {i + 1}/{len(track_frames)}")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            logging.debug(f"Failed to read frame.")
            return
        
        masks = image_tracker._generate_masks(frame)
        masks = image_tracker._filter_masks(frame, masks)
        masks = image_tracker._classify_socks(frame, masks)
  
        frame_indices = list(reversed([i for i in interesting_frames if i <= frame_idx]))
        extract_frames(TEMP_DIR, cap, frame_indices, image_tracker.target_size)
        track_bw = video_tracker.track_forward(TEMP_DIR, frame_indices, masks, _get_obj_ids())
        
        frame_indices = [i for i in interesting_frames if i >= frame_idx]
        extract_frames(TEMP_DIR, cap, frame_indices, image_tracker.target_size)
        track_fw = video_tracker.track_forward(TEMP_DIR, frame_indices, masks, _get_obj_ids())

        track = [*reversed(track_bw[1:]), *track_fw]
        tracks.append(track)
        obj_id_start += len(interesting_frames)

    cap.release()
    
    # Merge the dictionaries.
    logging.debug("Merging and deduplicating tracks...")
    master_track, num_objs = video_tracker.merge_into_master_track(len(interesting_frames), 4, obj_id_start, IOU_THRESH, tracks)
    video_tracker.export_master_track(interesting_frames, master_track, num_objs, output_dir, video_name)

def main() -> None:
    parser = argparse.ArgumentParser(description="Process videos to track socks")
    parser.add_argument("--input_dir", type=str, default="sock_videos", help="Directory containing input videos")
    parser.add_argument("--output_dir", type=str, default="sock_video_results", help="Directory for output results")
    parser.add_argument("--diff_threshold", type=float, default=0.04, help="Threshold for frame difference to detect changes")
    parser.add_argument("--skip_frames", type=int, default=5, help="Number of frames to skip between checks")
    args = parser.parse_args()
    
    # Setup parameters for stabilizer and tracker
    output_warped_size = (800, 600)
    aruco_dict_type = cv2.aruco.DICT_6X6_250
    aruco_marker_id = 5
    marker_size_mm = 80.0
    sam2_checkpoint = "sam2.1_hiera_large.pt"
    sam2_config = "sam2.1_hiera_l.yaml"
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        os.makedirs(args.input_dir)
        logging.debug(f"Created directory {args.input_dir} - please add your videos there and run again")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize stabilizer and image tracker
    stabilizer = Stabilizer(
        aruco_dict_type=aruco_dict_type,
        aruco_marker_id=aruco_marker_id,
        output_warped_size=output_warped_size,
        marker_size_mm=marker_size_mm
    )
    
    image_tracker = ImageTracker(
        target_size=output_warped_size,
        stabilizer=stabilizer,
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config
    )
    
    # Create video tracker
    video_tracker = VideoTracker(
        image_tracker,
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config
    )
    
    # Get list of video files
    video_files = [f for f in os.listdir(args.input_dir) 
                  if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    video_files = sorted(video_files)
    
    if video_files:
        logging.debug(f"Processing the video files: {video_files}")
    else:
        logging.debug(f"No videos found in {args.input_dir}")
        return
    
    # Run visualization if requested
    if os.environ.get('VIZ') == '1':
        for video_file in video_files:
            video_path = os.path.join(args.input_dir, video_file)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            json_path = os.path.join(args.output_dir, f"{video_name}.json")
            
            if os.path.exists(json_path):
                logging.debug(f"\nVisualizing tracking results for: {video_file}")
                visualize_tracking(args.output_dir, video_name, video_path, video_tracker, stabilizer)
    else:
        for video_file in video_files:
            video_path = os.path.join(args.input_dir, video_file)
            logging.debug(f"\nProcessing video: {video_file}")
            
            process_video(
                video_path=video_path,
                output_dir=args.output_dir,
                video_tracker=video_tracker,
                diff_threshold=args.diff_threshold,
                skip_frames=args.skip_frames,
            )
    
    logging.debug("\nAll operations completed successfully!")


if __name__ == "__main__":
    main()