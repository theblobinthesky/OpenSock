import numpy as np
import os
import json
from image_tracker import ImageTracker
import torch, cv2
from sam2.build_sam import build_sam2_video_predictor

class VideoTracker:
    def __init__(self, image_tracker: ImageTracker, sam2_checkpoint: str, sam2_config: str):
        self.image_tracker = image_tracker
        self.device = self.image_tracker.device
        self.sam = build_sam2_video_predictor(f"/{os.path.abspath(sam2_config)}", sam2_checkpoint)
    
    def track_forward(self, input_dir: str, frame_indices: list[int], masks: list, obj_ids: list):
        track = list(range(len(frame_indices)))
               
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.sam.init_state(input_dir)

            for mask, obj_id in zip(masks, obj_ids):
                self.sam.add_new_mask(state, 0, obj_id, mask['segmentation'])
           
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam.propagate_in_video(state):
                track[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                } 

        return track

    def merge_into_master_track(self, num_interesting_frames: int, min_num_occ_for_merge: int, max_num_objs: int, iou_thresh: float, tracks: list):
        master_track = []
        for frame_idx in range(num_interesting_frames):
            merged = {}
            for track in tracks:
                merged.update(track[frame_idx])
            
            master_track.append(merged) 

        # Based on intersection over union metric, merge the object ids 
        # that seem to track the same object in a majority of frames.
        to_merge = {}
        for i, frame in enumerate(master_track):
            print(f"Deduplicating frame {i + 1}/{num_interesting_frames}")
            for obj_id, mask in frame.items():
                for obj_id_2, mask_2 in frame.items():
                    if obj_id >= obj_id_2: continue
                    
                    inter = (mask & mask_2).sum()
                    union = (mask | mask_2).sum()
                    
                    if union > 0:
                        iou = inter / union
                        if iou >= iou_thresh:
                            to_merge[(obj_id, obj_id_2)] = to_merge.get((obj_id, obj_id_2), 0) + 1

            maj_votes = [ids for ids, num_votes in to_merge.items() if num_votes >= min_num_occ_for_merge]
            for obj_id_1, obj_id_2 in maj_votes:
                new_obj_id = max_num_objs
                max_num_objs += 1
                del to_merge[(obj_id_1, obj_id_2)]

                for frame in master_track:

                    if obj_id_1 in frame: 
                        frame[new_obj_id] = frame[obj_id_1]
                        del frame[obj_id_1]
                        
                    if obj_id_2 in frame: 
                        frame[new_obj_id] = frame[obj_id_2]
                        del frame[obj_id_2]
                
                            
        # Remap object ids to continuous integers starting from 0.
        unique_ids = set()
        for frame in master_track:
            unique_ids.update(frame.keys())

        sorted_ids = sorted(unique_ids)
        id_map = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}
        master_track = [{id_map[old_id]: mask for old_id, mask in frame.items()} for frame in master_track]
                    
        return master_track, len(sorted_ids)

    def _get_json_file_path(self, dir: str, filename: str):
        return f"{dir}/{filename}.json"

    def export_master_track(self, important_frame_indices: list[int], master_track: list, num_objs: int, dir: str, filename: str):
        important_frames = []
        for frame_idx, frame in zip(important_frame_indices, master_track):
            frame_data = {}
            for track_id, mask in frame.items():
                mask_uint8 = mask.astype(np.uint8).squeeze(0)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    largest_contour = cv2.approxPolyDP(largest_contour, 1e-3 * cv2.arcLength(largest_contour, True), True)
                    segmentation = largest_contour.reshape(-1, 2).flatten().tolist()
                    x, y, w, h = cv2.boundingRect(largest_contour)
                else:
                    segmentation = []
                    x, y, w, h = 0, 0, 0, 0
                frame_data[str(track_id)] = {
                    "segmentation": [segmentation],
                    "bbox": [x, y, w, h],
                    "area": float(np.sum(mask))
                }
            important_frames.append({
                'index': frame_idx,
                'data': frame_data
            })

        export_data = {
            'num_objects': num_objs,
            'important_frames': important_frames,
            'variant_frames': None
        }
        
        with open(self._get_json_file_path(dir, filename), 'w') as f:
            json.dump(export_data, f, indent=2)
            
        print(f"Master track exported to {filename}")


    def import_master_track(self, dir: str, filename: str):
        with open(self._get_json_file_path(dir, filename), 'r') as f:
            data = json.load(f)

        print(f"Master track imported from {filename}")

        return data