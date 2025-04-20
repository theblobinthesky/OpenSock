from .config import BaseConfig
from .timing import timed
from .video_capture import VideoContext, open_video_capture
import os
import torch, cv2, numpy as np
import logging


def extract_frames(video_ctx: VideoContext, temp_output_dir: str, frames: list[int]):
    for sub_path in os.listdir(temp_output_dir):
        os.remove(f"{temp_output_dir}/{sub_path}")

    if not os.path.exists(temp_output_dir):
        os.mkdir(temp_output_dir)

    for i, (_, frame) in enumerate(open_video_capture(video_ctx, frames)):
        cv2.imwrite(f"{temp_output_dir}/{i}.jpeg", frame)


def track_forward(input_dir: str, frame_indices: list[int], masks: list, obj_ids: list, sam2_video):
    track = list(range(len(frame_indices)))
            
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = sam2_video.init_state(input_dir)

        for mask, obj_id in zip(masks, obj_ids):
            sam2_video.add_new_mask(state, 0, obj_id, mask['segmentation'])
        
        for out_frame_idx, out_obj_ids, out_mask_logits in sam2_video.propagate_in_video(state):
            contours_of_obj_ids = []
            for i in range(len(out_obj_ids)):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                mask_uint8 = mask.astype(np.uint8).squeeze(0)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    largest_contour = cv2.approxPolyDP(largest_contour, 1e-3 * cv2.arcLength(largest_contour, True), True)
                    contours_of_obj_ids.append(largest_contour)
                else:
                    contours_of_obj_ids.append([])
                    
            track[out_frame_idx] = {
                out_obj_id: contours_of_obj_ids[i]
                for i, out_obj_id in enumerate(out_obj_ids)
            } 

    return track


def merge_into_master_track(num_interesting_frames: int, min_num_occ_for_action: int, instances: list, tracks: list):
    delete_instance = set()
    master_track = []
    for frame_idx in range(num_interesting_frames):
        merged = {}
        for track in tracks:
            merged.update(track[frame_idx])
        
        master_track.append(merged)


    # Bake all polygons.
    for frame in master_track:
        for obj_id in frame.keys():
            contour = frame[obj_id]
            if len(contour) == 0:
                frame[obj_id] = (None, contour)
            else:
                poly = Polygon(np.array(contour).reshape((-1, 2))).buffer(0)
                frame[obj_id] = (poly, contour)

    # Now apply the following rules:

    # Rule 1: a inside b for mt. thresh frames -> remove a
    # This might be a bad idea since we're rejecting non-class instances later.

    # Rule 2: Based on intersection over union metric, merge the object ids 
    # that seem to track the same object in a majority of frames.

    def bbox_inters(contour1, contour2):
        if len(contour1) == 0 or len(contour2) == 0: return False

        x1, y1, w1, h1 = cv2.boundingRect(contour1)
        x2, y2, w2, h2 = cv2.boundingRect(contour2)

        left = max(x1, x2)
        right = min(x1 + w1, x2 + w2)
        top = max(y1, y2)
        bottom = min(y1 + h1, y2 + h2)

        inters_width = max(0.0, right - left)
        inters_height = max(0.0, bottom - top)

        return inters_width * inters_height

    # Apply both rules.
    a_inside_b = {}
    to_merge = {}
    num_instances = len(instances)
    for frame in master_track:
        for obj_id, (poly, contour) in frame.items():
            if len(contour) == 0: continue
            x, y, w, h = cv2.boundingRect(contour)

            for obj_id_2 in frame.keys():
                if obj_id == obj_id_2: continue
                poly_2, contour_2 = frame[obj_id_2]

                inters_upper_bound = bbox_inters(contour, contour_2)
                if inters_upper_bound == 0.0: continue

                # Test for rule 1:
                mask_area = poly.area
                inters = None
                if mask_area > 0 and inters_upper_bound / mask_area >= self.config.video_tracker_inside_threshold:
                    # Test with the real rule 1.
                    inters = poly.intersection(poly_2).area

                    if inters / mask_area >= self.config.video_tracker_inside_threshold:
                        a_inside_b[(obj_id, obj_id_2)] = a_inside_b.get((obj_id, obj_id_2), 0) + 1

                # Rule 2 is symetric, Rule 1 is antisymmetric.
                if obj_id >= obj_id_2: continue

                # Test for full rule 2:
                union_lower_bound = max(poly.area + poly_2.area - inters_upper_bound, 0.1)
                if inters_upper_bound / union_lower_bound >= self.config.instance_merge_iou_thresh:
                    if inters is None: 
                        inters = poly.intersection(poly_2).area

                    union = poly.area + poly_2.area - inters
                    if union > 0 and inters / union >= self.config.instance_merge_iou_thresh:
                        to_merge[(obj_id, obj_id_2)] = to_merge.get((obj_id, obj_id_2), 0) + 1


        # Apply rule 1: 
        # maj_votes = [ids for ids, num_votes in a_inside_b.items() if num_votes >= min_num_occ_for_action]
        # for obj_id_1, _ in maj_votes:
        #     delete_instance.add(obj_id_1)
        #     for frame, rtree in zip(master_track, rtrees_for_frames):
        #         if obj_id_1 in frame: 
        #             del frame[obj_id_1]

        # Apply rule 2: 
        maj_votes = [ids for ids, num_votes in to_merge.items() if num_votes >= min_num_occ_for_action]
        for obj_id_1, obj_id_2 in maj_votes:
            new_obj_id = num_instances
            instances.append(instances[obj_id_1])
            num_instances += 1

            del to_merge[(obj_id_1, obj_id_2)]
            delete_instance.add(obj_id_1)
            delete_instance.add(obj_id_2)

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
    master_track = [{id_map[old_id]: pair for old_id, pair in frame.items()} for frame in master_track]

    instances = {id: instances[id] for id in unique_ids if id not in delete_instance}
    instances = {id_map[old_id]: instance for old_id, instance in instances.items()}


    # Add metadata.
    for frame in master_track:
        for obj_id, (_, contour) in list(frame.items()):

            if len(contour) == 0:
                area = 0.0
                segmentation = []
                y, x, h, w = 0, 0, 0, 0
            else:
                area = cv2.contourArea(contour)
                segmentation = contour.reshape(-1, 2).flatten().tolist()
                y, x, h, w = cv2.boundingRect(contour)

            frame[obj_id] = {
                "segmentation": segmentation,
                "area": area,
                "bbox": [x, y, w, h],
                "is_occluded": False
            }

                
    return instances, master_track


def mark_occlusions(master_track: list):
    for frame in master_track:
        for track_id, data in list(frame.items()):
            if len(data["segmentation"]) == 0:
                data["is_occluded"] = True

    return master_track


@timed
def process_multiple_frames(interesting_frames: list[int], track_frames: list[int], masks_per_frame, video_ctx: VideoContext, config: BaseConfig, sam2_video):
    instances = []
    tracks = []
    obj_id_start = 0

    def _get_obj_ids(masks):
        nonlocal obj_id_start
        ids = list(range(obj_id_start, obj_id_start + len(masks)))
        return ids
    
    for frame_idx, masks in zip(track_frames, masks_per_frame):
        if len(masks) == 0:
            logging.info("Failed to find socks. Skipping the frame.")
            continue

        instances.extend(masks)

        frame_indices = list(reversed([i for i in interesting_frames if i <= frame_idx]))
        extract_frames(video_ctx, config.temp_fs_dir, frame_indices)
        track_bw = track_forward(config.temp_fs_dir, frame_indices, masks, _get_obj_ids(masks), sam2_video)

        frame_indices = [i for i in interesting_frames if i >= frame_idx]
        extract_frames(video_ctx, config.temp_fs_dir, frame_indices)
        track_fw = track_forward(config.temp_fs_dir, frame_indices, masks, _get_obj_ids(masks), sam2_video)

        track = [*reversed(track_bw[1:]), *track_fw]
        tracks.append(track)
        obj_id_start += len(masks)

        track_frame_idx += 1
        logging.info(f"Tracked frame {track_frame_idx}/{len(track_frames)}.")


    instances, master_track = merge_into_master_track(len(interesting_frames), 4, instances, tracks)
    mark_occlusions(master_track)

    return instances, master_track
