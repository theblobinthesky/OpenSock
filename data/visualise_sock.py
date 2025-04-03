import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import pygame
from trackers import Stabilizer, ImageTracker, VideoTracker, apply_homography

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('autolabel.log'), logging.StreamHandler()]
)

# Layout constants
PREVIEW_HEIGHT = 720
TRACK_VIEWER_HEIGHT = 150
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = PREVIEW_HEIGHT + TRACK_VIEWER_HEIGHT
TRACK_VIEWER_LEFT_MARGIN = 50
ROW_HEIGHT = 20

# Scrollbar constants
SCROLLBAR_WIDTH = 10
SCROLLBAR_MARGIN = 5

def display_frame(master_track, current_idx, deleted_ids, cap, stabilizer, font, big_font, playing, variant_frames):
    frame_info = master_track[current_idx]
    frame_idx = frame_info['index']
    obj_data = frame_info['data']
    homography = np.array(frame_info['stabilizer_homography']).reshape((3, 3))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None, {}

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = apply_homography(frame_rgb, homography, stabilizer.output_warped_size)
    frame_surface = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
    overlay = frame_surface.copy()
    polygons = {}

    for obj_id, info in obj_data.items():
        if obj_id in deleted_ids:
            continue
        segmentation = info.get('segmentation', [])
        bbox = info.get('bbox', [0, 0, 0, 0])
        color = plt.cm.tab20(int(obj_id) % 20)
        color = tuple(int(c * 255) for c in color[:3])
        if segmentation and segmentation[0]:
            pts = np.array(segmentation[0]).reshape(-1, 2).astype(np.int32)
            pts_list = pts.tolist()
            polygons[obj_id] = pts_list
            pygame.draw.polygon(overlay, color, pts_list, 0)
            pygame.draw.polygon(overlay, color, pts_list, 4)
            x, y, w, h = bbox
            text_surface = font.render(f"ID: {obj_id}", True, color)
            overlay.blit(text_surface, (int(x), int(y - 10)))

    blended = frame_surface.copy()
    overlay.set_alpha(153)
    blended.blit(overlay, (0, 0))
    status_text = big_font.render(
        f"Frame: {frame_idx} {'PLAYING' if playing else ''} {'VARIANT' if frame_idx in variant_frames else ''}",
        True, (0, 255, 0))
    blended.blit(status_text, (10, 10))
    instructions = font.render(
        "j: mark variant | left-click: delete | right: play | space: pause | q: quit", True, (0, 255, 0))
    blended.blit(instructions, (10, 50))
    return blended, polygons

def draw_track_viewer(surface, master_track, deleted_ids, current_idx, scroll_offset, window_width, viewer_height, row_height, left_margin):
    # Compute sorted valid object IDs.
    obj_ids = set()
    for frame in master_track:
        obj_ids.update(frame['data'].keys())
    obj_ids = sorted([oid for oid in obj_ids if oid not in deleted_ids], key=lambda x: int(x))
    num_frames = len(master_track)
    timeline_width = window_width - left_margin - SCROLLBAR_WIDTH - SCROLLBAR_MARGIN * 2
    spacing = timeline_width / num_frames if num_frames > 0 else 0

    # Draw viewer background.
    surface.fill((50, 50, 50))
    # Highlight the current frame column.
    if num_frames > 0:
        x_current = left_margin + num_frames * spacing * (current_idx / num_frames) + spacing / 2
        pygame.draw.line(surface, (255, 255, 255), (x_current, 0), (x_current, viewer_height), 2)

    font_small = pygame.font.SysFont(None, 16)
    for row_idx, obj_id in enumerate(obj_ids):
        y_row = row_idx * row_height - scroll_offset
        if y_row < 0 or y_row > viewer_height - row_height:
            continue
        # Draw the object id label.
        text = font_small.render(f"ID: {obj_id}", True, (255, 255, 255))
        surface.blit(text, (5, y_row + row_height // 4))
        # Draw a dot per frame.
        for i, frame in enumerate(master_track):
            present = obj_id in frame['data']
            dot_color = (0, 255, 0) if present else (255, 0, 0)
            x = left_margin + i * spacing + spacing / 2
            y_center = y_row + row_height // 2
            pygame.draw.circle(surface, dot_color, (int(x), int(y_center)), 3)

    # Draw the scrollbar on the right.
    # Compute total content height.
    total_height = len(obj_ids) * row_height
    if total_height > viewer_height:
        scrollbar_track_rect = pygame.Rect(window_width - SCROLLBAR_WIDTH - SCROLLBAR_MARGIN,
                                             SCROLLBAR_MARGIN,
                                             SCROLLBAR_WIDTH,
                                             viewer_height - 2 * SCROLLBAR_MARGIN)
        # Draw track background.
        pygame.draw.rect(surface, (80, 80, 80), scrollbar_track_rect, border_radius=5)
        # Compute slider height.
        slider_height = (viewer_height - 2 * SCROLLBAR_MARGIN) * (viewer_height / total_height)
        slider_height = max(slider_height, 20)
        max_scroll = total_height - viewer_height
        slider_y = SCROLLBAR_MARGIN + (scroll_offset / max_scroll) * ((viewer_height - 2 * SCROLLBAR_MARGIN) - slider_height)
        slider_rect = pygame.Rect(window_width - SCROLLBAR_WIDTH - SCROLLBAR_MARGIN,
                                  int(slider_y),
                                  SCROLLBAR_WIDTH,
                                  int(slider_height))
        pygame.draw.rect(surface, (150, 150, 150), slider_rect, border_radius=5)
    else:
        # No scrolling needed; draw a full-height scrollbar.
        scrollbar_track_rect = pygame.Rect(window_width - SCROLLBAR_WIDTH - SCROLLBAR_MARGIN,
                                             SCROLLBAR_MARGIN,
                                             SCROLLBAR_WIDTH,
                                             viewer_height - 2 * SCROLLBAR_MARGIN)
        pygame.draw.rect(surface, (80, 80, 80), scrollbar_track_rect, border_radius=5)
        pygame.draw.rect(surface, (150, 150, 150), scrollbar_track_rect, border_radius=5)

def visualize_tracking_pygame(output_dir, video_name, video_path, video_tracker, stabilizer):
    data = video_tracker.import_master_track(output_dir, video_name)
    master_track = data['important_frames']
    if data.get('variant_frames') is not None:
        logging.debug(f"Skipping {video_name} - variant frames already marked")
        # return

    variant_frames = []
    deleted_obj_ids = set()
    current_idx = 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.debug(f"Error: Could not open video {video_path}")
        return

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Sock Tracking")
    font = pygame.font.SysFont(None, 24)
    big_font = pygame.font.SysFont(None, 32)
    clock = pygame.time.Clock()
    last_update = pygame.time.get_ticks()
    playing = False
    play_speed = 100  # milliseconds per frame

    scroll_offset = 0
    dragging_scrollbar = False
    drag_start_y = 0
    scroll_offset_start = 0

    preview_surface, polygons = display_frame(master_track, current_idx, deleted_obj_ids, cap, stabilizer, font, big_font, playing, variant_frames)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_j:
                    frame_idx = master_track[current_idx]['index']
                    if frame_idx in variant_frames:
                        variant_frames.remove(frame_idx)
                        logging.debug(f"Removed frame {frame_idx} from variants")
                    else:
                        variant_frames.append(frame_idx)
                        logging.debug(f"Added frame {frame_idx} to variants")
                    preview_surface, polygons = display_frame(master_track, current_idx, deleted_obj_ids, cap, stabilizer, font, big_font, playing, variant_frames)
                elif event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_RIGHT:
                    playing = True
                    last_update = pygame.time.get_ticks()
                elif event.key == pygame.K_SPACE:
                    playing = False
                    preview_surface, polygons = display_frame(master_track, current_idx, deleted_obj_ids, cap, stabilizer, font, big_font, playing, variant_frames)
                elif event.key == pygame.K_LEFT:
                    playing = False
                    current_idx = max(current_idx - 1, 0)
                    preview_surface, polygons = display_frame(master_track, current_idx, deleted_obj_ids, cap, stabilizer, font, big_font, playing, variant_frames)
                elif event.key == pygame.K_UP:
                    scroll_offset = max(scroll_offset - 10, 0)
                elif event.key == pygame.K_DOWN:
                    # Limit scroll based on total track viewer height.
                    obj_ids = set()
                    for frame in master_track:
                        obj_ids.update(frame['data'].keys())
                    valid_ids = sorted([oid for oid in obj_ids if oid not in deleted_obj_ids], key=lambda x: int(x))
                    total_height = len(valid_ids) * ROW_HEIGHT
                    scroll_offset = min(scroll_offset + 10, max(0, total_height - TRACK_VIEWER_HEIGHT))

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    pos = event.pos
                    if pos[1] >= PREVIEW_HEIGHT:
                        # Check if click is within the scrollbar region.
                        if pos[0] >= WINDOW_WIDTH - SCROLLBAR_WIDTH - SCROLLBAR_MARGIN:
                            dragging_scrollbar = True
                            drag_start_y = pos[1]
                            scroll_offset_start = scroll_offset
                        else:
                            # Click in track viewer (but not on scrollbar) -- could be extended.
                            pass
                    else:
                        # In preview area: check for deletion.
                        for obj_id, pts_list in polygons.items():
                            pts_array = np.array(pts_list, dtype=np.int32)
                            if cv2.pointPolygonTest(pts_array, pos, False) >= 0:
                                deleted_obj_ids.add(obj_id)
                                logging.debug(f"Deleted object {obj_id}")
                                preview_surface, polygons = display_frame(master_track, current_idx, deleted_obj_ids, cap, stabilizer, font, big_font, playing, variant_frames)
                                break

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging_scrollbar = False

            elif event.type == pygame.MOUSEMOTION:
                if dragging_scrollbar:
                    # Calculate new scroll offset based on mouse movement.
                    dy = event.pos[1] - drag_start_y
                    # Get total content height.
                    obj_ids = set()
                    for frame in master_track:
                        obj_ids.update(frame['data'].keys())
                    valid_ids = sorted([oid for oid in obj_ids if oid not in deleted_obj_ids], key=lambda x: int(x))
                    total_height = len(valid_ids) * ROW_HEIGHT
                    if total_height > TRACK_VIEWER_HEIGHT:
                        max_scroll = total_height - TRACK_VIEWER_HEIGHT
                        # Move scroll offset in proportion to mouse movement.
                        scroll_offset = scroll_offset_start + (dy / TRACK_VIEWER_HEIGHT) * max_scroll
                        scroll_offset = max(0, min(scroll_offset, max_scroll))
            
            elif event.type == pygame.MOUSEWHEEL:
                obj_ids = set()
                for frame in master_track:
                    obj_ids.update(frame['data'].keys())
                valid_ids = sorted([oid for oid in obj_ids if oid not in deleted_obj_ids], key=lambda x: int(x))
                total_height = len(valid_ids) * ROW_HEIGHT
                scroll_offset -= event.y * 10
                scroll_offset = max(0, min(scroll_offset, max(0, total_height - TRACK_VIEWER_HEIGHT)))

        if playing:
            current_time = pygame.time.get_ticks()
            if current_time - last_update >= play_speed:
                last_update = current_time
                if current_idx < len(master_track) - 1:
                    current_idx += 1
                    preview_surface, polygons = display_frame(master_track, current_idx, deleted_obj_ids, cap, stabilizer, font, big_font, playing, variant_frames)
                else:
                    playing = False

        screen.fill((0, 0, 0))
        # Draw video preview.
        if preview_surface is not None:
            preview_rect = preview_surface.get_rect(center=(WINDOW_WIDTH // 2, PREVIEW_HEIGHT // 2))
            screen.blit(preview_surface, preview_rect)
        # Draw track viewer.
        track_viewer_surface = pygame.Surface((WINDOW_WIDTH, TRACK_VIEWER_HEIGHT))
        draw_track_viewer(track_viewer_surface, master_track, deleted_obj_ids, current_idx, scroll_offset,
                          WINDOW_WIDTH, TRACK_VIEWER_HEIGHT, ROW_HEIGHT, TRACK_VIEWER_LEFT_MARGIN)
        screen.blit(track_viewer_surface, (0, PREVIEW_HEIGHT))

        pygame.display.flip()
        clock.tick(30)

    cap.release()
    pygame.quit()

    if deleted_obj_ids:
        for frame in master_track:
            for obj_id in deleted_obj_ids:
                if obj_id in frame['data']:
                    del frame['data'][obj_id]
        data['num_objects'] -= len(deleted_obj_ids)
        logging.debug(f"Removed {len(deleted_obj_ids)} objects from all frames")
    if variant_frames or deleted_obj_ids:
        data['variant_frames'] = variant_frames
        json_path = video_tracker._get_json_file_path(output_dir, video_name)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        logging.debug(f"Saved changes to {json_path}")

if __name__ == "__main__":
    diff_threshold = 0.04
    skip_frames = 5
    output_warped_size = (540, 960)
    aruco_dict_type = cv2.aruco.DICT_6X6_250
    aruco_marker_id = 5
    secondary_aruco_marker_id = 10
    marker_size_mm = 80.0
    sam2_checkpoint = "sam2.1_hiera_large.pt"
    sam2_config = "sam2.1_hiera_l.yaml"
    track_skip = 40
    max_interesting_frames = 180
    iou_thresh = 0.9

    stabilizer = Stabilizer(
        aruco_dict_type=aruco_dict_type,
        aruco_marker_id=aruco_marker_id,
        secondary_aruco_marker_id=secondary_aruco_marker_id,
        output_warped_size=output_warped_size,
        marker_size_mm=marker_size_mm
    )

    image_tracker = ImageTracker(
        target_size=output_warped_size,
        stabilizer=stabilizer,
        sam2=None,
        imagenet_class=None,
        classifier=None,
        classifier_transform=None
    )

    video_tracker = VideoTracker(image_tracker, sam2_video=None)

    input_dir = "sock_videos"
    output_dir = "sock_video_results"

    video_files = sorted(
        [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    )

    for video_file in video_files:
        video_file = video_file[:-4]
        video_path = os.path.join(input_dir, video_file + ".mov")
        logging.debug(f"Processing video: {video_file}")

        visualize_tracking_pygame(output_dir, video_file, video_path, video_tracker, stabilizer)
