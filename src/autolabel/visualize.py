import os
import json
import cv2
import numpy as np
import logging
from collections import deque
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QScrollArea, QShortcut
from PyQt5.QtCore import Qt, QTimer, QRect, pyqtSignal, QPoint, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QKeySequence, QPolygon
from .trackers import Stabilizer, ImageTracker, VideoTracker, apply_homography
from .config import BaseConfig


def import_master_track(self, dir: str, filename: str):
    with open(f"{dir}/{filename}.json", 'r') as f:
        data = json.load(f)

    logging.info(f"Master track imported from {filename}")

    return data


class FrameViewer(QWidget):
    object_deleted = pyqtSignal(str)
    object_toggled = pyqtSignal(str)  # New signal for toggling is_class_instance
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(720)
        self.pixmap = None
        self.scaled_pixmap = None
        self.scale_factor = 1.0 # Default scale factor
        self.polygons = {}
        self.instances = {}  # Store instance data including is_class_instance status
        self.playing = False
        self.playing_backward = False
        self.variant_frames = []
        self.current_frame_idx = 0
        self.highlighted_obj_id = None
        
    def update_frame(self, frame_rgb, polygons, instances, frame_idx, playing, playing_backward, variant_frames, highlighted_obj_id=None):
        height, width = frame_rgb.shape[:2]
        self.polygons = polygons
        self.instances = instances  # Store instance data
        self.current_frame_idx = frame_idx
        self.playing = playing
        self.playing_backward = playing_backward
        self.variant_frames = variant_frames if variant_frames is not None else []
        self.highlighted_obj_id = highlighted_obj_id
        
        # Convert to QPixmap
        qimg = QImage(frame_rgb.data, width, height, frame_rgb.strides[0], QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg)
        
        # Calculate appropriate scale factor
        max_width = int(self.width() * 0.8) 
        max_height = int(self.height() * 0.8)
        
        width_scale = max_width / self.pixmap.width() if self.pixmap.width() > max_width else 1.0
        height_scale = max_height / self.pixmap.height() if self.pixmap.height() > max_height else 1.0
        
        # Use the smaller of the two scales to ensure the image fits entirely
        self.scale_factor = min(width_scale, height_scale)
        
        # Scale down only if needed (if scale factor < 1)
        if self.scale_factor < 1.0:
            scaled_size = QSize(
                int(self.pixmap.width() * self.scale_factor),
                int(self.pixmap.height() * self.scale_factor)
            )
            self.scaled_pixmap = self.pixmap.scaled(
                scaled_size, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
        else:
            self.scale_factor = 1.0
            self.scaled_pixmap = self.pixmap
        
        self.update()
        
    def paintEvent(self, event):
        if self.scaled_pixmap is None:
            return
            
        painter = QPainter(self)
        
        # Calculate centered position
        x = (self.width() - self.scaled_pixmap.width()) // 2
        y = (self.height() - self.scaled_pixmap.height()) // 2
        
        # Draw frame
        painter.drawPixmap(x, y, self.scaled_pixmap)
        
        # Draw polygons - scale them by the scale factor
        for obj_id, points in self.polygons.items():
            # Determine color based on is_class_instance
            if obj_id in self.instances and self.instances[obj_id]['is_class_instance']:
                # Use green for class instances
                base_color = QColor(0, 200, 0)  # Green for class instances
            else:
                # Use red for non-class instances
                base_color = QColor(200, 0, 0)  # Red for non-class instances
            
            # Scale points by scale factor
            scaled_points = []
            for p in points:
                scaled_x = x + int(p[0] * self.scale_factor)
                scaled_y = y + int(p[1] * self.scale_factor)
                scaled_points.append(QPoint(scaled_x, scaled_y))
            
            # Draw filled polygon with transparency
            poly = QPolygon(scaled_points)
            
            # Adjust pen width based on scale factor for consistent appearance
            pen_width = max(1, int(3 * self.scale_factor))
            highlight_pen_width = max(2, int(6 * self.scale_factor))
            
            # Higher opacity and thicker border for highlighted object
            if obj_id == self.highlighted_obj_id:
                painter.setOpacity(0.8)
                painter.setBrush(base_color)
                painter.setPen(QPen(QColor(255, 255, 0), highlight_pen_width))  # Yellow border for highlighted object
            else:
                painter.setOpacity(0.6)
                painter.setBrush(base_color)
                painter.setPen(QPen(base_color, pen_width))
                
            painter.drawPolygon(poly)
            
            # Draw object ID and confidence with dark background
            painter.setOpacity(1.0)
            confidence_text = ""
            if obj_id in self.instances and 'class_confidence' in self.instances[obj_id]:
                confidence = self.instances[obj_id]['class_confidence']
                confidence_text = f"{confidence:.2f}"
            
            text = f"ID: {obj_id} {confidence_text}"
            
            # Scale label size and position based on scale factor
            label_width = int(100 * self.scale_factor)  # Wider to accommodate confidence value
            label_height = int(20 * self.scale_factor)
            label_y_offset = int(25 * self.scale_factor)
            
            # Use the first point for label position
            if len(points) > 0:
                first_point = points[0]
                text_x = x + int(first_point[0] * self.scale_factor)
                text_y = y + int(first_point[1] * self.scale_factor) - label_y_offset
                
                # Make sure label is visible on screen
                text_rect = QRect(text_x, text_y, label_width, label_height)
                
                painter.fillRect(text_rect, QColor(0, 0, 0, 180))
                painter.setPen(QColor(255, 255, 255))
                
                # Adjust font size based on scale
                font = painter.font()
                font.setPointSizeF(max(6, 10 * self.scale_factor))  # Min size of 6pt
                painter.setFont(font)
                
                painter.drawText(text_rect, Qt.AlignCenter, text)
        
        # Draw status text with background
        is_variant = self.variant_frames and self.current_frame_idx in self.variant_frames
        play_status = "PLAYING ▶" if self.playing else "PLAYING ◀" if self.playing_backward else ""
        status = f"Frame: {self.current_frame_idx} {play_status} {'VARIANT' if is_variant else ''}"
        scale_info = f"Scale: {self.scale_factor:.2f}x"
        
        status_rect = QRect(10, 10, 300, 30)
        painter.fillRect(status_rect, QColor(0, 0, 0, 180))
        painter.setPen(QColor(0, 255, 0))
        painter.drawText(status_rect, Qt.AlignLeft | Qt.AlignVCenter, f"{status} | {scale_info}")
        
        # Draw instructions with background
        instructions = "j: mark variant | u: undo deletion | click object: toggle class | right: play forward | left: play backward | space: pause | q: quit | p: quit all"
        instr_rect = QRect(10, 50, self.width() - 20, 25)
        painter.fillRect(instr_rect, QColor(0, 0, 0, 180))
        painter.setPen(QColor(0, 255, 0))
        painter.drawText(instr_rect, Qt.AlignLeft | Qt.AlignVCenter, instructions)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.scaled_pixmap:
            # Adjust for centered pixmap
            x_offset = (self.width() - self.scaled_pixmap.width()) // 2
            y_offset = (self.height() - self.scaled_pixmap.height()) // 2
            
            # Get click position in scaled image coordinates
            scaled_x = event.x() - x_offset
            scaled_y = event.y() - y_offset
            
            # Check if click is within the image boundaries
            if 0 <= scaled_x < self.scaled_pixmap.width() and 0 <= scaled_y < self.scaled_pixmap.height():
                # Convert to original image coordinates
                x = scaled_x / self.scale_factor
                y = scaled_y / self.scale_factor
                
                # Check if click is inside any polygon (using original coordinates)
                for obj_id, points in self.polygons.items():
                    points_array = np.array(points, dtype=np.int32)
                    if cv2.pointPolygonTest(points_array, (x, y), False) >= 0:
                        self.object_toggled.emit(obj_id)  # Emit toggle signal instead of delete
                        break

class TrackViewer(QWidget):
    object_clicked = pyqtSignal(str)
    object_toggled = pyqtSignal(str)  # New signal for toggling is_class_instance
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.master_track = []
        self.deleted_ids = set()
        self.current_idx = 0
        self.highlighted_obj_id = None
        self.instances = {}  # Store instance data
        
    def sizeHint(self):
        # Calculate size based on content
        if not self.master_track:
            return QSize(100, 300)
        
        obj_ids = self.get_visible_object_ids()
        height = len(obj_ids) * 20 + 10  # Row height (20) plus some padding
        return QSize(100, height)

    def minimumSizeHint(self):
        return self.sizeHint()

    def get_visible_object_ids(self):
        """Helper method to get sorted visible object IDs"""
        obj_ids = set()
        for frame in self.master_track:
            obj_ids.update(frame['data'].keys())
        return sorted([oid for oid in obj_ids if oid not in self.deleted_ids], key=lambda x: int(x))
        
    def update_data(self, master_track, deleted_ids, current_idx, instances=None, highlighted_obj_id=None):
        self.master_track = master_track
        self.deleted_ids = deleted_ids
        self.current_idx = current_idx
        self.highlighted_obj_id = highlighted_obj_id
        self.instances = instances or {}
        
        # Update widget size when data changes
        self.updateGeometry()
        self.update()
    
    def paintEvent(self, event):
        if not self.master_track:
            return
            
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(50, 50, 50))
        
        # Get sorted visible object IDs
        obj_ids = self.get_visible_object_ids()
        
        num_frames = len(self.master_track)
        left_margin = 80  # Increased margin to accommodate toggle icon + ID label
        toggle_width = 20  # Width of toggle icon area
        timeline_width = self.width() - left_margin
        spacing = timeline_width / num_frames if num_frames > 0 else 0
        row_height = 20
        
        # Highlight current frame column
        if num_frames > 0:
            x_current = left_margin + spacing * self.current_idx + spacing / 2
            painter.setPen(QColor(255, 255, 255))
            painter.drawLine(int(x_current), 0, int(x_current), self.height())
        
        # Draw object rows
        for row_idx, obj_id in enumerate(obj_ids):
            y_row = row_idx * row_height
            
            # Highlight selected row
            if obj_id == self.highlighted_obj_id:
                painter.fillRect(0, y_row, self.width(), row_height, QColor(80, 80, 100))
            
            # Draw toggle icon (checkbox-like)
            toggle_rect = QRect(5, y_row + 2, toggle_width - 2, row_height - 4)
            is_class = self.instances.get(obj_id, {}).get('is_class_instance', False)
            
            # Draw toggle icon background
            painter.fillRect(toggle_rect, QColor(20, 20, 20))
            
            # Draw toggle icon state
            if is_class:
                # Green checkmark or filled box for enabled
                painter.setPen(QPen(QColor(0, 200, 0), 2))
                painter.drawLine(toggle_rect.left() + 3, toggle_rect.center().y(), 
                                toggle_rect.center().x() - 2, toggle_rect.bottom() - 5)
                painter.drawLine(toggle_rect.center().x() - 2, toggle_rect.bottom() - 5,
                                toggle_rect.right() - 3, toggle_rect.top() + 5)
            else:
                # Red X for disabled
                painter.setPen(QPen(QColor(200, 0, 0), 2))
                painter.drawLine(toggle_rect.left() + 3, toggle_rect.top() + 3, 
                                toggle_rect.right() - 3, toggle_rect.bottom() - 3)
                painter.drawLine(toggle_rect.left() + 3, toggle_rect.bottom() - 3,
                                toggle_rect.right() - 3, toggle_rect.top() + 3)
            
            # Draw object ID label with background and color indicating class status
            id_rect = QRect(toggle_width + 5, y_row + 2, 50, row_height - 4)
            
            # Color based on class status
            label_color = QColor(30, 150, 30) if is_class else QColor(150, 30, 30)
            
            painter.fillRect(id_rect, label_color)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(id_rect, Qt.AlignCenter, f"ID: {obj_id}")
            
            # Draw dots for each frame
            for i, frame in enumerate(self.master_track):
                data = frame['data']
                present = obj_id in data
                occluded = data[obj_id]['is_occluded'] if present else False
                
                if present and not occluded:
                    dot_color = QColor(0, 255, 0)  # Green
                elif present and occluded:
                    dot_color = QColor(255, 128, 0)  # Orange
                else:
                    dot_color = QColor(255, 0, 0)  # Red
                
                x = left_margin + i * spacing + spacing / 2
                y_center = y_row + row_height // 2
                
                painter.setBrush(dot_color)
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(int(x) - 3, int(y_center) - 3, 6, 6)
    
    def mousePressEvent(self, event):
        if not self.master_track:
            return
 
        # Get sorted visible object IDs
        obj_ids = self.get_visible_object_ids()
        
        row_height = 20
        clicked_row = event.y() // row_height
        
        if 0 <= clicked_row < len(obj_ids):
            obj_id = obj_ids[clicked_row]
            
            # Check if click is in toggle icon area (left side)
            toggle_width = 20
            if event.x() <= toggle_width:
                # Toggle icon clicked
                self.object_toggled.emit(obj_id)
            else:
                # Rest of row clicked - highlight object
                self.object_clicked.emit(obj_id)


class TrackingVisualizer(QMainWindow):
    def __init__(self, output_dir, video_name, video_path, video_tracker, stabilizer):
        super().__init__()
        self.output_dir = output_dir
        self.video_name = video_name
        self.video_path = video_path
        self.video_tracker = video_tracker
        self.stabilizer = stabilizer
        
        data = video_tracker.import_master_track(output_dir, video_name)
        self.master_track = data['important_frames']
        self.variant_frames = data.get('variant_frames', [])
        self.instances = data.get('instances', {})  # Load instance data
        self.deleted_obj_ids = set()
        # Store deletion history for undo
        self.deletion_history = deque(maxlen=50)  # Limit history to 50 items
        
        self.current_idx = 0
        self.playing = False
        self.playing_backward = False
        self.highlighted_obj_id = None
        self.cap = cv2.VideoCapture(video_path)
        
        self.setup_ui()
        
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.play_next_frame)
        self.play_timer.setInterval(100)
        
        # Calculate scaling properly on startup
        self.initialized = False
        QTimer.singleShot(100, self.initial_display)
    
    def initial_display(self):
        self.display_current_frame()
        self.initialized = True

    def setup_ui(self):
        self.setWindowTitle("Sock Tracking")
        self.resize(1280, 870)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        self.frame_viewer = FrameViewer()
        self.frame_viewer.object_toggled.connect(self.toggle_class_instance)  # Connect toggle signal
        main_layout.addWidget(self.frame_viewer)
        
        # Create scrollarea with padding for bottom items
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        self.track_container = QWidget()
        self.scroll_area.setWidget(self.track_container)
        
        track_layout = QVBoxLayout(self.track_container)
        track_layout.setContentsMargins(0, 0, 0, 20)  # Add bottom padding
        
        self.track_viewer = TrackViewer()
        self.track_viewer.object_clicked.connect(self.highlight_object)
        self.track_viewer.object_toggled.connect(self.toggle_class_instance)  # Connect the new toggle signal
        track_layout.addWidget(self.track_viewer)
        track_layout.addStretch()
        
        main_layout.addWidget(self.scroll_area)
        
        # Keyboard shortcuts
        QShortcut(QKeySequence(Qt.Key_J), self, self.toggle_variant)
        QShortcut(QKeySequence(Qt.Key_Q), self, self.close)
        QShortcut(QKeySequence(Qt.Key_P), self, lambda: exit(0))
        QShortcut(QKeySequence(Qt.Key_Right), self, self.start_playing_forward)
        QShortcut(QKeySequence(Qt.Key_Left), self, self.start_playing_backward)
        QShortcut(QKeySequence(Qt.Key_Space), self, self.stop_playing)
        QShortcut(QKeySequence(Qt.Key_U), self, self.undo_deletion)  # Add undo shortcut
        QShortcut(QKeySequence(Qt.Key_D), self, self.delete_highlighted_object)  # New shortcut for deletion


    def highlight_object(self, obj_id):
        self.highlighted_obj_id = obj_id
        self.display_current_frame()
    
    def toggle_class_instance(self, obj_id):
        """Toggle the is_class_instance field for the clicked object"""
        if obj_id in self.instances:
            # Toggle the is_class_instance field
            self.instances[obj_id]['is_class_instance'] = not self.instances[obj_id]['is_class_instance']
            logging.info(f"Toggled object {obj_id} is_class_instance to {self.instances[obj_id]['is_class_instance']}")
            self.display_current_frame()
        else:
            # If the instance doesn't exist in the instances dict, create it
            self.instances[obj_id] = {
                'class_confidence': 0.0,
                'is_class_instance': True  # Start as True when created manually
            }
            logging.info(f"Created new instance {obj_id} with is_class_instance=True")
            self.display_current_frame()
    
    def delete_highlighted_object(self):
        """Delete the currently highlighted object"""
        if self.highlighted_obj_id:
            self.delete_object(self.highlighted_obj_id)
    
    def display_current_frame(self, image_tracker):
        if not self.master_track:
            return
        
        frame_info = self.master_track[self.current_idx]
        frame_idx = frame_info['index']
        obj_data = frame_info['data']
        homography = np.array(frame_info['stabilizer_homography']).reshape((3, 3))
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = image_tracker.apply_calibration(frame)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = apply_homography(frame_rgb, homography, frame_rgb.shape[:2])
        
        polygons = {}
        for obj_id, info in obj_data.items():
            if obj_id in self.deleted_obj_ids:
                continue
            segmentation = info.get('segmentation', [])
            if segmentation and segmentation[0]:
                pts = np.array(segmentation[0]).reshape(-1, 2).astype(np.int32)
                polygons[obj_id] = pts.tolist()
        
        self.frame_viewer.update_frame(
            frame_rgb, 
            polygons,
            self.instances,  # Pass instance data 
            frame_idx, 
            self.playing, 
            self.playing_backward, 
            self.variant_frames,
            self.highlighted_obj_id
        )
        
        self.track_viewer.update_data(
            self.master_track, 
            self.deleted_obj_ids, 
            self.current_idx,
            self.instances,  # Pass instance data
            self.highlighted_obj_id
        )
    
    def toggle_variant(self):
        frame_idx = self.master_track[self.current_idx]['index']
        if self.variant_frames is None:
            self.variant_frames = []
            
        if frame_idx in self.variant_frames:
            self.variant_frames.remove(frame_idx)
            logging.info(f"Removed frame {frame_idx} from variants")
        else:
            self.variant_frames.append(frame_idx)
            logging.info(f"Added frame {frame_idx} to variants")
        self.display_current_frame()
    
    def delete_object(self, obj_id):
        # Add to deletion history before deleting
        self.deletion_history.append(obj_id)
        self.deleted_obj_ids.add(obj_id)
        self.highlighted_obj_id = None  # Clear highlight after deletion
        logging.info(f"Deleted object {obj_id}")
        self.display_current_frame()
    
    def undo_deletion(self):
        if not self.deletion_history:
            logging.info("Nothing to undo")
            return
            
        # Get the last deleted object ID
        obj_id = self.deletion_history.pop()
        
        # Remove it from the deleted set
        if obj_id in self.deleted_obj_ids:
            self.deleted_obj_ids.remove(obj_id)
            logging.info(f"Undid deletion of object {obj_id}")
            
            # Highlight the restored object
            self.highlighted_obj_id = obj_id
            self.display_current_frame()
    
    def play_next_frame(self):
        if self.playing and self.current_idx < len(self.master_track) - 1:
            self.current_idx += 1
            self.display_current_frame()
        elif self.playing_backward and self.current_idx > 0:
            self.current_idx -= 1
            self.display_current_frame()
        else:
            self.stop_playing()
    
    def start_playing_forward(self):
        self.playing = True
        self.playing_backward = False
        self.play_timer.start()
        self.display_current_frame()
    
    def start_playing_backward(self):
        self.playing = False
        self.playing_backward = True
        self.play_timer.start()
        self.display_current_frame()
    
    def stop_playing(self):
        self.playing = False
        self.playing_backward = False
        self.play_timer.stop()
        self.display_current_frame()
    
    def closeEvent(self, event):
        self.play_timer.stop()
        self.cap.release()
        
        if self.deleted_obj_ids or (self.variant_frames and self.current_idx == len(self.master_track) - 1) or self.instances:
            # Apply deletions and save changes
            data = self.video_tracker.import_master_track(self.output_dir, self.video_name)
            
            if self.deleted_obj_ids:
                for frame in self.master_track:
                    for obj_id in self.deleted_obj_ids:
                        if obj_id in frame['data']:
                            del frame['data'][obj_id]
                
                data['important_frames'] = self.master_track
                if 'num_objects' in data:
                    data['num_objects'] -= len(self.deleted_obj_ids)
                logging.info(f"Removed {len(self.deleted_obj_ids)} objects from all frames")
            
            # Save instance data
            data['instances'] = self.instances
            logging.info(f"Saved {len(self.instances)} instances with class status")
            
            if self.variant_frames and self.current_idx == len(self.master_track) - 1:
                data['important_frames'] = self.master_track
                data['variant_frames'] = self.variant_frames

            json_path = self.video_tracker._get_json_file_path(self.output_dir, self.video_name)
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            logging.info(f"Saved changes to {json_path}")
        else:
            logging.info(f"Quit without saving changes.")

        event.accept()


def visualize_all(config: BaseConfig):
    stabilizer = Stabilizer(config)
    image_tracker = ImageTracker(config, stabilizer, None, None)
    video_tracker = VideoTracker(config, image_tracker, None)

    video_files = sorted(
        [f for f in os.listdir(config.input_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    )

    # video_files = video_files[:2]

    app = QApplication([])
    logging.info(f"Input directory: {config.input_dir}")
    logging.info(f"Output directory: {config.output_dir}")
    
    for video_file in video_files:
        # Remove file extension if present
        video_name = video_file.rsplit('.', 1)[0] if '.' in video_file else video_file
        
        # Get full path with the original extension
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            potential_path = os.path.join(config.input_dir, video_name + ext)
            if os.path.exists(potential_path):
                video_path = potential_path
                break
        else:
            # If no matching file is found with extensions
            video_path = os.path.join(config.input_dir, video_file)
        
        logging.info(f"Processing video: {video_name}")
        
        # Check if the track file exists
        try:
            data = video_tracker.import_master_track(config.output_dir, video_name)
            # Ensure instances are present in the data
            if 'instances' not in data:
                logging.info(f"Adding instances field to {video_name}")
                data['instances'] = {}
                json_path = video_tracker._get_json_file_path(config.output_dir, video_name)
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=2)
                    
            if data.get('variant_frames') is not None and config.skip_variant_marked:
                logging.info(f"Skipping {video_name} - variant frames already marked")
                continue
        except Exception as e:
            logging.error(f"Error loading track file for {video_name}: {e}")
            continue
        
        visualizer = TrackingVisualizer(config.output_dir, video_name, video_path, video_tracker, stabilizer)
        visualizer.show()
        app.exec_()
