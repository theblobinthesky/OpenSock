import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from collections import deque
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QScrollArea, QShortcut
from PyQt5.QtCore import Qt, QTimer, QRect, pyqtSignal, QPoint, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QKeySequence, QPolygon
from .trackers import Stabilizer, ImageTracker, VideoTracker, apply_homography
from .config import BaseConfig


class FrameViewer(QWidget):
    object_deleted = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(720)
        self.pixmap = None
        self.scaled_pixmap = None
        self.scale_factor = 1.0 # Default scale factor
        self.polygons = {}
        self.playing = False
        self.playing_backward = False
        self.variant_frames = []
        self.current_frame_idx = 0
        self.highlighted_obj_id = None
        
    def update_frame(self, frame_rgb, polygons, frame_idx, playing, playing_backward, variant_frames, highlighted_obj_id=None):
        height, width = frame_rgb.shape[:2]
        self.polygons = polygons
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
            color = plt.cm.tab20(int(obj_id) % 20)
            r, g, b = [int(c * 255) for c in color[:3]]
            
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
                painter.setBrush(QColor(r, g, b))
                painter.setPen(QPen(QColor(255, 255, 0), highlight_pen_width))  # Yellow border for highlighted object
            else:
                painter.setOpacity(0.6)
                painter.setBrush(QColor(r, g, b))
                painter.setPen(QPen(QColor(r, g, b), pen_width))
                
            painter.drawPolygon(poly)
            
            # Draw object ID with dark background
            painter.setOpacity(1.0)
            text = f"ID: {obj_id}"
            
            # Scale label size and position based on scale factor
            label_width = int(60 * self.scale_factor)
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
        instructions = "j: mark variant | u: undo deletion | click object: delete | right: play forward | left: play backward | space: pause | q: quit"
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
                        self.object_deleted.emit(obj_id)
                        break

class TrackViewer(QWidget):
    object_clicked = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(150)
        self.master_track = []
        self.deleted_ids = set()
        self.current_idx = 0
        self.highlighted_obj_id = None
        
    def update_data(self, master_track, deleted_ids, current_idx, highlighted_obj_id=None):
        self.master_track = master_track
        self.deleted_ids = deleted_ids
        self.current_idx = current_idx
        self.highlighted_obj_id = highlighted_obj_id
        self.update()
    
    def paintEvent(self, event):
        if not self.master_track:
            return
            
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(50, 50, 50))
        
        # Compute sorted valid object IDs
        obj_ids = set()
        for frame in self.master_track:
            obj_ids.update(frame['data'].keys())
        obj_ids = sorted([oid for oid in obj_ids if oid not in self.deleted_ids], key=lambda x: int(x))
        
        num_frames = len(self.master_track)
        left_margin = 50
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
            
            # Draw object ID label with background
            id_rect = QRect(5, y_row + 2, 40, row_height - 4)
            painter.fillRect(id_rect, QColor(30, 30, 30))
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
            
        # Calculate row height and find which row was clicked
        obj_ids = set()
        for frame in self.master_track:
            obj_ids.update(frame['data'].keys())
        obj_ids = sorted([oid for oid in obj_ids if oid not in self.deleted_ids], key=lambda x: int(x))
        
        row_height = 20
        clicked_row = event.y() // row_height
        
        if 0 <= clicked_row < len(obj_ids):
            self.object_clicked.emit(obj_ids[clicked_row])

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
        
        self.display_current_frame()
    
    def setup_ui(self):
        self.setWindowTitle("Sock Tracking")
        self.resize(1280, 870)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        self.frame_viewer = FrameViewer()
        self.frame_viewer.object_deleted.connect(self.delete_object)
        main_layout.addWidget(self.frame_viewer)
        
        # Create scrollarea with padding for bottom items
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.track_container = QWidget()
        self.scroll_area.setWidget(self.track_container)
        
        track_layout = QVBoxLayout(self.track_container)
        track_layout.setContentsMargins(0, 0, 0, 20)  # Add bottom padding
        
        self.track_viewer = TrackViewer()
        self.track_viewer.object_clicked.connect(self.highlight_object)
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
    
    def highlight_object(self, obj_id):
        self.highlighted_obj_id = obj_id
        self.display_current_frame()
    
    def display_current_frame(self):
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
        
        if self.deleted_obj_ids or (self.variant_frames and self.current_idx == len(self.master_track) - 1):
            # Apply deletions and save changes
            data = self.video_tracker.import_master_track(self.output_dir, self.video_name)
            
            if self.deleted_obj_ids:
                for frame in self.master_track:
                    for obj_id in self.deleted_obj_ids:
                        if obj_id in frame['data']:
                            del frame['data'][obj_id]
                
                data['important_frames'] = self.master_track
                data['num_objects'] -= len(self.deleted_obj_ids)
                logging.info(f"Removed {len(self.deleted_obj_ids)} objects from all frames")
            
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
        video_file = video_file[:-4]
        video_path = os.path.join(config.input_dir, video_file + ".mov")
        logging.info(f"Processing video: {video_file}")
        
        data = video_tracker.import_master_track(config.output_dir, video_file)
        if data.get('variant_frames') is not None:
            logging.info(f"Skipping {video_file} - variant frames already marked")
            continue
        
        visualizer = TrackingVisualizer(config.output_dir, video_file, video_path, video_tracker, stabilizer)
        visualizer.show()
        app.exec_()
    
    return 0
