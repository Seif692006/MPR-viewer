"""
3D Slicer-Style Medical Image Viewer
With Reference Lines, Dynamic Surface View, ROI Slice Selection, and Oblique View
Enhanced with Rotation, Smart ROI, AI Organ Detection, and 3D Segmentation
Updated: ROI limits now apply to ALL views (Axial, Sagittal, Coronal)
"""

import sys
import os
import numpy as np
import json
import math

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QGridLayout, QLabel, QPushButton, QToolBar, 
                            QStatusBar, QAction, QFileDialog, QMessageBox, 
                            QSlider, QGroupBox, QSpinBox, QComboBox, QDoubleSpinBox,
                            QCheckBox, QDialog, QDialogButtonBox, QTextEdit, QApplication,
                            QButtonGroup, QRadioButton, QProgressDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QPoint, QRect, QPointF, QLineF, QThread
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QMouseEvent, QBrush

# Import image loader
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from image_loader import MedicalImageLoader
except ImportError:
    print("Warning: Could not import MedicalImageLoader")
    print("Please ensure the path is correct or create a simple loader")
    class MedicalImageLoader:
        def __init__(self):
            self.image_array = None
        def load_file(self, path):
            return False, "Loader not available"
        def load_dicom_series(self, path):
            return False, "Loader not available"
        def get_slice(self, orientation, idx):
            return None
        def get_num_slices(self, orientation):
            return 0

# Check for scipy availability
SCIPY_AVAILABLE = False
try:
    from scipy import ndimage
    from scipy.ndimage import binary_erosion, binary_dilation, label
    SCIPY_AVAILABLE = True
except ImportError:
    print("Info: scipy not available. Some features will be limited.")

# Check for scikit-image
SKIMAGE_AVAILABLE = False
try:
    from skimage import measure
    from skimage.morphology import binary_closing, remove_small_objects
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Info: scikit-image not available. 3D surface generation will be limited.")


class OrganDetectionThread(QThread):
    """Thread for AI-based organ detection"""
    detection_complete = pyqtSignal(str, dict)  # organ_name, stats
    
    def __init__(self, volume, roi_start, roi_end):
        super().__init__()
        self.volume = volume
        self.roi_start = roi_start
        self.roi_end = roi_end
    
    def run(self):
        """Detect main organ in ROI"""
        try:
            roi_volume = self.volume[self.roi_start:self.roi_end+1, :, :]
            
            # Simple intensity-based organ detection
            # This is a basic implementation - you can integrate TotalSegmentator API here
            mean_intensity = np.mean(roi_volume)
            std_intensity = np.std(roi_volume)
            max_intensity = np.max(roi_volume)
            min_intensity = np.min(roi_volume)
            
            # Heuristic organ detection based on intensity ranges
            organ_name = "Unknown"
            
            if -200 < mean_intensity < 100:
                # Soft tissue range
                if 20 < mean_intensity < 60:
                    organ_name = "Liver"
                elif 30 < mean_intensity < 50:
                    organ_name = "Kidney"
                elif 10 < mean_intensity < 40:
                    organ_name = "Spleen"
                elif -50 < mean_intensity < 20:
                    organ_name = "Muscle/Soft Tissue"
                else:
                    organ_name = "Abdomen"
            elif -1000 < mean_intensity < -200:
                organ_name = "Lung"
            elif 100 < mean_intensity < 1000:
                organ_name = "Bone"
            elif mean_intensity > 30 and std_intensity < 20:
                organ_name = "Brain"
            
            stats = {
                'mean_hu': float(mean_intensity),
                'std_hu': float(std_intensity),
                'min_hu': float(min_intensity),
                'max_hu': float(max_intensity),
                'volume_size': roi_volume.shape,
                'slice_count': self.roi_end - self.roi_start + 1
            }
            
            self.detection_complete.emit(organ_name, stats)
            
        except Exception as e:
            print(f"Organ detection error: {e}")
            self.detection_complete.emit("Detection Failed", {})


class SegmentationThread(QThread):
    """Thread for 3D organ segmentation"""
    segmentation_complete = pyqtSignal(np.ndarray)  # 3D mask
    progress_update = pyqtSignal(int)
    
    def __init__(self, volume, roi_start, roi_end, organ_name):
        super().__init__()
        self.volume = volume
        self.roi_start = roi_start
        self.roi_end = roi_end
        self.organ_name = organ_name
    
    def run(self):
        """Generate 3D segmentation mask"""
        try:
            roi_volume = self.volume[self.roi_start:self.roi_end+1, :, :]
            
            self.progress_update.emit(10)
            
            # Adaptive thresholding based on organ type
            if self.organ_name == "Liver":
                lower, upper = 20, 150
            elif self.organ_name == "Kidney":
                lower, upper = 20, 200
            elif self.organ_name == "Spleen":
                lower, upper = 30, 200
            elif self.organ_name == "Lung":
                lower, upper = -1000, -400
            elif self.organ_name == "Bone":
                lower, upper = 200, 3000
            elif self.organ_name == "Brain":
                lower, upper = 0, 80
            else:
                # Auto threshold
                mean_val = np.mean(roi_volume)
                std_val = np.std(roi_volume)
                lower = mean_val - std_val
                upper = mean_val + 2 * std_val
            
            self.progress_update.emit(30)
            
            # Create binary mask
            mask = (roi_volume >= lower) & (roi_volume <= upper)
            
            self.progress_update.emit(50)
            
            if SCIPY_AVAILABLE:
                # Morphological operations to clean up mask
                mask = binary_closing(mask, structure=np.ones((3, 3, 3)))
                self.progress_update.emit(60)
                
                # Remove small objects
                labeled, num_features = label(mask)
                if num_features > 0:
                    # Keep only the largest connected component
                    sizes = np.bincount(labeled.ravel())
                    sizes[0] = 0  # Ignore background
                    largest_label = sizes.argmax()
                    mask = labeled == largest_label
                
                self.progress_update.emit(80)
                
                # Slight dilation to smooth
                mask = binary_dilation(mask, iterations=1)
            
            self.progress_update.emit(100)
            
            self.segmentation_complete.emit(mask.astype(np.uint8))
            
        except Exception as e:
            print(f"Segmentation error: {e}")
            self.segmentation_complete.emit(np.zeros((1, 1, 1), dtype=np.uint8))


class ObliqueLineHandle:
    """Handle for controlling oblique line endpoints"""
    def __init__(self, position, radius=8):
        self.position = position
        self.radius = radius
        self.is_hovered = False
        self.is_dragging = False
    
    def contains(self, point):
        """Check if point is within handle"""
        dx = point.x() - self.position.x()
        dy = point.y() - self.position.y()
        return math.sqrt(dx*dx + dy*dy) <= self.radius
    
    def draw(self, painter):
        """Draw the handle"""
        if self.is_dragging:
            painter.setBrush(QBrush(QColor(255, 100, 100, 200)))
        elif self.is_hovered:
            painter.setBrush(QBrush(QColor(255, 200, 100, 200)))
        else:
            painter.setBrush(QBrush(QColor(255, 255, 255, 150)))
        
        painter.setPen(QPen(QColor(0, 0, 0, 200), 2))
        painter.drawEllipse(self.position, self.radius, self.radius)


class SlicerViewer(QLabel):
    """3D Slicer-style viewer with reference lines"""
    
    position_changed = pyqtSignal(int, int)
    slice_changed = pyqtSignal(int)
    oblique_line_changed = pyqtSignal(QPointF, QPointF, float)
    
    def __init__(self, orientation, parent=None):
        super().__init__(parent)
        self.orientation = orientation
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #000000; border: 2px solid #2a2a2a;")
        self.setMinimumSize(400, 400)
        
        self.reference_position = [0, 0, 0]
        self.show_reference_lines = True
        
        self.show_oblique_line = False
        self.oblique_start = QPointF(100, 200)
        self.oblique_end = QPointF(300, 200)
        self.oblique_handle_start = ObliqueLineHandle(self.oblique_start)
        self.oblique_handle_end = ObliqueLineHandle(self.oblique_end)
        self.active_handle = None
        
        self.current_slice = None
        self.original_pixmap = None
        self.zoom_factor = 1.0
        self.window = 400
        self.level = 40
        
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        self.last_mouse_pos = QPoint()
        
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
    
    def set_oblique_line_visible(self, visible):
        """Show/hide oblique line"""
        self.show_oblique_line = visible
        if self.current_slice is not None:
            self.display_slice(self.current_slice)
    
    def update_oblique_line(self, start, end):
        """Update oblique line position"""
        self.oblique_start = start
        self.oblique_end = end
        self.oblique_handle_start.position = start
        self.oblique_handle_end.position = end
        if self.current_slice is not None:
            self.display_slice(self.current_slice)
        
    def set_reference_position(self, position):
        """Set reference position [axial, coronal, sagittal]"""
        self.reference_position = position
        if self.current_slice is not None:
            self.display_slice(self.current_slice)
        
    def display_slice(self, slice_array, apply_wl=True):
        """Display image with window/level and reference lines"""
        if slice_array is None:
            self.setText("No Image")
            return
        
        self.current_slice = slice_array
        
        if apply_wl and slice_array.dtype != np.uint8:
            min_val = self.level - self.window / 2
            max_val = self.level + self.window / 2
            windowed = np.clip(slice_array, min_val, max_val)
            normalized = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            if slice_array.max() > 255 or slice_array.min() < 0:
                normalized = ((slice_array - slice_array.min()) / 
                            (slice_array.max() - slice_array.min()) * 255).astype(np.uint8)
            else:
                normalized = slice_array.astype(np.uint8)
        
        height, width = normalized.shape
        q_image = QImage(normalized.data, width, height, width, QImage.Format_Grayscale8)
        self.original_pixmap = QPixmap.fromImage(q_image)
        
        scaled_size = self.original_pixmap.size() * self.zoom_factor
        scaled_pixmap = self.original_pixmap.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        if self.show_reference_lines or (self.show_oblique_line and self.orientation == 'axial'):
            scaled_pixmap = self.draw_overlays(scaled_pixmap)
        
        self.setPixmap(scaled_pixmap)
    
    def draw_overlays(self, pixmap):
        """Draw reference lines and oblique line"""
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = pixmap.width()
        height = pixmap.height()
        
        if self.show_reference_lines:
            colors = {
                'axial': QColor(255, 0, 0, 180),
                'sagittal': QColor(0, 255, 0, 180),
                'coronal': QColor(255, 255, 0, 180)
            }
            
            line_thickness = 2
            
            if self.orientation == 'axial':
                x = int(self.reference_position[2] * self.zoom_factor)
                pen = QPen(colors['sagittal'], line_thickness)
                painter.setPen(pen)
                painter.drawLine(x, 0, x, height)
                
                y = int(self.reference_position[1] * self.zoom_factor)
                pen = QPen(colors['coronal'], line_thickness)
                painter.setPen(pen)
                painter.drawLine(0, y, width, y)
                
                painter.setBrush(QBrush(QColor(255, 255, 255, 100)))
                painter.drawEllipse(x - 5, y - 5, 10, 10)
                
            elif self.orientation == 'sagittal':
                y = int(self.reference_position[0] * self.zoom_factor)
                pen = QPen(colors['axial'], line_thickness)
                painter.setPen(pen)
                painter.drawLine(0, y, width, y)
                
                x = int(self.reference_position[1] * self.zoom_factor)
                pen = QPen(colors['coronal'], line_thickness)
                painter.setPen(pen)
                painter.drawLine(x, 0, x, height)
                
                painter.setBrush(QBrush(QColor(255, 255, 255, 100)))
                painter.drawEllipse(x - 5, y - 5, 10, 10)
                
            elif self.orientation == 'coronal':
                y = int(self.reference_position[0] * self.zoom_factor)
                pen = QPen(colors['axial'], line_thickness)
                painter.setPen(pen)
                painter.drawLine(0, y, width, y)
                
                x = int(self.reference_position[2] * self.zoom_factor)
                pen = QPen(colors['sagittal'], line_thickness)
                painter.setPen(pen)
                painter.drawLine(x, 0, x, height)
                
                painter.setBrush(QBrush(QColor(255, 255, 255, 100)))
                painter.drawEllipse(x - 5, y - 5, 10, 10)
        
        if self.show_oblique_line and self.orientation == 'axial':
            start_scaled = QPointF(
                self.oblique_start.x() * self.zoom_factor,
                self.oblique_start.y() * self.zoom_factor
            )
            end_scaled = QPointF(
                self.oblique_end.x() * self.zoom_factor,
                self.oblique_end.y() * self.zoom_factor
            )
            
            pen = QPen(QColor(255, 0, 255, 220), 3)
            painter.setPen(pen)
            painter.drawLine(start_scaled, end_scaled)
            
            handle_radius = 10
            
            if self.oblique_handle_start.is_dragging:
                painter.setBrush(QBrush(QColor(255, 100, 100, 220)))
            elif self.oblique_handle_start.is_hovered:
                painter.setBrush(QBrush(QColor(255, 200, 100, 220)))
            else:
                painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
            painter.setPen(QPen(QColor(0, 0, 0, 255), 2))
            painter.drawEllipse(start_scaled, handle_radius, handle_radius)
            
            if self.oblique_handle_end.is_dragging:
                painter.setBrush(QBrush(QColor(255, 100, 100, 220)))
            elif self.oblique_handle_end.is_hovered:
                painter.setBrush(QBrush(QColor(255, 200, 100, 220)))
            else:
                painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
            painter.setPen(QPen(QColor(0, 0, 0, 255), 2))
            painter.drawEllipse(end_scaled, handle_radius, handle_radius)
            
            dx = self.oblique_end.x() - self.oblique_start.x()
            dy = self.oblique_end.y() - self.oblique_start.y()
            angle = math.degrees(math.atan2(dy, dx))
            
            mid_x = (start_scaled.x() + end_scaled.x()) / 2
            mid_y = (start_scaled.y() + end_scaled.y()) / 2
            
            font = QFont("Arial", 10, QFont.Bold)
            painter.setFont(font)
            text = f"Oblique: {angle:.1f}°"
            metrics = painter.fontMetrics()
            text_rect = metrics.boundingRect(text)
            text_rect.moveCenter(QPoint(int(mid_x + 10), int(mid_y - 20)))
            
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 180)))
            painter.drawRect(text_rect.adjusted(-3, -2, 3, 2))
            
            painter.setPen(QPen(QColor(255, 0, 255), 1))
            painter.drawText(text_rect, Qt.AlignCenter, text)
        
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)
        painter.drawText(10, 20, self.orientation.upper())
        
        painter.end()
        return pixmap
    
    def widget_to_image_coords(self, widget_pos):
        """Convert widget coordinates to image coordinates"""
        if self.original_pixmap is None:
            return None
        
        pixmap = self.pixmap()
        if pixmap is None:
            return None
        
        pixmap_rect = pixmap.rect()
        offset_x = (self.width() - pixmap_rect.width()) // 2 + self.pan_offset.x()
        offset_y = (self.height() - pixmap_rect.height()) // 2 + self.pan_offset.y()
        
        pixmap_x = widget_pos.x() - offset_x
        pixmap_y = widget_pos.y() - offset_y
        
        if not pixmap_rect.contains(int(pixmap_x), int(pixmap_y)):
            return None
        
        img_x = pixmap_x / self.zoom_factor
        img_y = pixmap_y / self.zoom_factor
        
        img_x = max(0, min(img_x, self.original_pixmap.width() - 1))
        img_y = max(0, min(img_y, self.original_pixmap.height() - 1))
        
        return QPointF(img_x, img_y)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse click"""
        if self.original_pixmap is None:
            return
        
        if event.button() == Qt.MiddleButton:
            self.is_panning = True
            self.last_mouse_pos = event.pos()
            return
        
        if self.show_oblique_line and self.orientation == 'axial' and event.button() == Qt.LeftButton:
            pixmap = self.pixmap()
            if pixmap:
                pixmap_rect = pixmap.rect()
                offset_x = (self.width() - pixmap_rect.width()) // 2 + self.pan_offset.x()
                offset_y = (self.height() - pixmap_rect.height()) // 2 + self.pan_offset.y()
                
                start_widget = QPointF(
                    self.oblique_start.x() * self.zoom_factor + offset_x,
                    self.oblique_start.y() * self.zoom_factor + offset_y
                )
                end_widget = QPointF(
                    self.oblique_end.x() * self.zoom_factor + offset_x,
                    self.oblique_end.y() * self.zoom_factor + offset_y
                )
                
                self.oblique_handle_start.position = start_widget
                self.oblique_handle_end.position = end_widget
                
                if self.oblique_handle_start.contains(event.pos()):
                    self.active_handle = self.oblique_handle_start
                    self.active_handle.is_dragging = True
                    return
                elif self.oblique_handle_end.contains(event.pos()):
                    self.active_handle = self.oblique_handle_end
                    self.active_handle.is_dragging = True
                    return
        
        if event.button() == Qt.LeftButton:
            img_coords = self.widget_to_image_coords(event.pos())
            if img_coords:
                self.position_changed.emit(int(img_coords.x()), int(img_coords.y()))
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move"""
        if self.is_panning:
            delta = event.pos() - self.last_mouse_pos
            self.pan_offset += delta
            self.last_mouse_pos = event.pos()
            if self.current_slice is not None:
                self.display_slice(self.current_slice)
            return
        
        if self.active_handle and self.active_handle.is_dragging:
            img_coords = self.widget_to_image_coords(event.pos())
            if img_coords:
                if self.active_handle == self.oblique_handle_start:
                    self.oblique_start = img_coords
                else:
                    self.oblique_end = img_coords
                
                dx = self.oblique_end.x() - self.oblique_start.x()
                dy = self.oblique_end.y() - self.oblique_start.y()
                angle = math.degrees(math.atan2(dy, dx))
                self.oblique_line_changed.emit(self.oblique_start, self.oblique_end, angle)
                
                if self.current_slice is not None:
                    self.display_slice(self.current_slice)
            return
        
        if self.show_oblique_line and self.orientation == 'axial':
            pixmap = self.pixmap()
            if pixmap:
                pixmap_rect = pixmap.rect()
                offset_x = (self.width() - pixmap_rect.width()) // 2 + self.pan_offset.x()
                offset_y = (self.height() - pixmap_rect.height()) // 2 + self.pan_offset.y()
                
                start_widget = QPointF(
                    self.oblique_start.x() * self.zoom_factor + offset_x,
                    self.oblique_start.y() * self.zoom_factor + offset_y
                )
                end_widget = QPointF(
                    self.oblique_end.x() * self.zoom_factor + offset_x,
                    self.oblique_end.y() * self.zoom_factor + offset_y
                )
                
                self.oblique_handle_start.position = start_widget
                self.oblique_handle_end.position = end_widget
                
                start_hovered = self.oblique_handle_start.contains(event.pos())
                end_hovered = self.oblique_handle_end.contains(event.pos())
                
                if start_hovered != self.oblique_handle_start.is_hovered or \
                   end_hovered != self.oblique_handle_end.is_hovered:
                    self.oblique_handle_start.is_hovered = start_hovered
                    self.oblique_handle_end.is_hovered = end_hovered
                    if self.current_slice is not None:
                        self.display_slice(self.current_slice)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release"""
        if event.button() == Qt.MiddleButton:
            self.is_panning = False
        elif event.button() == Qt.LeftButton and self.active_handle:
            self.active_handle.is_dragging = False
            self.active_handle = None
            if self.current_slice is not None:
                self.display_slice(self.current_slice)
    
    def wheelEvent(self, event):
        """Mouse wheel navigation"""
        delta = event.angleDelta().y()
        if delta > 0:
            self.slice_changed.emit(1)
        else:
            self.slice_changed.emit(-1)
    
    def set_zoom(self, zoom):
        """Set zoom level"""
        self.zoom_factor = zoom
        if self.current_slice is not None:
            self.display_slice(self.current_slice)
    
    def set_window_level(self, window, level):
        """Set window/level"""
        self.window = window
        self.level = level
        if self.current_slice is not None:
            self.display_slice(self.current_slice)
    
    def reset_pan(self):
        """Reset pan offset"""
        self.pan_offset = QPoint(0, 0)
        if self.current_slice is not None:
            self.display_slice(self.current_slice)


class ObliqueViewer(QLabel):
    """Oblique plane viewer with rotation capability"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a; border: 2px solid #2a2a2a;")
        self.setMinimumSize(400, 400)
        self.setText("Oblique View\n(Define line in Axial view)")
        self.setStyleSheet("background-color: #1a1a1a; color: #888; border: 2px solid #2a2a2a; font-size: 12pt;")
        
        self.volume_data = None
        self.oblique_start = QPointF(100, 200)
        self.oblique_end = QPointF(300, 200)
        self.current_axial_slice = 0
        self.zoom_factor = 1.0
        self.window = 400
        self.level = 40
        self.rotation_angle = 0  # Rotation angle in degrees
        
    def set_volume(self, volume):
        """Set volume data"""
        self.volume_data = volume
    
    def set_rotation(self, angle):
        """Set rotation angle"""
        self.rotation_angle = angle
        self.update_oblique(self.oblique_start, self.oblique_end, self.current_axial_slice)
    
    def update_oblique(self, start, end, axial_slice):
        """Update oblique view with new line position"""
        if self.volume_data is None:
            return
        
        self.oblique_start = start
        self.oblique_end = end
        self.current_axial_slice = axial_slice
        
        oblique_slice = self.extract_oblique_plane()
        if oblique_slice is not None:
            self.display_slice(oblique_slice)
    
    def extract_oblique_plane(self):
        """Extract oblique plane from volume with rotation"""
        if self.volume_data is None:
            return None
        
        x1, y1 = int(self.oblique_start.x()), int(self.oblique_start.y())
        x2, y2 = int(self.oblique_end.x()), int(self.oblique_end.y())
        
        line_length = int(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))
        if line_length == 0:
            return None
        
        depth, height, width = self.volume_data.shape
        
        oblique_plane = np.zeros((depth, line_length))
        
        for i in range(line_length):
            t = i / line_length
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            
            oblique_plane[:, i] = self.volume_data[:, y, x]
        
        # Apply rotation if needed
        if self.rotation_angle != 0 and SCIPY_AVAILABLE:
            oblique_plane = ndimage.rotate(oblique_plane, self.rotation_angle, reshape=False, order=1)
        
        return oblique_plane
    
    def display_slice(self, slice_array, apply_wl=True):
        """Display image with window/level and reference lines - FIXED VERSION"""
        if slice_array is None:
            self.setText("No Image")
            return
        
        self.current_slice = slice_array
        
        if apply_wl and slice_array.dtype != np.uint8:
            min_val = self.level - self.window / 2
            max_val = self.level + self.window / 2
            windowed = np.clip(slice_array, min_val, max_val)
            normalized = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            if slice_array.max() > 255 or slice_array.min() < 0:
                normalized = ((slice_array - slice_array.min()) / 
                            (slice_array.max() - slice_array.min()) * 255).astype(np.uint8)
            else:
                normalized = slice_array.astype(np.uint8)
        
        height, width = normalized.shape
        
        # CRITICAL FIX: Ensure data is C-contiguous and use proper bytes per line
        normalized = np.ascontiguousarray(normalized)
        bytes_per_line = normalized.strides[0]  # Get actual bytes per line
        
        # Create QImage with proper stride
        q_image = QImage(normalized.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        self.original_pixmap = QPixmap.fromImage(q_image)
        
        scaled_size = self.original_pixmap.size() * self.zoom_factor
        scaled_pixmap = self.original_pixmap.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        if self.show_reference_lines or (self.show_oblique_line and self.orientation == 'axial'):
            scaled_pixmap = self.draw_overlays(scaled_pixmap)
        
        self.setPixmap(scaled_pixmap)
    
    def draw_crosshair(self, pixmap):
        """Draw crosshair at current axial position"""
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        y = self.current_axial_slice
        if 0 <= y < pixmap.height():
            pen = QPen(QColor(255, 0, 0, 180), 2)
            painter.setPen(pen)
            painter.drawLine(0, y, pixmap.width(), y)
        
        painter.setPen(QPen(QColor(255, 0, 255), 1))
        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)
        if self.rotation_angle != 0:
            painter.drawText(10, 20, f"OBLIQUE (Rot: {self.rotation_angle:.0f}°)")
        else:
            painter.drawText(10, 20, "OBLIQUE")
        
        painter.end()
        return pixmap
    
    def set_zoom(self, zoom):
        """Set zoom level"""
        self.zoom_factor = zoom
        self.update_oblique(self.oblique_start, self.oblique_end, self.current_axial_slice)
    
    def set_window_level(self, window, level):
        """Set window/level"""
        self.window = window
        self.level = level
        self.update_oblique(self.oblique_start, self.oblique_end, self.current_axial_slice)


class SurfaceViewer(QLabel):
    """3D surface view showing organ segmentation"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a; border: 2px solid #2a2a2a;")
        self.setMinimumSize(400, 400)
        self.setText("3D Surface View\n(Waiting for segmentation...)")
        self.setStyleSheet("background-color: #1a1a1a; color: #666; border: 2px solid #2a2a2a; font-size: 12pt;")
        
        self.segmentation_mask = None
        self.current_slice = 0
        self.rotation_x = 0
        self.rotation_y = 0
        self.zoom_factor = 1.0
        
    def set_segmentation(self, mask):
        """Set 3D segmentation mask"""
        self.segmentation_mask = mask
        self.update_surface()
    
    def set_rotation(self, x_rot, y_rot):
        """Set rotation angles"""
        self.rotation_x = x_rot
        self.rotation_y = y_rot
        self.update_surface()
    
    def update_surface(self):
        """Update 3D surface visualization"""
        if self.segmentation_mask is None or self.segmentation_mask.sum() == 0:
            self.setText("No segmentation available")
            return
        
        # Create 3D projection
        projection = self.create_3d_projection()
        if projection is not None:
            self.display_projection(projection)
    
    def create_3d_projection(self):
        """Create 3D projection of segmentation"""
        if self.segmentation_mask is None:
            return None
        
        # Maximum intensity projection with rotation effect
        depth, height, width = self.segmentation_mask.shape
        
        # Simple rotation by selecting different projection angles
        if self.rotation_y != 0:
            # Rotate around Y axis by shifting slices
            shift = int(self.rotation_y / 10)
            if shift != 0 and SCIPY_AVAILABLE:
                rotated_mask = np.zeros_like(self.segmentation_mask)
                for i in range(depth):
                    rotated_mask[i] = ndimage.shift(self.segmentation_mask[i], [0, shift], order=0)
                mask = rotated_mask
            else:
                mask = self.segmentation_mask
        else:
            mask = self.segmentation_mask
        
        # Create edge detection for hollow surface
        edges = np.zeros_like(mask)
        
        if SCIPY_AVAILABLE:
            # Detect edges in all three dimensions
            for i in range(depth):
                if mask[i].any():
                    # XY plane edges
                    sx = ndimage.sobel(mask[i].astype(float), axis=0)
                    sy = ndimage.sobel(mask[i].astype(float), axis=1)
                    edges[i] = np.hypot(sx, sy) > 0.1
            
            # Add edges between slices (Z direction)
            for i in range(depth - 1):
                diff = np.abs(mask[i].astype(float) - mask[i+1].astype(float))
                edges[i] = np.logical_or(edges[i], diff > 0)
        else:
            # Simple edge detection
            for i in range(depth):
                if mask[i].any():
                    edges[i, :-1, :] |= (mask[i, 1:, :] != mask[i, :-1, :])
                    edges[i, :, :-1] |= (mask[i, :, 1:] != mask[i, :, :-1])
        
        # Create projection based on rotation_x
        if abs(self.rotation_x) < 30:
            # Front view (maximum intensity projection along depth)
            projection = np.max(edges, axis=0)
        elif abs(self.rotation_x) > 60:
            # Top view
            projection = np.max(edges, axis=1)
        else:
            # Mixed view
            proj1 = np.max(edges, axis=0)
            proj2 = np.max(edges, axis=1)
            # Blend based on angle
            blend = (abs(self.rotation_x) - 30) / 30
            projection = proj1 * (1 - blend) + proj2 * blend
            projection = projection > 0.5
        
        return projection.astype(np.uint8) * 255
    
    def display_projection(self, projection):
        """Display the 3D projection"""
        if projection is None:
            return
        
        # Create colored surface
        height, width = projection.shape
        rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Cyan colored surface
        mask = projection > 0
        rgb_array[mask, 0] = 0
        rgb_array[mask, 1] = 255
        rgb_array[mask, 2] = 255
        
        # Add depth shading
        if SCIPY_AVAILABLE:
            depth_map = ndimage.distance_transform_edt(mask)
            depth_normalized = np.clip(depth_map / (depth_map.max() + 1e-8), 0, 1)
            for c in range(3):
                rgb_array[:, :, c] = (rgb_array[:, :, c] * depth_normalized).astype(np.uint8)
        
        q_image = QImage(rgb_array.data, width, height, width * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Add labels
        pixmap = self.add_labels(pixmap)
        
        scaled = pixmap.scaled(
            int(pixmap.width() * self.zoom_factor),
            int(pixmap.height() * self.zoom_factor),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.setPixmap(scaled)
    
    def add_labels(self, pixmap):
        """Add labels to surface view"""
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.setPen(QPen(QColor(0, 255, 255), 1))
        font = QFont("Arial", 10, QFont.Bold)
        painter.setFont(font)
        painter.drawText(10, 20, f"3D SURFACE (X:{self.rotation_x:.0f}° Y:{self.rotation_y:.0f}°)")
        
        painter.end()
        return pixmap


class ROIDialog(QDialog):
    """Dialog for ROI slice selection"""
    
    def __init__(self, max_slices, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ROI Slice Selection")
        self.setModal(True)
        self.resize(350, 200)
        
        layout = QVBoxLayout(self)
        
        info = QLabel(f"Select slice range (Total: {max_slices} slices)")
        info.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(info)
        
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("Start Slice:"))
        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, max_slices - 1)
        self.start_spin.setValue(0)
        self.start_spin.setMinimumWidth(100)
        start_layout.addWidget(self.start_spin)
        start_layout.addStretch()
        layout.addLayout(start_layout)
        
        end_layout = QHBoxLayout()
        end_layout.addWidget(QLabel("End Slice:"))
        self.end_spin = QSpinBox()
        self.end_spin.setRange(0, max_slices - 1)
        self.end_spin.setValue(max_slices - 1)
        self.end_spin.setMinimumWidth(100)
        end_layout.addWidget(self.end_spin)
        end_layout.addStretch()
        layout.addLayout(end_layout)
        
        self.preview_label = QLabel(f"Selected: {max_slices} slices")
        self.preview_label.setStyleSheet("color: #44ff44; font-weight: bold; margin-top: 10px;")
        layout.addWidget(self.preview_label)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.start_spin.valueChanged.connect(self.update_preview)
        self.end_spin.valueChanged.connect(self.update_preview)
        
    def update_preview(self):
        """Update preview text"""
        start = self.start_spin.value()
        end = self.end_spin.value()
        count = max(0, end - start + 1)
        self.preview_label.setText(f"Selected: {count} slices (from {start} to {end})")
        
    def get_roi(self):
        """Get selected ROI range"""
        return self.start_spin.value(), self.end_spin.value()


class MainWindow(QMainWindow):
    """3D Slicer-Style Medical Viewer with Enhanced Features"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Slicer-Style Medical Viewer - Enhanced Edition")
        self.setGeometry(50, 50, 1800, 1000)
        
        self.image_loader = MedicalImageLoader()
        
        self.reference_position = [0, 0, 0]
        
        self.roi_start = 0
        self.roi_end = 0
        self.roi_active = False
        
        self.fourth_window_mode = "surface"
        
        self.detected_organ = "Unknown"
        self.organ_stats = {}
        
        self.is_playing = False
        self.playback_speed = 100
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.advance_slice)
        
        self.init_ui()
        self.create_menu_bar()
        self.create_status_bar()
        
    def init_ui(self):
        """Initialize UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel)
        
        viewer_widget = QWidget()
        viewer_layout = QGridLayout(viewer_widget)
        viewer_layout.setSpacing(3)
        
        self.axial_panel = self.create_viewer_panel("Axial (Red)", "axial")
        self.sagittal_panel = self.create_viewer_panel("Sagittal (Green)", "sagittal")
        self.coronal_panel = self.create_viewer_panel("Coronal (Yellow)", "coronal")
        self.fourth_panel = self.create_fourth_panel()
        
        viewer_layout.addWidget(self.axial_panel, 0, 0)
        viewer_layout.addWidget(self.sagittal_panel, 0, 1)
        viewer_layout.addWidget(self.coronal_panel, 1, 0)
        viewer_layout.addWidget(self.fourth_panel, 1, 1)
        
        main_layout.addWidget(viewer_widget, stretch=1)
        
    def create_viewer_panel(self, title, orientation):
        """Create viewer panel with controls"""
        panel = QGroupBox()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(3, 3, 3, 3)
        
        top_bar = QHBoxLayout()
        title_label = QLabel(f"<b>{title}</b>")
        
        if orientation == 'axial':
            title_label.setStyleSheet("color: #ff4444; font-size: 11pt;")
        elif orientation == 'sagittal':
            title_label.setStyleSheet("color: #44ff44; font-size: 11pt;")
        elif orientation == 'coronal':
            title_label.setStyleSheet("color: #ffff44; font-size: 11pt;")
            
        top_bar.addWidget(title_label)
        top_bar.addStretch()
        
        slice_info = QLabel("Slice: 0/0")
        slice_info.setObjectName(f"{orientation}_info")
        top_bar.addWidget(slice_info)
        layout.addLayout(top_bar)
        
        viewer = SlicerViewer(orientation)
        viewer.setObjectName(f"{orientation}_viewer")
        viewer.position_changed.connect(lambda x, y: self.update_reference_from_view(orientation, x, y))
        viewer.slice_changed.connect(lambda d: self.navigate_slice(orientation, d))
        
        if orientation == 'axial':
            viewer.oblique_line_changed.connect(self.update_oblique_view)
        
        layout.addWidget(viewer)
        
        controls = QHBoxLayout()
        
        zoom_label = QLabel("Zoom:")
        controls.addWidget(zoom_label)
        
        zoom_spin = QDoubleSpinBox()
        zoom_spin.setRange(0.1, 5.0)
        zoom_spin.setSingleStep(0.1)
        zoom_spin.setValue(1.0)
        zoom_spin.setPrefix("×")
        zoom_spin.setObjectName(f"{orientation}_zoom")
        zoom_spin.valueChanged.connect(lambda v: self.set_view_zoom(orientation, v))
        controls.addWidget(zoom_spin)
        
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(lambda: self.reset_single_view(orientation))
        controls.addWidget(reset_btn)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        return panel
    
    def create_fourth_panel(self):
        """Create the switchable 4th panel"""
        panel = QGroupBox()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(3, 3, 3, 3)
        
        top_bar = QHBoxLayout()
        title_label = QLabel("<b>4th Window</b>")
        title_label.setStyleSheet("color: #00ffff; font-size: 11pt;")
        top_bar.addWidget(title_label)
        top_bar.addStretch()
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["3D Surface View", "Oblique View"])
        self.mode_combo.currentIndexChanged.connect(self.switch_fourth_window_mode)
        top_bar.addWidget(QLabel("Mode:"))
        top_bar.addWidget(self.mode_combo)
        
        layout.addLayout(top_bar)
        
        self.fourth_container = QWidget()
        self.fourth_layout = QVBoxLayout(self.fourth_container)
        self.fourth_layout.setContentsMargins(0, 0, 0, 0)
        
        self.surface_viewer = SurfaceViewer()
        self.oblique_viewer = ObliqueViewer()
        
        self.fourth_layout.addWidget(self.surface_viewer)
        self.surface_viewer.setVisible(True)
        self.oblique_viewer.setVisible(False)
        
        layout.addWidget(self.fourth_container)
        
        # Controls
        controls = QHBoxLayout()
        
        # Rotation controls for both modes
        self.rotation_x_label = QLabel("Rot X:")
        controls.addWidget(self.rotation_x_label)
        
        self.rotation_x_spin = QSpinBox()
        self.rotation_x_spin.setRange(-180, 180)
        self.rotation_x_spin.setValue(0)
        self.rotation_x_spin.setSuffix("°")
        self.rotation_x_spin.valueChanged.connect(self.update_rotation)
        controls.addWidget(self.rotation_x_spin)
        
        self.rotation_y_label = QLabel("Rot Y:")
        controls.addWidget(self.rotation_y_label)
        
        self.rotation_y_spin = QSpinBox()
        self.rotation_y_spin.setRange(-180, 180)
        self.rotation_y_spin.setValue(0)
        self.rotation_y_spin.setSuffix("°")
        self.rotation_y_spin.valueChanged.connect(self.update_rotation)
        controls.addWidget(self.rotation_y_spin)
        
        # Zoom control
        self.fourth_zoom_label = QLabel("Zoom:")
        controls.addWidget(self.fourth_zoom_label)
        
        self.fourth_zoom_spin = QDoubleSpinBox()
        self.fourth_zoom_spin.setRange(0.1, 5.0)
        self.fourth_zoom_spin.setSingleStep(0.1)
        self.fourth_zoom_spin.setValue(1.0)
        self.fourth_zoom_spin.setPrefix("×")
        self.fourth_zoom_spin.valueChanged.connect(self.set_fourth_zoom)
        controls.addWidget(self.fourth_zoom_spin)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        return panel
    
    def update_rotation(self):
        """Update rotation for current mode"""
        x_rot = self.rotation_x_spin.value()
        y_rot = self.rotation_y_spin.value()
        
        if self.fourth_window_mode == "surface":
            self.surface_viewer.set_rotation(x_rot, y_rot)
        else:
            # For oblique, only use X rotation
            self.oblique_viewer.set_rotation(x_rot)
    
    def set_fourth_zoom(self, zoom):
        """Set zoom for 4th window"""
        if self.fourth_window_mode == "surface":
            self.surface_viewer.zoom_factor = zoom
            self.surface_viewer.update_surface()
        else:
            self.oblique_viewer.set_zoom(zoom)
    
    def switch_fourth_window_mode(self, index):
        """Switch between Surface and Oblique view"""
        for i in reversed(range(self.fourth_layout.count())):
            widget = self.fourth_layout.itemAt(i).widget()
            if widget:
                self.fourth_layout.removeWidget(widget)
                widget.setVisible(False)
        
        if index == 0:  # 3D Surface View
            self.fourth_window_mode = "surface"
            self.fourth_layout.addWidget(self.surface_viewer)
            self.surface_viewer.setVisible(True)
            
            # Show Y rotation for surface
            self.rotation_y_label.setVisible(True)
            self.rotation_y_spin.setVisible(True)
            
            axial_viewer = self.findChild(SlicerViewer, "axial_viewer")
            if axial_viewer:
                axial_viewer.set_oblique_line_visible(False)
            
            # Trigger segmentation if ROI is active
            if self.roi_active and self.detected_organ != "Unknown":
                self.generate_segmentation()
            
        else:  # Oblique View
            self.fourth_window_mode = "oblique"
            self.fourth_layout.addWidget(self.oblique_viewer)
            self.oblique_viewer.setVisible(True)
            
            # Hide Y rotation for oblique
            self.rotation_y_label.setVisible(False)
            self.rotation_y_spin.setVisible(False)
            
            axial_viewer = self.findChild(SlicerViewer, "axial_viewer")
            if axial_viewer:
                axial_viewer.set_oblique_line_visible(True)
            
            if self.image_loader.image_array is not None:
                self.oblique_viewer.set_volume(self.image_loader.image_array)
                self.update_oblique_view(
                    axial_viewer.oblique_start,
                    axial_viewer.oblique_end,
                    0
                )
    
    def update_oblique_view(self, start, end, angle):
        """Update oblique view when line changes"""
        if self.fourth_window_mode == "oblique" and self.image_loader.image_array is not None:
            self.oblique_viewer.update_oblique(start, end, self.reference_position[0])
            
            self.statusBar().showMessage(
                f"Oblique line: angle={angle:.1f}°, "
                f"start=({start.x():.0f}, {start.y():.0f}), "
                f"end=({end.x():.0f}, {end.y():.0f})"
            )
    
    def create_control_panel(self):
        """Create control panel"""
        panel = QWidget()
        panel.setMaximumWidth(300)
        layout = QVBoxLayout(panel)
        
        # File group
        file_group = QGroupBox("📁 File Operations")
        file_layout = QVBoxLayout()
        
        open_btn = QPushButton("Open DICOM/NIfTI")
        open_btn.clicked.connect(self.open_file)
        file_layout.addWidget(open_btn)
        
        open_series_btn = QPushButton("Open DICOM Series")
        open_series_btn.clicked.connect(self.open_dicom_series)
        file_layout.addWidget(open_series_btn)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # AI Organ Detection group
        ai_group = QGroupBox("🤖 AI Organ Detection")
        ai_layout = QVBoxLayout()
        
        detect_btn = QPushButton("Detect Main Organ")
        detect_btn.clicked.connect(self.detect_organ)
        ai_layout.addWidget(detect_btn)
        
        self.organ_label = QLabel("Organ: Not Detected")
        self.organ_label.setStyleSheet("color: #888; font-weight: bold;")
        ai_layout.addWidget(self.organ_label)
        
        self.organ_stats_label = QLabel("")
        self.organ_stats_label.setStyleSheet("color: #aaa; font-size: 9pt;")
        self.organ_stats_label.setWordWrap(True)
        ai_layout.addWidget(self.organ_stats_label)
        
        segment_btn = QPushButton("Generate 3D Segmentation")
        segment_btn.clicked.connect(self.generate_segmentation)
        ai_layout.addWidget(segment_btn)
        
        ai_group.setLayout(ai_layout)
        layout.addWidget(ai_group)
        
        # ROI group
        roi_group = QGroupBox("📐 ROI Slice Selection")
        roi_layout = QVBoxLayout()
        
        roi_select_btn = QPushButton("Select ROI Slices")
        roi_select_btn.clicked.connect(self.select_roi)
        roi_layout.addWidget(roi_select_btn)
        
        self.roi_status = QLabel("ROI: Not Set")
        self.roi_status.setStyleSheet("color: #888;")
        roi_layout.addWidget(self.roi_status)
        
        roi_btns = QHBoxLayout()
        apply_roi_btn = QPushButton("Apply ROI")
        apply_roi_btn.clicked.connect(self.apply_roi)
        roi_btns.addWidget(apply_roi_btn)
        
        clear_roi_btn = QPushButton("Clear ROI")
        clear_roi_btn.clicked.connect(self.clear_roi)
        roi_btns.addWidget(clear_roi_btn)
        roi_layout.addLayout(roi_btns)
        
        export_roi_btn = QPushButton("Export ROI Volume")
        export_roi_btn.clicked.connect(self.export_roi)
        roi_layout.addWidget(export_roi_btn)
        
        roi_group.setLayout(roi_layout)
        layout.addWidget(roi_group)
        
        # Navigation
        nav_group = QGroupBox("🎮 Navigation")
        nav_layout = QVBoxLayout()
        
        nav_layout.addWidget(QLabel("Axial (Red):"))
        self.axial_slider = QSlider(Qt.Horizontal)
        self.axial_slider.valueChanged.connect(lambda v: self.set_slice('axial', v))
        nav_layout.addWidget(self.axial_slider)
        
        nav_layout.addWidget(QLabel("Sagittal (Green):"))
        self.sagittal_slider = QSlider(Qt.Horizontal)
        self.sagittal_slider.valueChanged.connect(lambda v: self.set_slice('sagittal', v))
        nav_layout.addWidget(self.sagittal_slider)
        
        nav_layout.addWidget(QLabel("Coronal (Yellow):"))
        self.coronal_slider = QSlider(Qt.Horizontal)
        self.coronal_slider.valueChanged.connect(lambda v: self.set_slice('coronal', v))
        nav_layout.addWidget(self.coronal_slider)
        
        nav_group.setLayout(nav_layout)
        layout.addWidget(nav_group)
        
        # Playback
        playback_group = QGroupBox("▶ Playback")
        playback_layout = QVBoxLayout()
        
        btns = QHBoxLayout()
        self.play_btn = QPushButton("▶")
        self.play_btn.clicked.connect(self.play_slices)
        btns.addWidget(self.play_btn)
        
        self.pause_btn = QPushButton("⏸")
        self.pause_btn.clicked.connect(self.pause_slices)
        self.pause_btn.setEnabled(False)
        btns.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("⏹")
        self.stop_btn.clicked.connect(self.stop_slices)
        self.stop_btn.setEnabled(False)
        btns.addWidget(self.stop_btn)
        playback_layout.addLayout(btns)
        
        self.speed_spinbox = QSpinBox()
        self.speed_spinbox.setRange(10, 1000)
        self.speed_spinbox.setValue(100)
        self.speed_spinbox.setSuffix(" ms")
        self.speed_spinbox.valueChanged.connect(lambda v: setattr(self, 'playback_speed', v))
        playback_layout.addWidget(QLabel("Speed:"))
        playback_layout.addWidget(self.speed_spinbox)
        
        playback_group.setLayout(playback_layout)
        layout.addWidget(playback_group)
        
        # Window/Level
        wl_group = QGroupBox("🎨 Window/Level")
        wl_layout = QVBoxLayout()
        
        self.window_slider = QSlider(Qt.Horizontal)
        self.window_slider.setRange(1, 2000)
        self.window_slider.setValue(400)
        self.window_slider.valueChanged.connect(self.update_window_level)
        wl_layout.addWidget(QLabel("Window:"))
        wl_layout.addWidget(self.window_slider)
        
        self.level_slider = QSlider(Qt.Horizontal)
        self.level_slider.setRange(-1000, 1000)
        self.level_slider.setValue(40)
        self.level_slider.valueChanged.connect(self.update_window_level)
        wl_layout.addWidget(QLabel("Level:"))
        wl_layout.addWidget(self.level_slider)
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Custom", "Brain", "Bone", "Lung", "Abdomen"])
        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        wl_layout.addWidget(QLabel("Presets:"))
        wl_layout.addWidget(self.preset_combo)
        
        wl_group.setLayout(wl_layout)
        layout.addWidget(wl_group)
        
        # Reference lines
        ref_group = QGroupBox("➕ Reference Lines")
        ref_layout = QVBoxLayout()
        
        self.show_ref_checkbox = QCheckBox("Show Reference Lines")
        self.show_ref_checkbox.setChecked(True)
        self.show_ref_checkbox.stateChanged.connect(self.toggle_reference_lines)
        ref_layout.addWidget(self.show_ref_checkbox)
        
        ref_group.setLayout(ref_layout)
        layout.addWidget(ref_group)
        
        layout.addStretch()
        return panel
    
    def detect_organ(self):
        """Detect main organ in current ROI"""
        if self.image_loader.image_array is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        
        if not self.roi_active:
            QMessageBox.warning(self, "No ROI", "Please select and apply an ROI first.")
            return
        
        # Show progress
        progress = QProgressDialog("Detecting organ...", None, 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(30)
        
        # Start detection thread
        self.detection_thread = OrganDetectionThread(
            self.image_loader.image_array,
            self.roi_start,
            self.roi_end
        )
        self.detection_thread.detection_complete.connect(self.on_organ_detected)
        self.detection_thread.finished.connect(lambda: progress.setValue(100))
        self.detection_thread.start()
        
        progress.setValue(50)
        
    def on_organ_detected(self, organ_name, stats):
        """Handle organ detection completion"""
        self.detected_organ = organ_name
        self.organ_stats = stats
        
        self.organ_label.setText(f"Organ: {organ_name}")
        self.organ_label.setStyleSheet("color: #44ff44; font-weight: bold;")
        
        if stats:
            stats_text = (f"Mean HU: {stats.get('mean_hu', 0):.1f}\n"
                         f"Std HU: {stats.get('std_hu', 0):.1f}\n"
                         f"Slices: {stats.get('slice_count', 0)}")
            self.organ_stats_label.setText(stats_text)
        
        self.statusBar().showMessage(f"Detected: {organ_name}")
        
        QMessageBox.information(self, "Organ Detected", 
                               f"Main organ detected: {organ_name}\n\n"
                               f"You can now generate 3D segmentation.")
    
    def generate_segmentation(self):
        """Generate 3D segmentation of detected organ"""
        if self.image_loader.image_array is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        
        if not self.roi_active:
            QMessageBox.warning(self, "No ROI", "Please select and apply an ROI first.")
            return
        
        if self.detected_organ == "Unknown":
            reply = QMessageBox.question(
                self, "No Organ Detected",
                "No organ has been detected yet. Would you like to detect it first?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.detect_organ()
            return
        
        # Switch to surface view if not already
        if self.fourth_window_mode != "surface":
            self.mode_combo.setCurrentIndex(0)
        
        # Show progress dialog
        self.seg_progress = QProgressDialog("Generating 3D segmentation...", "Cancel", 0, 100, self)
        self.seg_progress.setWindowModality(Qt.WindowModal)
        self.seg_progress.setMinimumDuration(0)
        
        # Start segmentation thread
        self.segmentation_thread = SegmentationThread(
            self.image_loader.image_array,
            self.roi_start,
            self.roi_end,
            self.detected_organ
        )
        self.segmentation_thread.segmentation_complete.connect(self.on_segmentation_complete)
        self.segmentation_thread.progress_update.connect(self.seg_progress.setValue)
        self.segmentation_thread.finished.connect(self.seg_progress.close)
        
        self.segmentation_thread.start()
    
    def on_segmentation_complete(self, mask):
        """Handle segmentation completion"""
        if mask.sum() == 0:
            QMessageBox.warning(self, "Segmentation Failed", 
                               "Could not generate segmentation. Try adjusting ROI.")
            return
        
        self.surface_viewer.set_segmentation(mask)
        
        self.statusBar().showMessage(
            f"3D segmentation complete for {self.detected_organ}. "
            f"Use rotation controls to view."
        )
        
        QMessageBox.information(self, "Segmentation Complete",
                               f"3D surface generated for {self.detected_organ}.\n\n"
                               f"Use the rotation controls (Rot X, Rot Y) to view from different angles.")
    
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("File")
        open_action = QAction("Open", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        export_action = QAction("Export ROI", self)
        export_action.triggered.connect(self.export_roi)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        view_menu = menubar.addMenu("View")
        reset_action = QAction("Reset All Views", self)
        reset_action.setShortcut("Ctrl+R")
        reset_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_action)
        
        view_menu.addSeparator()
        surface_action = QAction("Switch to 3D Surface", self)
        surface_action.triggered.connect(lambda: self.mode_combo.setCurrentIndex(0))
        view_menu.addAction(surface_action)
        
        oblique_action = QAction("Switch to Oblique View", self)
        oblique_action.triggered.connect(lambda: self.mode_combo.setCurrentIndex(1))
        view_menu.addAction(oblique_action)
        
        ai_menu = menubar.addMenu("AI Tools")
        detect_action = QAction("Detect Organ", self)
        detect_action.triggered.connect(self.detect_organ)
        ai_menu.addAction(detect_action)
        
        segment_action = QAction("Generate 3D Segmentation", self)
        segment_action.triggered.connect(self.generate_segmentation)
        ai_menu.addAction(segment_action)
    
    def create_status_bar(self):
        """Create status bar"""
        self.statusBar().showMessage("Ready - Load an image and select ROI to begin")
    
    def update_reference_from_view(self, orientation, x, y):
        """Update reference position from view click"""
        if self.image_loader.image_array is None:
            return
        
        if orientation == 'axial':
            self.reference_position[1] = y
            self.reference_position[2] = x
            
        elif orientation == 'sagittal':
            self.reference_position[0] = y
            self.reference_position[1] = x
            
        elif orientation == 'coronal':
            self.reference_position[0] = y
            self.reference_position[2] = x
        
        self.update_all_reference_lines()
        
        if self.fourth_window_mode == "oblique":
            axial_viewer = self.findChild(SlicerViewer, "axial_viewer")
            if axial_viewer:
                self.oblique_viewer.update_oblique(
                    axial_viewer.oblique_start,
                    axial_viewer.oblique_end,
                    self.reference_position[0]
                )
        
        self.statusBar().showMessage(
            f"Reference: Axial={self.reference_position[0]}, "
            f"Coronal={self.reference_position[1]}, "
            f"Sagittal={self.reference_position[2]}"
        )
    
    def update_all_reference_lines(self):
        """Update reference lines in all views"""
        for orientation in ['axial', 'sagittal', 'coronal']:
            viewer = self.findChild(SlicerViewer, f"{orientation}_viewer")
            if viewer:
                viewer.set_reference_position(self.reference_position)
    
    def toggle_reference_lines(self, state):
        """Toggle reference lines visibility"""
        show = state == Qt.Checked
        for orientation in ['axial', 'sagittal', 'coronal']:
            viewer = self.findChild(SlicerViewer, f"{orientation}_viewer")
            if viewer:
                viewer.show_reference_lines = show
                if viewer.current_slice is not None:
                    viewer.display_slice(viewer.current_slice)
    
    def navigate_slice(self, orientation, direction):
        """Navigate slices with mouse wheel"""
        if self.image_loader.image_array is None:
            return
        
        if orientation == 'axial':
            new_slice = self.reference_position[0] + direction
            
            # Enforce ROI limits if active
            if self.roi_active:
                new_slice = max(self.roi_start, min(new_slice, self.roi_end))
            else:
                max_slices = self.image_loader.get_num_slices(orientation)
                new_slice = max(0, min(new_slice, max_slices - 1))
            
            self.reference_position[0] = new_slice
            self.axial_slider.blockSignals(True)
            self.axial_slider.setValue(new_slice)
            self.axial_slider.blockSignals(False)
            self.update_axial_view()
            
        elif orientation == 'sagittal':
            new_slice = self.reference_position[2] + direction
            
            # Enforce ROI limits if active
            if self.roi_active and self.image_loader.image_array is not None:
                roi_volume = self.image_loader.image_array[self.roi_start:self.roi_end+1, :, :]
                max_slices = roi_volume.shape[2]
            else:
                max_slices = self.image_loader.get_num_slices(orientation)
            
            new_slice = max(0, min(new_slice, max_slices - 1))
            self.reference_position[2] = new_slice
            self.sagittal_slider.blockSignals(True)
            self.sagittal_slider.setValue(new_slice)
            self.sagittal_slider.blockSignals(False)
            self.update_sagittal_view()
            
        elif orientation == 'coronal':
            new_slice = self.reference_position[1] + direction
            
            # Enforce ROI limits if active
            if self.roi_active and self.image_loader.image_array is not None:
                roi_volume = self.image_loader.image_array[self.roi_start:self.roi_end+1, :, :]
                max_slices = roi_volume.shape[1]
            else:
                max_slices = self.image_loader.get_num_slices(orientation)
            
            new_slice = max(0, min(new_slice, max_slices - 1))
            self.reference_position[1] = new_slice
            self.coronal_slider.blockSignals(True)
            self.coronal_slider.setValue(new_slice)
            self.coronal_slider.blockSignals(False)
            self.update_coronal_view()
        
        self.update_all_reference_lines()
        
        if self.fourth_window_mode == "oblique":
            axial_viewer = self.findChild(SlicerViewer, "axial_viewer")
            if axial_viewer:
                self.oblique_viewer.update_oblique(
                    axial_viewer.oblique_start,
                    axial_viewer.oblique_end,
                    self.reference_position[0]
                )
    
    def set_slice(self, orientation, value):
        """Set slice from slider"""
        # Enforce ROI limits for axial if active
        if orientation == 'axial' and self.roi_active:
            value = max(self.roi_start, min(value, self.roi_end))
        
        if orientation == 'axial':
            self.reference_position[0] = value
            self.update_axial_view()
        elif orientation == 'sagittal':
            self.reference_position[2] = value
            self.update_sagittal_view()
        elif orientation == 'coronal':
            self.reference_position[1] = value
            self.update_coronal_view()
        
        self.update_all_reference_lines()
        
        if self.fourth_window_mode == "oblique":
            axial_viewer = self.findChild(SlicerViewer, "axial_viewer")
            if axial_viewer:
                self.oblique_viewer.update_oblique(
                    axial_viewer.oblique_start,
                    axial_viewer.oblique_end,
                    self.reference_position[0]
                )
    
    def set_view_zoom(self, orientation, zoom):
        """Set zoom for specific view"""
        viewer = self.findChild(SlicerViewer, f"{orientation}_viewer")
        if viewer:
            viewer.set_zoom(zoom)
    
    def reset_single_view(self, orientation):
        """Reset single view"""
        viewer = self.findChild(SlicerViewer, f"{orientation}_viewer")
        if viewer:
            viewer.zoom_factor = 1.0
            viewer.reset_pan()
            zoom_spin = self.findChild(QDoubleSpinBox, f"{orientation}_zoom")
            if zoom_spin:
                zoom_spin.setValue(1.0)
    
    def update_window_level(self):
        """Update window/level for all views"""
        window = self.window_slider.value()
        level = self.level_slider.value()
        
        for orientation in ['axial', 'sagittal', 'coronal']:
            viewer = self.findChild(SlicerViewer, f"{orientation}_viewer")
            if viewer:
                viewer.set_window_level(window, level)
        
        self.oblique_viewer.set_window_level(window, level)
    
    def apply_preset(self, preset):
        """Apply window/level preset"""
        presets = {
            "Brain": (80, 40),
            "Bone": (2000, 300),
            "Lung": (1500, -600),
            "Abdomen": (400, 40)
        }
        
        if preset in presets:
            w, l = presets[preset]
            self.window_slider.blockSignals(True)
            self.level_slider.blockSignals(True)
            self.window_slider.setValue(w)
            self.level_slider.setValue(l)
            self.window_slider.blockSignals(False)
            self.level_slider.blockSignals(False)
            self.update_window_level()
    
    def select_roi(self):
        """Open ROI selection dialog"""
        if self.image_loader.image_array is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        
        max_slices = self.image_loader.get_num_slices('axial')
        dialog = ROIDialog(max_slices, self)
        
        if self.roi_active:
            dialog.start_spin.setValue(self.roi_start)
            dialog.end_spin.setValue(self.roi_end)
        
        if dialog.exec_() == QDialog.Accepted:
            self.roi_start, self.roi_end = dialog.get_roi()
            count = self.roi_end - self.roi_start + 1
            self.roi_status.setText(f"ROI: Slices {self.roi_start}-{self.roi_end} ({count} slices)")
            self.roi_status.setStyleSheet("color: #ffaa00;")
            self.statusBar().showMessage(f"ROI selected: {count} slices (not yet applied)")
    
    def apply_roi(self):
        """Apply ROI to limit view range"""
        if self.roi_start >= self.roi_end:
            QMessageBox.warning(self, "Invalid ROI", "Please select a valid ROI first.")
            return
        
        self.roi_active = True
        
        # Update all slider ranges to ROI limits
        # Axial slider is limited by ROI start/end
        self.axial_slider.setMinimum(self.roi_start)
        self.axial_slider.setMaximum(self.roi_end)
        
        # Get ROI volume dimensions for other views
        if self.image_loader.image_array is not None:
            roi_volume = self.image_loader.image_array[self.roi_start:self.roi_end+1, :, :]
            _, height, width = roi_volume.shape
            
            # Update coronal and sagittal sliders to match ROI dimensions
            self.coronal_slider.setMaximum(height - 1)
            self.sagittal_slider.setMaximum(width - 1)
        
        # Clamp current position to ROI
        if self.reference_position[0] < self.roi_start:
            self.reference_position[0] = self.roi_start
        elif self.reference_position[0] > self.roi_end:
            self.reference_position[0] = self.roi_end
        
        # Clamp coronal and sagittal positions
        if self.image_loader.image_array is not None:
            if self.reference_position[1] >= height:
                self.reference_position[1] = height - 1
            if self.reference_position[2] >= width:
                self.reference_position[2] = width - 1
        
        self.axial_slider.setValue(self.reference_position[0])
        self.coronal_slider.setValue(self.reference_position[1])
        self.sagittal_slider.setValue(self.reference_position[2])
        
        # Update ROI status
        count = self.roi_end - self.roi_start + 1
        self.roi_status.setText(f"ROI: Active ({self.roi_start}-{self.roi_end}, {count} slices)")
        self.roi_status.setStyleSheet("color: #44ff44;")
        
        self.display_all_views()
        
        QMessageBox.information(self, "ROI Applied", 
                               f"Navigation limited to ROI region:\n"
                               f"Axial slices: {self.roi_start}-{self.roi_end} ({count} slices)\n\n"
                               f"All views now show only the ROI volume.\n"
                               f"You can now detect the main organ in this region.")
    
    def clear_roi(self):
        """Clear ROI selection"""
        if self.image_loader.image_array is None:
            return
        
        self.roi_active = False
        self.roi_start = 0
        self.roi_end = self.image_loader.get_num_slices('axial') - 1
        
        # Reset all slider ranges to full volume
        self.axial_slider.setMinimum(0)
        self.axial_slider.setMaximum(self.roi_end)
        
        sagittal_max = self.image_loader.get_num_slices('sagittal') - 1
        coronal_max = self.image_loader.get_num_slices('coronal') - 1
        self.sagittal_slider.setMaximum(sagittal_max)
        self.coronal_slider.setMaximum(coronal_max)
        
        self.roi_status.setText("ROI: Not Set")
        self.roi_status.setStyleSheet("color: #888;")
        
        # Clear detection
        self.detected_organ = "Unknown"
        self.organ_label.setText("Organ: Not Detected")
        self.organ_label.setStyleSheet("color: #888; font-weight: bold;")
        self.organ_stats_label.setText("")
        
        # Clear segmentation
        self.surface_viewer.set_segmentation(None)
        
        # Refresh all views
        self.display_all_views()
        
        self.statusBar().showMessage("ROI cleared - Full volume restored")
    
    def export_roi(self):
        """Export ROI volume"""
        if self.image_loader.image_array is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        
        if not self.roi_active or self.roi_start >= self.roi_end:
            QMessageBox.warning(self, "No ROI", "Please select and apply an ROI first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export ROI Volume", "",
            "NumPy Array (*.npy);;NIfTI Files (*.nii *.nii.gz)"
        )
        
        if not file_path:
            return
        
        try:
            roi_volume = self.image_loader.image_array[self.roi_start:self.roi_end+1, :, :]
            
            if file_path.endswith('.npy'):
                np.save(file_path, roi_volume)
            else:
                try:
                    import nibabel as nib
                    nii_img = nib.Nifti1Image(roi_volume, np.eye(4))
                    nib.save(nii_img, file_path)
                except ImportError:
                    QMessageBox.warning(self, "Missing Library", 
                                      "nibabel not installed. Saving as .npy instead.")
                    file_path = file_path.rsplit('.', 1)[0] + '.npy'
                    np.save(file_path, roi_volume)
            
            metadata = {
                'roi_start': self.roi_start,
                'roi_end': self.roi_end,
                'original_shape': list(self.image_loader.image_array.shape),
                'roi_shape': list(roi_volume.shape),
                'window': self.window_slider.value(),
                'level': self.level_slider.value(),
                'detected_organ': self.detected_organ,
                'organ_stats': self.organ_stats
            }
            
            metadata_path = file_path + '.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            QMessageBox.information(self, "Export Successful", 
                                   f"ROI volume exported:\n{file_path}\n\n"
                                   f"Detected Organ: {self.detected_organ}\n"
                                   f"Slices: {self.roi_start}-{self.roi_end}\n"
                                   f"Shape: {roi_volume.shape}\n"
                                   f"Metadata saved to: {metadata_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export ROI:\n{str(e)}")
    
    def play_slices(self):
        """Start playback"""
        if self.image_loader.image_array is None:
            return
        
        self.is_playing = True
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.playback_timer.start(self.playback_speed)
    
    def pause_slices(self):
        """Pause playback"""
        self.is_playing = False
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.playback_timer.stop()
    
    def stop_slices(self):
        """Stop playback"""
        self.is_playing = False
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.playback_timer.stop()
    
    def advance_slice(self):
        """Advance to next slice during playback"""
        self.navigate_slice('axial', 1)
        
        # Loop back if at end (respecting ROI limits)
        max_slice = self.roi_end if self.roi_active else self.image_loader.get_num_slices('axial') - 1
        if self.reference_position[0] >= max_slice:
            min_slice = self.roi_start if self.roi_active else 0
            self.reference_position[0] = min_slice
            self.axial_slider.setValue(min_slice)
    
    def reset_view(self):
        """Reset all views to center"""
        if self.image_loader.image_array is None:
            return
        
        if self.roi_active:
            center_slice = (self.roi_start + self.roi_end) // 2
        else:
            center_slice = self.image_loader.get_num_slices('axial') // 2
        
        self.reference_position[0] = center_slice
        self.reference_position[1] = self.image_loader.get_num_slices('coronal') // 2
        self.reference_position[2] = self.image_loader.get_num_slices('sagittal') // 2
        
        for orientation in ['axial', 'sagittal', 'coronal']:
            self.reset_single_view(orientation)
        
        # Reset rotations
        self.rotation_x_spin.setValue(0)
        self.rotation_y_spin.setValue(0)
        
        self.display_all_views()
    
    def open_file(self):
        """Open medical image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Medical Image", "",
            "Medical Images (*.dcm *.nii *.nii.gz);;All Files (*.*)"
        )
        if file_path:
            self.load_image(file_path)
    
    def open_dicom_series(self):
        """Open DICOM series"""
        directory = QFileDialog.getExistingDirectory(self, "Select DICOM Series", "")
        if directory:
            self.load_dicom_series(directory)
    
    def load_image(self, file_path):
        """Load image file"""
        success, message = self.image_loader.load_file(file_path)
        if success:
            self.initialize_after_load()
            self.statusBar().showMessage(message)
        else:
            QMessageBox.critical(self, "Error", message)
    
    def load_dicom_series(self, directory):
        """Load DICOM series"""
        success, message = self.image_loader.load_dicom_series(directory)
        if success:
            self.initialize_after_load()
            self.statusBar().showMessage(message)
        else:
            QMessageBox.critical(self, "Error", message)
    
    def initialize_after_load(self):
        """Initialize UI after loading image"""
        axial_max = self.image_loader.get_num_slices('axial') - 1
        sagittal_max = self.image_loader.get_num_slices('sagittal') - 1
        coronal_max = self.image_loader.get_num_slices('coronal') - 1
        
        self.axial_slider.setMaximum(axial_max)
        self.sagittal_slider.setMaximum(sagittal_max)
        self.coronal_slider.setMaximum(coronal_max)
        
        # Reset ROI and detection
        self.roi_start = 0
        self.roi_end = axial_max
        self.roi_active = False
        self.roi_status.setText("ROI: Not Set")
        self.roi_status.setStyleSheet("color: #888;")
        
        self.detected_organ = "Unknown"
        self.organ_label.setText("Organ: Not Detected")
        self.organ_label.setStyleSheet("color: #888; font-weight: bold;")
        self.organ_stats_label.setText("")
        
        self.reference_position = [
            axial_max // 2,
            coronal_max // 2,
            sagittal_max // 2
        ]
        
        axial_viewer = self.findChild(SlicerViewer, "axial_viewer")
        if axial_viewer:
            width = self.image_loader.image_array.shape[2]
            height = self.image_loader.image_array.shape[1]
            axial_viewer.oblique_start = QPointF(width * 0.25, height * 0.5)
            axial_viewer.oblique_end = QPointF(width * 0.75, height * 0.5)
        
        self.oblique_viewer.set_volume(self.image_loader.image_array)
        
        self.display_all_views()
        self.setup_sliders()
    
    def setup_sliders(self):
        """Setup slider positions"""
        self.axial_slider.setValue(self.reference_position[0])
        self.sagittal_slider.setValue(self.reference_position[2])
        self.coronal_slider.setValue(self.reference_position[1])
    
    def display_all_views(self):
        """Display all views"""
        if self.image_loader.image_array is None:
            return
        
        self.update_axial_view()
        self.update_sagittal_view()
        self.update_coronal_view()
        self.update_all_reference_lines()
        
        if self.fourth_window_mode == "oblique":
            axial_viewer = self.findChild(SlicerViewer, "axial_viewer")
            if axial_viewer:
                self.oblique_viewer.update_oblique(
                    axial_viewer.oblique_start,
                    axial_viewer.oblique_end,
                    self.reference_position[0]
                )
    
    def update_axial_view(self):
        """Update axial view"""
        slice_idx = self.reference_position[0]
        
        # Get slice from ROI volume if active
        if self.roi_active:
            slice_data = self.image_loader.get_slice('axial', slice_idx)
        else:
            slice_data = self.image_loader.get_slice('axial', slice_idx)
        
        viewer = self.findChild(SlicerViewer, "axial_viewer")
        if viewer:
            viewer.display_slice(slice_data)
        
        info = self.findChild(QLabel, "axial_info")
        if info:
            if self.roi_active:
                total = self.roi_end - self.roi_start + 1
                relative_idx = slice_idx - self.roi_start
                info.setText(f"Slice: {relative_idx + 1}/{total} (Abs: {slice_idx + 1})")
            else:
                total = self.image_loader.get_num_slices('axial')
                info.setText(f"Slice: {slice_idx + 1}/{total}")
    
    def update_sagittal_view(self):
        """Update sagittal view"""
        slice_idx = self.reference_position[2]
        
        # Get slice from ROI volume if active
        if self.roi_active and self.image_loader.image_array is not None:
            roi_volume = self.image_loader.image_array[self.roi_start:self.roi_end+1, :, :]
            # Sagittal slice from ROI volume
            if slice_idx < roi_volume.shape[2]:
                slice_data = roi_volume[:, :, slice_idx]
            else:
                slice_data = None
        else:
            slice_data = self.image_loader.get_slice('sagittal', slice_idx)
        
        viewer = self.findChild(SlicerViewer, "sagittal_viewer")
        if viewer and slice_data is not None:
            viewer.display_slice(slice_data)
        
        info = self.findChild(QLabel, "sagittal_info")
        if info:
            if self.roi_active and self.image_loader.image_array is not None:
                total = self.image_loader.image_array.shape[2]
                info.setText(f"Slice: {slice_idx + 1}/{total} (ROI Mode)")
            else:
                total = self.image_loader.get_num_slices('sagittal')
                info.setText(f"Slice: {slice_idx + 1}/{total}")
    
    def update_coronal_view(self):
        """Update coronal view"""
        slice_idx = self.reference_position[1]
        
        # Get slice from ROI volume if active
        if self.roi_active and self.image_loader.image_array is not None:
            roi_volume = self.image_loader.image_array[self.roi_start:self.roi_end+1, :, :]
            # Coronal slice from ROI volume
            if slice_idx < roi_volume.shape[1]:
                slice_data = roi_volume[:, slice_idx, :]
            else:
                slice_data = None
        else:
            slice_data = self.image_loader.get_slice('coronal', slice_idx)
        
        viewer = self.findChild(SlicerViewer, "coronal_viewer")
        if viewer and slice_data is not None:
            viewer.display_slice(slice_data)
        
        info = self.findChild(QLabel, "coronal_info")
        if info:
            if self.roi_active and self.image_loader.image_array is not None:
                total = self.image_loader.image_array.shape[1]
                info.setText(f"Slice: {slice_idx + 1}/{total} (ROI Mode)")
            else:
                total = self.image_loader.get_num_slices('coronal')
                info.setText(f"Slice: {slice_idx + 1}/{total}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set dark theme
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QGroupBox {
            border: 1px solid #444;
            border-radius: 5px;
            margin-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #3a3a3a;
            border: 1px solid #555;
            padding: 5px;
            border-radius: 3px;
        }
        QPushButton:hover {
            background-color: #4a4a4a;
        }
        QPushButton:pressed {
            background-color: #2a2a2a;
        }
        QPushButton:disabled {
            background-color: #252525;
            color: #666;
        }
        QSlider::groove:horizontal {
            border: 1px solid #999;
            height: 8px;
            background: #3a3a3a;
            margin: 2px 0;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #5a5a5a;
            border: 1px solid #777;
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }
        QComboBox, QSpinBox, QDoubleSpinBox {
            background-color: #3a3a3a;
            border: 1px solid #555;
            padding: 3px;
            border-radius: 3px;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid #aaa;
            margin-right: 5px;
        }
        QStatusBar {
            background-color: #1a1a1a;
            color: #aaa;
        }
        QMenuBar {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QMenuBar::item:selected {
            background-color: #3a3a3a;
        }
        QMenu {
            background-color: #2b2b2b;
            color: #ffffff;
            border: 1px solid #555;
        }
        QMenu::item:selected {
            background-color: #3a3a3a;
        }
        QCheckBox {
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 1px solid #555;
            border-radius: 3px;
            background-color: #3a3a3a;
        }
        QCheckBox::indicator:checked {
            background-color: #4a9eff;
            border-color: #4a9eff;
        }
        QProgressDialog {
            background-color: #2b2b2b;
        }
    """)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())