
import sys
import os
import random
import xml.etree.ElementTree as ET
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QScrollArea
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QTimer

# Configuration
DATASET_ROOT = r"c:\Users\mr08456\Desktop\fyp\aesthetic-fault-detection\new part dataset"
OUTPUT_DIR = r"c:\Users\mr08456\Desktop\fyp\aesthetic-fault-detection\annotations_new_dataset"

class AnnotationTool(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Annotation Tool (F=Faulty, G=Good, Q=Quit)")
        self.setGeometry(100, 100, 1000, 800)

        # Image Label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setScaledContents(False) # Ensure we don't just stretch, but use pixmap size

        # Scroll Area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True) # Allow label to resize to content (pixmap)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.info_label = QLabel("Loading...", self)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")

        layout = QVBoxLayout()
        layout.addWidget(self.info_label)
        layout.addWidget(self.scroll_area) # Use scroll area
        self.setLayout(layout)

        self.tasks = []  # List of (folder_path, folder_key) tuples
        self.current_task_idx = 0
        self.current_images = []
        self.current_image_idx = 0
        self.images_data = [] # Data for current folder being processed
        self.current_folder_key = ""
        self.scale_factor = 1.0 # Default zoom level
        
        self.collect_tasks()
        self.process_next_task()

    def collect_tasks(self):
        print("Collecting tasks...")
        for root, dirs, files in os.walk(DATASET_ROOT):
            images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not images: continue
            
            folder_name = os.path.basename(root).lower()
            if folder_name in ["good1", "good2"]:
                continue

            rel_path = os.path.relpath(root, DATASET_ROOT)
            path_parts = rel_path.split(os.sep)
            folder_key = "_".join(path_parts).replace(" ", "")
            
            is_auto = "goodtest" in path_parts
            
            if is_auto:
                # Process auto-folders immediately without GUI
                self.process_auto_folder(root, folder_key, images)
            else:
                self.tasks.append((root, folder_key, images))
        
        print(f"Found {len(self.tasks)} folders requiring manual annotation.")

    def get_existing_annotations(self, folder_name):
        xml_path = os.path.join(OUTPUT_DIR, f"{folder_name}.xml")
        if not os.path.exists(xml_path):
            return set(), []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            existing_files = set()
            existing_data = []
            for image in root.findall("image"):
                name = image.get("name")
                width = image.get("width")
                height = image.get("height")
                existing_files.add(name)
                boxes = []
                for box in image.findall("box"):
                    boxes.append((float(box.get("xtl")), float(box.get("ytl")), float(box.get("xbr")), float(box.get("ybr"))))
                existing_data.append((name, width, height, boxes))
            return existing_files, existing_data
        except Exception:
            return set(), []

    def save_folder_xml(self, folder_name, images_data):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        xml_path = os.path.join(OUTPUT_DIR, f"{folder_name}.xml")
        root = ET.Element("annotations")
        ET.SubElement(root, "version").text = "1.1"
        meta = ET.SubElement(root, "meta")
        job = ET.SubElement(meta, "job")
        ET.SubElement(job, "created").text = str(datetime.now())
        labels = ET.SubElement(job, "labels")
        ET.SubElement(labels, "label").append(ET.Element("name", text="anomaly"))
        
        # Sort by ID (original order of processing)
        for i, (fname, w, h, boxes) in enumerate(images_data):
            img_elem = ET.SubElement(root, "image", id=str(i), name=fname, width=str(w), height=str(h))
            for box in boxes:
                ET.SubElement(img_elem, "box", label="anomaly", source="manual", occluded="0", 
                              xtl=f"{box[0]:.2f}", ytl=f"{box[1]:.2f}", xbr=f"{box[2]:.2f}", ybr=f"{box[3]:.2f}", z_order="0")
                
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        print(f"Saved {xml_path}")

    def process_auto_folder(self, folder_path, folder_key, images):
        existing, existing_data = self.get_existing_annotations(folder_key)
        todo = [f for f in images if f not in existing]
        if not todo: return

        print(f"Auto-processing {folder_key} ({len(todo)} images)...")
        new_data = []
        for img_name in todo:
            img_path = os.path.join(folder_path, img_name)
            pixmap = QPixmap(img_path)
            if pixmap.isNull(): continue
            new_data.append((img_name, pixmap.width(), pixmap.height(), [])) # Empty boxes for Good
        
        # Merge and save
        final_data = existing_data + new_data
        self.save_folder_xml(folder_key, final_data)

    def process_next_task(self):
        if self.current_task_idx >= len(self.tasks):
            print("All tasks completed.")
            self.close()
            return

        folder_path, folder_key, all_images = self.tasks[self.current_task_idx]
        self.current_folder_key = folder_key
        existing, self.images_data = self.get_existing_annotations(folder_key)
        
        self.current_images = [f for f in all_images if f not in existing]
        self.current_images.sort()
        
        if not self.current_images:
            self.current_task_idx += 1
            self.process_next_task()
            return
            
        self.current_image_idx = 0
        self.scale_factor = 1.0 # Reset zoom for new folder
        self.load_current_image()

    def load_current_image(self):
        if self.current_image_idx >= len(self.current_images):
            # Finish current folder
            self.save_folder_xml(self.current_folder_key, self.images_data)
            self.current_task_idx += 1
            self.process_next_task()
            return

        self.current_img_name = self.current_images[self.current_image_idx]
        self.current_img_path = os.path.join(self.tasks[self.current_task_idx][0], self.current_img_name)
        
        pixmap = QPixmap(self.current_img_path)
        if pixmap.isNull():
            print(f"Failed to load {self.current_img_name}")
            self.current_image_idx += 1
            self.load_current_image()
            return
            
        self.current_w = pixmap.width()
        self.current_h = pixmap.height()
        
        # Scale for display with Zoom
        base_w, base_h = 900, 700
        target_w = int(base_w * self.scale_factor)
        target_h = int(base_h * self.scale_factor)
        
        scaled_pixmap = pixmap.scaled(target_w, target_h, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)
        
        task_info = f"Folder: {self.current_folder_key} | Image {self.current_image_idx + 1}/{len(self.current_images)} | Zoom: {self.scale_factor:.1f}x"
        self.info_label.setText(task_info + "\n[F]aulty | [G]ood | [Z] Undo | [+/-] Zoom | [Q]uit")

    def keyPressEvent(self, event):
        if not self.current_images: return

        key = event.key()
        boxes = []
        
        if key == Qt.Key.Key_F:
            # Faulty - Random Box
            box_w = random.randint(100, 300)
            box_h = random.randint(100, 300)
            x1 = random.randint(int(self.current_w*0.2), int(self.current_w*0.8))
            y1 = random.randint(int(self.current_h*0.2), int(self.current_h*0.8))
            x2 = min(self.current_w, x1 + box_w)
            y2 = min(self.current_h, y1 + box_h)
            boxes.append((float(x1), float(y1), float(x2), float(y2)))
            print(f"Marked Faulty: {self.current_img_name}")
            
            # Save data and next
            self.images_data.append((self.current_img_name, self.current_w, self.current_h, boxes))
            self.current_image_idx += 1
            self.load_current_image()
            
        elif key == Qt.Key.Key_G:
            # Good - No Box
            # Save data and next
            self.images_data.append((self.current_img_name, self.current_w, self.current_h, boxes))
            self.current_image_idx += 1
            self.load_current_image()
            
        elif key == Qt.Key.Key_Z:
            # Undo
            if self.current_image_idx > 0:
                print(f"Undoing last image: {self.current_img_name}")
                self.current_image_idx -= 1
                if self.images_data:
                    self.images_data.pop() # Remove last annotation
                self.load_current_image()
            else:
                print("Cannot undo (start of folder).")

        elif key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
            # Zoom In
            self.scale_factor += 0.2
            self.load_current_image()
            
        elif key == Qt.Key.Key_Minus:
            # Zoom Out
            self.scale_factor = max(0.2, self.scale_factor - 0.2)
            self.load_current_image()

        elif key == Qt.Key.Key_Q:
            # Save current progress
            if self.images_data:
                self.save_folder_xml(self.current_folder_key, self.images_data)
            self.close()
            return

if __name__ == "__main__":
    app = QApplication(sys.argv)
    tool = AnnotationTool()
    tool.show()
    sys.exit(app.exec())
