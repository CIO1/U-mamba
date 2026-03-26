# -*- coding: utf-8 -*-
import os
import sys

# CRITICAL: Set locale BEFORE importing tkinter to force English UI
os.environ['LC_ALL'] = 'C'
os.environ['LANG'] = 'C'
os.environ['LC_MESSAGES'] = 'C'

import json
import shutil
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from os.path import join, basename, splitext, abspath


# -------------------------- Core Utility Functions --------------------------
def get_file_extension(file_path, normalize_tiff=True):
    ext = splitext(file_path)[1].lower()
    if normalize_tiff and ext == ".tiff":
        return ".tif"
    return ext


def check_image_format_consistency(src_dir):
    valid_exts = [".bmp", ".png", ".tif", ".tiff"]
    image_files = [f for f in os.listdir(src_dir) if get_file_extension(f) in valid_exts]
    if not image_files:
        return False, "No valid images found (support bmp/png/tif/tiff)"
    base_ext = get_file_extension(image_files[0])
    for f in image_files:
        if get_file_extension(f) != base_ext:
            return False, f"Mixed formats: {f} is {get_file_extension(f)}, base is {base_ext}"
    return True, base_ext


def read_image_with_chinese_path(img_path):
    try:
        img_path = abspath(img_path).replace("\\", "/")
        img_buffer = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        return img
    except Exception as e:
        raise ValueError(f"Read image failed: {e}")


def detect_image_channels(src_dir):
    valid_exts = [".bmp", ".png", ".tif", ".tiff"]
    image_files = [f for f in os.listdir(src_dir) if get_file_extension(f) in valid_exts]
    if not image_files:
        return 1
    img_path = join(src_dir, image_files[0])
    try:
        img = read_image_with_chinese_path(img_path)
        if len(img.shape) == 2:
            return 1
        elif len(img.shape) == 3:
            if img.shape[2] == 3:
                return 3
            elif img.shape[2] == 4:
                return 3
        return 1
    except Exception as e:
        raise ValueError(f"Detect channels failed: {e}")


def get_channel_desc(channel_count):
    return "RGB 3-Channel" if channel_count == 3 else "Single Channel"


def extract_all_label_classes(json_dir):
    all_classes = set()
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    if not json_files:
        raise ValueError("No labelme JSON files found!")
    for json_file in json_files:
        with open(join(json_dir, json_file), "r", encoding="utf-8") as f:
            data = json.load(f)
            for shape in data["shapes"]:
                class_name = shape["label"].strip()
                if class_name:
                    all_classes.add(class_name)
    sorted_classes = sorted(list(all_classes))
    label_mapping = {"background": 0}
    for idx, class_name in enumerate(sorted_classes):
        label_mapping[class_name] = idx + 1
    return label_mapping


# -------------------------- Core Functions --------------------------
def create_nnunet_directory_structure(base_path, task_id, task_name):
    task_dir = join(base_path, f"Dataset{task_id:03d}_{task_name}")
    for subdir in ["imagesTr", "labelsTr", "imagesTs"]:
        os.makedirs(join(task_dir, subdir), exist_ok=True)
    return task_dir


def split_and_rename_files(src_dir, tr_dir, ts_dir, split_ratio=0.0, suffix="_0000"):
    is_consistent, ext = check_image_format_consistency(src_dir)
    if not is_consistent:
        raise ValueError(ext)
    
    valid_exts = [".bmp", ".png", ".tif", ".tiff"]
    file_list = [f for f in os.listdir(src_dir) if get_file_extension(f) in valid_exts]
    total_count = len(file_list)
    tr_count = int(total_count * (1 - split_ratio))
    tr_files = file_list[:tr_count]
    ts_files = file_list[tr_count:]

    renamed_tr = []
    for idx, file_name in enumerate(tr_files):
        case_name = f"case{idx:04d}{suffix}{ext}"
        shutil.copy2(join(src_dir, file_name), join(tr_dir, case_name))
        renamed_tr.append((file_name, case_name))

    renamed_ts = []
    for idx, file_name in enumerate(ts_files):
        case_idx = tr_count + idx
        case_name = f"case{case_idx:04d}{suffix}{ext}"
        shutil.copy2(join(src_dir, file_name), join(ts_dir, case_name))
        renamed_ts.append((file_name, case_name))

    return renamed_tr, renamed_ts, ext


def generate_multi_class_mask_from_labelme_json(json_path, mask_save_path, label_mapping):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        mask = np.zeros((data["imageHeight"], data["imageWidth"]), dtype=np.uint8)
        for shape in data["shapes"]:
            class_value = label_mapping[shape["label"].strip()]
            points = np.array(shape["points"], dtype=np.int32)
            cv2.fillPoly(mask, [points], color=class_value)
        
        _, ext = splitext(mask_save_path)
        _, im_buf_arr = cv2.imencode(ext, mask)
        im_buf_arr.tofile(mask_save_path)
        return True
    except Exception as e:
        print(f"Generate mask failed: {e}")
        return False


def generate_nnunet_v2_dataset_json(task_dir, task_name, tr_cases, ts_cases, img_ext, label_mapping, channel_count):
    channel_names = {"0": "R", "1": "G", "2": "B"} if channel_count == 3 else {"0": "Image"}
    dataset_json = {
        "channel_names": channel_names,
        "labels": label_mapping,
        "numTraining": len(tr_cases),
        "file_ending": img_ext
    }
    if img_ext in [".tif", ".tiff"]:
        dataset_json["overwrite_image_reader_writer"] = "SimpleITKIO"
    
    json_path = join(task_dir, "dataset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4, ensure_ascii=False)
    return json_path


# -------------------------- GUI Class --------------------------
class nnUNetV2DatasetCreator(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Force English locale for Tcl
        try:
            self.tk.eval('::msgcat::mclocale en')
        except:
            pass
        
        self.title("nnUNet v2 Dataset Generator")
        self.geometry("800x600")
        self.resizable(False, False)
        
        # Variables
        self.raw_img_dir = tk.StringVar()
        self.json_label_dir = tk.StringVar()
        self.task_id = tk.StringVar(value="123")
        self.task_name = tk.StringVar(value="MyTask")
        self.nnUNet_raw_path = tk.StringVar()
        self.split_ratio = tk.StringVar(value="0.0")
        
        # Auto-load nnUNet_raw env var
        if os.environ.get('nnUNet_raw'):
            self.nnUNet_raw_path.set(os.environ.get('nnUNet_raw'))
        
        self._create_widgets()

    def _create_widgets(self):
        # Path Configuration
        frame_path = ttk.LabelFrame(self, text="Path Configuration")
        frame_path.pack(fill="x", padx=20, pady=10)

        ttk.Label(frame_path, text="Image Directory:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(frame_path, textvariable=self.raw_img_dir, width=60).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame_path, text="Browse", command=self._select_raw_img_dir).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(frame_path, text="JSON Label Dir:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(frame_path, textvariable=self.json_label_dir, width=60).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(frame_path, text="Browse", command=self._select_json_label_dir).grid(row=1, column=2, padx=5, pady=5)

        ttk.Label(frame_path, text="nnUNet_raw Path:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(frame_path, textvariable=self.nnUNet_raw_path, width=60).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(frame_path, text="Browse", command=self._select_nnunet_raw_dir).grid(row=2, column=2, padx=5, pady=5)

        # Task Configuration
        frame_task = ttk.LabelFrame(self, text="Task Configuration")
        frame_task.pack(fill="x", padx=20, pady=10)

        ttk.Label(frame_task, text="Task ID:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(frame_task, textvariable=self.task_id, width=15).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame_task, text="Task Name:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        ttk.Entry(frame_task, textvariable=self.task_name, width=15).grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(frame_task, text="Test Split:").grid(row=0, column=4, padx=5, pady=5, sticky="w")
        ttk.Entry(frame_task, textvariable=self.split_ratio, width=8).grid(row=0, column=5, padx=5, pady=5)
        ttk.Label(frame_task, text="(0=all train)").grid(row=0, column=6, padx=5, pady=5, sticky="w")

        # Log Area
        frame_log = ttk.LabelFrame(self, text="Processing Log")
        frame_log.pack(fill="both", expand=True, padx=20, pady=10)

        self.log_text = tk.Text(frame_log, height=15, width=90)
        self.log_text.pack(padx=5, pady=5)

        # Execute Button
        ttk.Button(self, text="Generate nnUNet v2 Dataset", command=self._run_dataset_creation).pack(pady=10)

    def _select_raw_img_dir(self):
        dir_path = filedialog.askdirectory(title="Select Image Directory")
        if dir_path:
            self.raw_img_dir.set(dir_path)
            try:
                channels = detect_image_channels(dir_path)
                self._log(f"[OK] Auto-detected channels: {channels} ({get_channel_desc(channels)})")
            except Exception as e:
                self._log(f"[WARN] Channel detection: {e}")

    def _select_json_label_dir(self):
        dir_path = filedialog.askdirectory(title="Select JSON Label Directory")
        if dir_path:
            self.json_label_dir.set(dir_path)
            try:
                label_mapping = extract_all_label_classes(dir_path)
                self._log(f"[OK] Found classes: {label_mapping}")
            except Exception as e:
                self._log(f"[WARN] Label scan: {e}")

    def _select_nnunet_raw_dir(self):
        dir_path = filedialog.askdirectory(title="Select nnUNet_raw Directory")
        if dir_path:
            self.nnUNet_raw_path.set(dir_path)

    def _log(self, msg):
        self.log_text.insert(tk.END, f"{msg}\n")
        self.log_text.see(tk.END)
        self.update()

    def _run_dataset_creation(self):
        if not self.raw_img_dir.get():
            messagebox.showerror("Error", "Please select image directory!")
            return
        if not self.json_label_dir.get():
            messagebox.showerror("Error", "Please select JSON label directory!")
            return
        if not self.nnUNet_raw_path.get():
            messagebox.showerror("Error", "Please select nnUNet_raw directory!")
            return
        if not self.task_id.get().isdigit():
            messagebox.showerror("Error", "Task ID must be a number!")
            return

        try:
            split_ratio = float(self.split_ratio.get())
            if not (0.0 <= split_ratio <= 1.0):
                raise ValueError("Split ratio must be 0-1")
        except ValueError:
            messagebox.showerror("Error", "Split ratio must be a number (e.g., 0.2)!")
            return

        try:
            task_id = int(self.task_id.get())
            task_name = self.task_name.get().strip()
            raw_img_dir = self.raw_img_dir.get()
            json_label_dir = self.json_label_dir.get()
            nnunet_raw = self.nnUNet_raw_path.get()

            self._log("=" * 60)
            self._log(f"Starting dataset generation...")
            self._log(f"Task ID: {task_id} | Name: {task_name} | Test split: {split_ratio}")

            is_consistent, img_ext = check_image_format_consistency(raw_img_dir)
            if not is_consistent:
                raise ValueError(img_ext)
            self._log(f"[OK] Image format: {img_ext}")

            channel_count = detect_image_channels(raw_img_dir)
            self._log(f"[OK] Channels: {channel_count} ({get_channel_desc(channel_count)})")

            label_mapping = extract_all_label_classes(json_label_dir)
            self._log(f"[OK] Label mapping: {label_mapping}")

            self._log("[...] Creating directory structure...")
            task_dir = create_nnunet_directory_structure(nnunet_raw, task_id, task_name)
            self._log(f"[OK] Created: {task_dir}")

            self._log("[...] Splitting and renaming images...")
            images_tr_dir = join(task_dir, "imagesTr")
            images_ts_dir = join(task_dir, "imagesTs")
            renamed_tr, renamed_ts, img_ext = split_and_rename_files(
                raw_img_dir, images_tr_dir, images_ts_dir, split_ratio=split_ratio
            )
            self._log(f"[OK] Images: {len(renamed_tr)} train, {len(renamed_ts)} test")

            self._log("[...] Generating masks...")
            labels_tr_dir = join(task_dir, "labelsTr")
            json_files = [f for f in os.listdir(json_label_dir) if f.endswith(".json")]
            json_tr_files = json_files[:len(renamed_tr)]
            success_count = 0

            for idx, json_file in enumerate(json_tr_files):
                mask_name = f"case{idx:04d}{img_ext}"
                json_path = join(json_label_dir, json_file)
                mask_path = join(labels_tr_dir, mask_name)
                if generate_multi_class_mask_from_labelme_json(json_path, mask_path, label_mapping):
                    success_count += 1
                    self._log(f"[OK] Mask {idx+1}/{len(json_tr_files)}: {json_file}")
                else:
                    self._log(f"[FAIL] Mask: {json_file}")

            self._log(f"[OK] Masks: {success_count}/{len(json_tr_files)} generated")

            self._log("[...] Generating dataset.json...")
            tr_cases = [case[1] for case in renamed_tr]
            ts_cases = [case[1] for case in renamed_ts]
            dataset_json_path = generate_nnunet_v2_dataset_json(
                task_dir, task_name, tr_cases, ts_cases, img_ext,
                label_mapping=label_mapping, channel_count=channel_count
            )
            self._log(f"[OK] Saved: {dataset_json_path}")

            self._log("=" * 60)
            self._log("SUCCESS! Dataset ready for nnUNet v2 training.")
            self._log(f"Path: {task_dir}")
            self._log(f"Stats: {len(renamed_tr)} train + {len(renamed_ts)} test | {len(label_mapping)-1} classes")
            messagebox.showinfo("Success", "Dataset generated successfully!\nCheck the log for details.")

        except Exception as e:
            self._log(f"[ERROR] {str(e)}")
            messagebox.showerror("Error", f"Failed: {str(e)}")


if __name__ == "__main__":
    try:
        import cv2
    except ImportError:
        print("Please install: pip install opencv-python numpy")
        exit(1)

    app = nnUNetV2DatasetCreator()
    app.mainloop()