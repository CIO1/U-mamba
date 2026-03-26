import os
import json
os.environ['LC_ALL'] = 'C'
os.environ['LANG'] = 'C'
os.environ['LC_MESSAGES'] = 'C'
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import Toplevel, Label, Button
from os.path import join, basename, splitext, abspath, isfile, isdir
from pathlib import Path
from typing import Tuple, Union
import time

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn

# UMamba specific imports
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.network_initialization import InitWeights_He
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
from nnunetv2.nets.UMambaBot_2d import UMambaBot
import warnings
warnings.filterwarnings('ignore')

# ===================== Device Configuration =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    raise RuntimeError("UMamba operators require CUDA. Please run in GPU environment.")

# ===================== Custom Dialog to Avoid Encoding Issues =====================
class CustomDialog(Toplevel):
    """Custom dialog to replace messagebox and avoid Chinese button encoding issues in WSL"""
    def __init__(self, parent, title, message, dialog_type="info"):
        super().__init__(parent)
        self.title(title)
        self.geometry("400x150")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        # Center the dialog
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (200)
        y = (self.winfo_screenheight() // 2) - (75)
        self.geometry(f"+{x}+{y}")
        
        # Icon based on type
        icon_frame = ttk.Frame(self)
        icon_frame.pack(pady=10)
        
        if dialog_type == "error":
            icon_label = Label(icon_frame, text="X", fg="red", font=("Arial", 24, "bold"))
        elif dialog_type == "warning":
            icon_label = Label(icon_frame, text="!", fg="orange", font=("Arial", 24, "bold"))
        else:
            icon_label = Label(icon_frame, text="i", fg="blue", font=("Arial", 24, "bold"))
        icon_label.pack()
        
        # Message
        msg_label = Label(self, text=message, wraplength=350, justify="center")
        msg_label.pack(pady=10, padx=20)
        
        # OK Button (English text to avoid encoding issues)
        ok_btn = Button(self, text="OK", width=10, command=self.destroy)
        ok_btn.pack(pady=10)
        
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.wait_window(self)

def show_info(parent, title, message):
    CustomDialog(parent, title, message, "info")

def show_warning(parent, title, message):
    CustomDialog(parent, title, message, "warning")

def show_error(parent, title, message):
    CustomDialog(parent, title, message, "error")

# ===================== Utility Functions =====================
def get_umamba_bot_2d_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True
):
    """Build UMambaBot network architecture"""
    num_stages = len(configuration_manager.conv_kernel_sizes)
    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)
    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = 'UMambaBot'
    network_class = UMambaBot
    kwargs = {
        'UMambaBot': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }

    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))
    return model


def round_to_32_multiple(size: int) -> int:
    """Round up to nearest multiple of 32"""
    if size % 32 == 0:
        return size
    return ((size // 32) + 1) * 32


def generate_input_shape(channel_num: int, raw_h: int, raw_w: int, batch_size: int = 1) -> tuple:
    """Generate compliant [b,c,h,w] shape"""
    h_processed = round_to_32_multiple(raw_h)
    w_processed = round_to_32_multiple(raw_w)
    return (batch_size, channel_num, h_processed, w_processed)


def get_timestamp() -> str:
    """Generate timestamp string (format: YYYYMMDD_HHMMSS)"""
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def get_dataset_and_plans_info(dataset_path: str, config: str = "2d") -> tuple:
    """
    Read plans.json and dataset.json from the given path
    :return: (channel_num, class_num, total_class_num, raw_h, raw_w, model_dir)
    """
    model_dir = Path(dataset_path)

    if not model_dir.is_dir():
        raise RuntimeError(f"Model directory not found: {model_dir}")

    # Load dataset.json
    dataset_json_path = join(model_dir, "dataset.json")
    if not isfile(dataset_json_path):
        raise FileNotFoundError(f"dataset.json not found: {dataset_json_path}")
    dataset_json = load_json(dataset_json_path)
    channel_num = len(dataset_json["channel_names"].keys())
    total_class_num = len(dataset_json["labels"].keys())
    class_num = total_class_num - 1

    # Load plans.json
    plans_path = join(model_dir, "plans.json")
    if not isfile(plans_path):
        raise FileNotFoundError(f"plans.json not found: {plans_path}")
    plans = load_json(plans_path)
    patch_size = plans['configurations'][config]['patch_size']

    # Extract h/w (compatible with 2D/3D)
    if len(patch_size) == 3:  # 3D (d, h, w)
        raw_h, raw_w = patch_size[1], patch_size[2]
    else:  # 2D (h, w)
        raw_h, raw_w = patch_size[0], patch_size[1]

    return channel_num, class_num, total_class_num, raw_h, raw_w, str(model_dir)


# ===================== Core Export Logic =====================
def export_umamba_model(
        model_dir: str,
        output_dir: Path,
        config_name: str = "2d",
        folds: Tuple[Union[int, str], ...] = (0,),
        checkpoint_name: str = "checkpoint_final.pth",
        output_name: str = None,
        verbose: bool = False,
        log_callback=None,
        custom_input_shape: tuple = None,
        enable_dynamic_dim: bool = True,
        deep_supervision: bool = False
) -> None:
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    # Generate timestamp to avoid file overwriting
    timestamp = get_timestamp()
    if not output_name:
        output_name = f"{checkpoint_name[:-4]}_{timestamp}.onnx"

    log(f"Starting UMamba ONNX export...")
    log(f"Input shape: {custom_input_shape} | Dynamic dims: {enable_dynamic_dim} | Timestamp: {timestamp}")

    # Load configuration files
    plans_path = join(model_dir, "plans.json")
    dataset_json_path = join(model_dir, "dataset.json")
    
    if not isfile(plans_path) or not isfile(dataset_json_path):
        raise FileNotFoundError("plans.json or dataset.json not found in model directory")
    
    plans_dict = load_json(plans_path)
    dataset_json = load_json(dataset_json_path)
    config = plans_dict["configurations"][config_name]
    
    plans_manager = PlansManager(plans_dict)
    configuration_manager = ConfigurationManager(config)

    log(f"\n===== Processing config: {config_name} =====")

    for fold in folds:
        log(f"\nProcessing Fold {fold}...")
        
        # Load model weights
        checkpoint_path = join(model_dir, f"fold_{fold}", checkpoint_name)
        if not isfile(checkpoint_path):
            log(f"Warning: Checkpoint not found at {checkpoint_path}, skipping...")
            continue
            
        # Create model
        input_channels = len(dataset_json["channel_names"].keys())
        model = get_umamba_bot_2d_from_plans(
            plans_manager=plans_manager,
            dataset_json=dataset_json,
            configuration_manager=configuration_manager,
            num_input_channels=input_channels,
            deep_supervision=deep_supervision
        )
        
        # Load weights to CUDA
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "network_weights" in checkpoint:
            model.load_state_dict(checkpoint["network_weights"])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()

        # Prepare output directory
        curr_output_dir = Path(output_dir) / f"outputmodel_{config_name}" / f"fold_{fold}"
        curr_output_dir.mkdir(parents=True, exist_ok=True)
        log(f"Output directory: {curr_output_dir}")

        # Prepare input tensor on CUDA
        b, ch, h, w = custom_input_shape
        rand_input = torch.rand((b, ch, h, w), device=device)
        log(f"Using input shape: (b={b}, c={ch}, h={h}, w={w})")

        # Dynamic axes configuration
        dynamic_axes = None
        if enable_dynamic_dim:
            dynamic_axes = {
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size", 2: "height", 3: "width"}
            }
            log("Dynamic dimensions enabled: batch_size/height/width are variable")

        # Forward pass
        with torch.no_grad():
            torch_output = model(rand_input)
            # Handle deep supervision output (list) or single output
            if isinstance(torch_output, list):
                torch_output_single = torch_output[-1]
            else:
                torch_output_single = torch_output

        # Export ONNX (with timestamp)
        onnx_path = curr_output_dir / output_name
        log(f"Exporting ONNX: {onnx_path}")
        
        torch.onnx.export(
            model,
            rand_input,
            onnx_path,
            export_params=True,
            verbose=verbose,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=False,
            opset_version=12,
            keep_initializers_as_inputs=True,
        )

        # Validate ONNX model
        log("Validating ONNX model...")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        # ONNX Runtime validation (CPU-based)
        log("Testing with ONNX Runtime...")
        ort_session = onnxruntime.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"]
        )
        ort_inputs = {ort_session.get_inputs()[0].name: rand_input.cpu().numpy()}
        ort_outs = ort_session.run(None, ort_inputs)

        # Precision check
        try:
            np.testing.assert_allclose(
                torch_output_single.detach().cpu().numpy(),
                ort_outs[0],
                rtol=1e-03,
                atol=1e-05,
                verbose=True,
            )
            log("Torch and ONNX outputs match within tolerance")
        except AssertionError as e:
            log(f"Warning: Torch vs ONNX output difference detected:\n{e}")
            log("Export completed, but please verify pipeline consistency!")

        # Save configuration (with timestamp)
        config_filename = f"config_{timestamp}.json"
        config_dict = {
            "configuration": config_name,
            "fold": fold,
            "export_timestamp": timestamp,
            "model_parameters": {
                "input_shape": custom_input_shape,
                "enable_dynamic_dim": enable_dynamic_dim,
                "patch_size": configuration_manager.patch_size,
                "spacing": configuration_manager.spacing,
                "normalization_schemes": configuration_manager.normalization_schemes,
            },
            "dataset_parameters": {
                "dataset_path": str(model_dir),
                "num_channels": input_channels,
                "num_classes": len(dataset_json["labels"].keys()) - 1,
                "total_classes_with_background": len(dataset_json["labels"].keys()),
                "class_names": {v: k for k, v in dataset_json["labels"].items()},
            },
        }
        config_path = curr_output_dir / config_filename
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)
        log(f"Configuration saved: {config_path}")
        log(f"Successfully exported: {onnx_path}")

    log("\n===== Export process completed =====")


# ===================== UI Class =====================
class UMambaONNXExporterUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("UMamba ONNX Exporter")
        self.geometry("900x800")
        self.resizable(False, False)

        # Initialize variables
        self.model_dir_var = tk.StringVar(value=os.getcwd())
        self.output_dir_var = tk.StringVar(value=os.getcwd())
        self.folds_var = tk.StringVar(value="0")
        self.config_var = tk.StringVar(value="2d")
        self.checkpoint_var = tk.StringVar(value="checkpoint_final.pth")
        self.custom_shape_var = tk.StringVar(value="1,1,512,512")
        self.enable_dynamic_dim_var = tk.BooleanVar(value=True)
        self.deep_supervision_var = tk.BooleanVar(value=False)
        self.channel_num_var = tk.StringVar(value="")
        self.class_num_var = tk.StringVar(value="")
        self.total_class_num_var = tk.StringVar(value="")

        self._create_widgets()
        self._auto_load_default_info()

    def _create_widgets(self):
        """Create UI layout"""
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. Model Directory Input
        ttk.Label(main_frame, text="1. Model Directory (containing plans.json):").grid(
            row=0, column=0, sticky=tk.W, pady=(0, 10))
        ttk.Entry(main_frame, textvariable=self.model_dir_var, width=50).grid(
            row=0, column=1, sticky=tk.W, pady=(0, 10))
        ttk.Button(main_frame, text="Browse", command=self._select_model_dir).grid(
            row=0, column=2, sticky=tk.W, padx=(10, 0))

        # 2. Refresh Dataset Info
        ttk.Button(
            main_frame, text="Load Dataset Info",
            command=self._auto_load_default_info
        ).grid(row=1, column=1, sticky=tk.W, pady=(0, 10))

        # 3. Dataset Info Display
        ttk.Label(main_frame, text="2. Dataset Information:").grid(row=2, column=0, sticky=tk.W, pady=(0, 10))
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=2, column=1, sticky=tk.W, pady=(0, 10))
        
        ttk.Label(info_frame, text="Input Channels: ").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(info_frame, textvariable=self.channel_num_var, foreground="blue").grid(
            row=0, column=1, sticky=tk.W, padx=(5, 20))
        ttk.Label(info_frame, text="Foreground Classes: ").grid(row=0, column=2, sticky=tk.W)
        ttk.Label(info_frame, textvariable=self.class_num_var, foreground="blue").grid(
            row=0, column=3, sticky=tk.W, padx=(5, 20))
        ttk.Label(info_frame, text="Total Classes (with bg): ").grid(row=0, column=4, sticky=tk.W)
        ttk.Label(info_frame, textvariable=self.total_class_num_var, foreground="red").grid(
            row=0, column=5, sticky=tk.W, padx=(5, 0))

        # 4. Output Directory
        ttk.Label(main_frame, text="3. Output Directory:").grid(row=3, column=0, sticky=tk.W, pady=(0, 10))
        ttk.Entry(main_frame, textvariable=self.output_dir_var, width=50).grid(row=3, column=1, sticky=tk.W, pady=(0, 10))
        ttk.Button(main_frame, text="Browse", command=self._select_output_dir).grid(
            row=3, column=2, sticky=tk.W, padx=(10, 0))

        # 5. Model Configuration
        ttk.Label(main_frame, text="4. Model Configuration:").grid(row=4, column=0, sticky=tk.W, pady=(0, 10))
        config_combobox = ttk.Combobox(
            main_frame, textvariable=self.config_var,
            values=["2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"],
            width=27
        )
        config_combobox.grid(row=4, column=1, sticky=tk.W, pady=(0, 10))
        ttk.Label(main_frame, text="(Select configuration to export)").grid(row=4, column=2, sticky=tk.W, padx=(10, 0))

        # 6. Fold Selection
        ttk.Label(main_frame, text="5. Folds to Export:").grid(row=5, column=0, sticky=tk.W, pady=(0, 10))
        ttk.Entry(main_frame, textvariable=self.folds_var, width=30).grid(row=5, column=1, sticky=tk.W, pady=(0, 10))
        ttk.Label(main_frame, text="(Comma separated, e.g., 0 or 0,1,2)").grid(row=5, column=2, sticky=tk.W, padx=(10, 0))

        # 7. Checkpoint Name
        ttk.Label(main_frame, text="6. Checkpoint Name:").grid(row=6, column=0, sticky=tk.W, pady=(0, 10))
        ttk.Entry(main_frame, textvariable=self.checkpoint_var, width=30).grid(row=6, column=1, sticky=tk.W, pady=(0, 10))
        ttk.Label(main_frame, text="(Default: checkpoint_final.pth)").grid(row=6, column=2, sticky=tk.W, padx=(10, 0))

        # 8. Input Shape
        ttk.Label(main_frame, text="7. Input Shape [b,c,h,w]:").grid(row=7, column=0, sticky=tk.W, pady=(0, 10))
        ttk.Entry(main_frame, textvariable=self.custom_shape_var, width=30, foreground="green").grid(
            row=7, column=1, sticky=tk.W, pady=(0, 10))
        ttk.Label(main_frame, text="(h/w auto-rounded to 32 multiples)").grid(row=7, column=2, sticky=tk.W, padx=(10, 0))

        # 9. Dynamic Dimension
        ttk.Label(main_frame, text="8. Dynamic Dimensions:").grid(row=8, column=0, sticky=tk.W, pady=(0, 10))
        ttk.Checkbutton(
            main_frame, text="Enable dynamic dimensions (batch/height/width variable) [Default: ON]",
            variable=self.enable_dynamic_dim_var
        ).grid(row=8, column=1, sticky=tk.W, pady=(0, 10))

        # 10. Deep Supervision
        ttk.Label(main_frame, text="9. Deep Supervision:").grid(row=9, column=0, sticky=tk.W, pady=(0, 10))
        ttk.Checkbutton(
            main_frame, text="Model uses deep supervision (export only final output)",
            variable=self.deep_supervision_var
        ).grid(row=9, column=1, sticky=tk.W, pady=(0, 10))

        # 11. Export Button
        export_btn = ttk.Button(main_frame, text="Start Export", command=self._start_export, style="Accent.TButton")
        export_btn.grid(row=10, column=0, columnspan=3, pady=(20, 20))

        # 12. Log Output Area
        ttk.Label(main_frame, text="Export Log:").grid(row=11, column=0, sticky=tk.W)
        self.log_text = tk.Text(main_frame, width=100, height=20, wrap=tk.WORD, font=("Consolas", 9))
        self.log_text.grid(row=12, column=0, columnspan=3)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=12, column=3, sticky=tk.NS)
        self.log_text.config(yscrollcommand=scrollbar.set)

        # Style configuration
        self.style = ttk.Style()
        self.style.configure("Accent.TButton", font=("Arial", 10, "bold"))

    def _select_model_dir(self):
        """Select model directory containing plans.json"""
        dir_path = filedialog.askdirectory(title="Select Model Directory (containing plans.json)")
        if dir_path:
            self.model_dir_var.set(dir_path)
            self._auto_load_default_info()

    def _select_output_dir(self):
        """Select output directory"""
        dir_path = filedialog.askdirectory(title="Select Output Directory")
        if dir_path:
            self.output_dir_var.set(dir_path)

    def _log(self, msg):
        """Log output to text widget"""
        self.log_text.insert(tk.END, f"{msg}\n")
        self.log_text.see(tk.END)
        self.update()

    def _auto_load_default_info(self):
        """Load dataset info and auto-fill input shape"""
        try:
            if self.log_text.get(1.0, tk.END).strip() == "":
                self._log("===== Auto-loading dataset configuration =====")

            model_dir = self.model_dir_var.get().strip()
            config = self.config_var.get().strip()

            if not model_dir:
                raise ValueError("Please select model directory first")

            if not isdir(model_dir):
                raise ValueError(f"Directory not found: {model_dir}")

            self._log(f"Loading dataset info from {model_dir} (config: {config})...")
            channel_num, class_num, total_class_num, raw_h, raw_w, model_dir_path = get_dataset_and_plans_info(
                model_dir, config)

            self.channel_num_var.set(str(channel_num))
            self.class_num_var.set(str(class_num))
            self.total_class_num_var.set(str(total_class_num))
            self._log(f"Dataset info - Channels: {channel_num} | Classes: {class_num} | Total: {total_class_num}")
            self._log(f"Raw inference size - h={raw_h}, w={raw_w}")

            input_shape = generate_input_shape(channel_num, raw_h, raw_w, batch_size=1)
            input_shape_str = ",".join(map(str, input_shape))
            self.custom_shape_var.set(input_shape_str)
            self._log(f"Generated compliant input shape: {input_shape_str} (b,c,h,w)")

            # Auto-set output directory to model directory if not set
            current_output = self.output_dir_var.get()
            if current_output == os.getcwd():
                self.output_dir_var.set(model_dir_path)
                self._log(f"Auto-set output directory to: {model_dir_path}")

            show_info(self, "Success", 
                f"Dataset loaded successfully!\n"
                f"Input Channels: {channel_num}\n"
                f"Classes: {class_num}\n"
                f"Total Classes: {total_class_num}\n"
                f"Raw Size: {raw_h}x{raw_w}\n"
                f"Compliant Shape: {input_shape_str}")
        except Exception as e:
            err_msg = f"Failed to load dataset info: {str(e)}\nPlease verify the model directory contains plans.json and dataset.json"
            self._log(f"Error: {err_msg}")
            show_warning(self, "Loading Error", err_msg)

    def _parse_custom_shape(self, shape_str):
        """Parse and validate custom shape (h/w must be 32 multiples)"""
        try:
            shape = tuple(map(int, shape_str.strip().split(",")))
            if len(shape) != 4:
                raise ValueError("Shape must have 4 dimensions (b,c,h,w)")

            b, c, h, w = shape
            if h % 32 != 0 or w % 32 != 0:
                raise ValueError(f"h={h} or w={w} not multiple of 32 (nnUNet requirement)")
            if b <= 0 or c <= 0 or h <= 0 or w <= 0:
                raise ValueError("All dimensions must be positive")

            return shape
        except Exception as e:
            raise ValueError(f"Shape parsing/validation failed: {str(e)}")

    def _start_export(self):
        """Start export process"""
        try:
            self.log_text.delete(1.0, tk.END)
            self._log("===== Starting UMamba ONNX Export =====")

            model_dir = self.model_dir_var.get().strip()
            output_dir = Path(self.output_dir_var.get().strip())
            folds = tuple(map(int, self.folds_var.get().strip().split(",")))
            config = self.config_var.get().strip()
            checkpoint = self.checkpoint_var.get().strip()

            custom_shape_str = self.custom_shape_var.get().strip()
            custom_shape = self._parse_custom_shape(custom_shape_str)
            self._log(f"Input shape validated: {custom_shape} (b,c,h,w)")

            enable_dynamic_dim = self.enable_dynamic_dim_var.get()
            deep_supervision = self.deep_supervision_var.get()

            if not model_dir:
                raise ValueError("Model directory cannot be empty")
            if not output_dir:
                raise ValueError("Output directory cannot be empty")

            export_umamba_model(
                model_dir=model_dir,
                output_dir=output_dir,
                config_name=config,
                folds=folds,
                checkpoint_name=checkpoint,
                log_callback=self._log,
                custom_input_shape=custom_shape,
                enable_dynamic_dim=enable_dynamic_dim,
                deep_supervision=deep_supervision
            )

            show_info(self, "Success", "ONNX model export completed! Files have timestamps to prevent overwriting.")

        except Exception as e:
            err_msg = f"Export failed: {str(e)}"
            self._log(f"Error: {err_msg}")
            show_error(self, "Error", err_msg)


# ===================== Main Entry =====================
if __name__ == '__main__':
    # Check dependencies
    try:
        import nnunetv2
        import batchgenerators
        import dynamic_network_architectures
    except ImportError as e:
        # Use standard messagebox here since main window doesn't exist yet
        root = tk.Tk()
        root.withdraw()
        show_error(root, "Missing Dependencies", f"Please install required packages:\n{e}")
        root.destroy()
    else:
        app = UMambaONNXExporterUI()
        app.mainloop()