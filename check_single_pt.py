#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Testing Script - Test performance and detection results of a single PyTorch model
Usage:
    python check_single_pt.py model_path [image_path]
"""

import os
import cv2
import sys
import time
import subprocess
import numpy as np
from pathlib import Path

# Default test image
DEFAULT_TEST_IMAGE = "test_image/1.jpg"
# Default results save path
DEFAULT_SAVE_PATH = "results/"

# Import ultralytics library
try:
    from ultralytics import YOLO
    print("Successfully imported ultralytics library")
except ImportError:
    print("Error: Unable to import ultralytics library")
    print("Please install: pip install ultralytics")
    sys.exit(1)

def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def open_image(image_path):
    """Open image with default system program"""
    try:
        if sys.platform == "darwin":  # macOS
            subprocess.call(["open", image_path])
        elif sys.platform == "win32":  # Windows
            os.startfile(image_path)
        else:  # Linux or others
            subprocess.call(["xdg-open", image_path])
        print(f"Opened image: {image_path}")
    except Exception as e:
        print(f"Unable to open image: {e}")

def test_model(model_path, image_path, save_path=None, model_name=None, auto_open=True):
    """Test YOLO model"""
    try:
        # If model name not provided, extract from path
        if model_name is None:
            model_name = os.path.basename(model_path)
            
        # Check if files exist
        if not os.path.exists(model_path):
            print(f"Error: Model file doesn't exist - {model_path}")
            return
            
        if not os.path.exists(image_path):
            print(f"Error: Image file doesn't exist - {image_path}")
            return
        
        # If save path not provided, create default path
        if save_path is None:
            # Create results directory
            ensure_dir(DEFAULT_SAVE_PATH)
            # Extract image file name (without extension)
            image_basename = os.path.splitext(os.path.basename(image_path))[0]
            # Extract model file name (without extension)
            model_basename = os.path.splitext(os.path.basename(model_path))[0]
            # Compose save path
            save_path = os.path.join(DEFAULT_SAVE_PATH, f"{image_basename}_{model_basename}_result.jpg")
            
        # Load model
        print(f"Loading model: {model_name} ({model_path})")
        load_start_time = time.time()
        model = YOLO(model_path)
        load_time = time.time() - load_start_time
        print(f"Model loading time: {load_time:.2f} seconds")
        
        # Read image
        print(f"Reading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to read image {image_path}")
            return
            
        # Run detection - Only measure the model inference time
        print("Starting detection...")
        detect_start_time = time.time()
        results = model(image)
        detect_time = time.time() - detect_start_time
        print(f"Detection time: {detect_time:.2f} seconds")
        
        # Process detection results
        has_violence = False
        violence_count = 0
        nonviolence_count = 0
        
        # Process results
        print("\n===== Detection Results =====")
        for r in results:
            boxes = r.boxes
            for i, box in enumerate(boxes):
                # Get class ID and confidence
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get class name (if available)
                if hasattr(r, 'names') and cls_id in r.names:
                    class_name = r.names[cls_id]
                else:
                    class_name = f"class_{cls_id}"
                
                # Print detection results
                print(f"Object {i+1}: {class_name} (confidence: {conf:.2f}), position: [{x1}, {y1}, {x2}, {y2}]")
                
                # Violence detection logic
                if cls_id == 1 or class_name.lower() in ['violence', 'fight', '暴力']:
                    has_violence = True
                    violence_count += 1
                else:
                    nonviolence_count += 1
                    
        # Print detection summary
        print("\n===== Detection Summary =====")
        if violence_count == 0 and nonviolence_count == 0:
            print("No objects detected")
        elif has_violence:
            print(f"⚠️  WARNING: Violence content detected in image!")
            print(f"Detected {violence_count} violent objects, {nonviolence_count} non-violent objects")
        else:
            print("✓ Image is safe: No violence detected")
            print(f"Detected {nonviolence_count} non-violent objects")
        
        # Print performance information
        print(f"\n===== Performance Information =====")
        print(f"Model loading time: {load_time:.2f} seconds")
        print(f"Detection time: {detect_time:.2f} seconds")
        print(f"Total time: {load_time + detect_time:.2f} seconds")
        
        # Measure post-processing time separately
        postprocess_start_time = time.time()
        
        # Generate custom annotated image and save (only showing violence labels)
        # First, make a copy of the original image for drawing
        custom_result = image.copy()
        
        for r in results:
            boxes = r.boxes
            for i, box in enumerate(boxes):
                # Get class ID and confidence
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Get class name (if available)
                if hasattr(r, 'names') and cls_id in r.names:
                    class_name = r.names[cls_id]
                else:
                    class_name = f"class_{cls_id}"
                
                # Only draw violence boxes
                if cls_id == 1 or class_name.lower() in ['violence', 'fight', '暴力']:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw bounding box (red for violence)
                    cv2.rectangle(custom_result, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Create label text
                    label = f"{class_name} {conf:.2f}"
                    
                    # Draw label background and text
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(custom_result, (x1, y1-text_size[1]-5), (x1+text_size[0], y1), (0, 0, 255), -1)
                    cv2.putText(custom_result, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add performance information to the top of the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (255, 255, 255)  # White
            thickness = 2
            
            # Define text to add
            text_lines = [
                f"Model: {model_name}",
                f"Load time: {load_time:.2f}s", 
                f"Detection time: {detect_time:.2f}s",
                f"Total time: {load_time + detect_time:.2f}s"
            ]
            
            # Calculate text box position
            padding = 10
            text_height = 30
            y_pos = padding
            
            # Add semi-transparent black background
            overlay = custom_result.copy()
            bg_height = (len(text_lines) + 1) * text_height
            cv2.rectangle(overlay, (0, 0), (custom_result.shape[1], bg_height), (0, 0, 0), -1)
            alpha = 0.6  # Transparency
            cv2.addWeighted(overlay, alpha, custom_result, 1 - alpha, 0, custom_result)
            
            # Add text
            for line in text_lines:
                y_pos += text_height
                cv2.putText(custom_result, line, (padding, y_pos), font, font_scale, color, thickness)
            
            # Save custom annotated image
            cv2.imwrite(save_path, custom_result)
            
            # Calculate post-processing time
            postprocess_time = time.time() - postprocess_start_time
            print(f"Post-processing time: {postprocess_time:.2f} seconds")
            print(f"Total processing time: {load_time + detect_time + postprocess_time:.2f} seconds")
            print(f"\nResult image saved to: {save_path}")
            
            # Auto open image
            if auto_open:
                open_image(save_path)
                
        return has_violence, results, load_time, detect_time, save_path
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, 0, 0, None

def export_to_onnx(model_path, output_path=None):
    """Convert PT model to ONNX format"""
    try:
        if not os.path.exists(model_path):
            print(f"Error: Model file doesn't exist - {model_path}")
            return False
            
        # If output path not specified, use original filename + .onnx
        if output_path is None:
            output_path = os.path.splitext(model_path)[0] + ".onnx"
            
        print(f"Loading model: {model_path}")
        model = YOLO(model_path)
        
        print(f"Exporting model to ONNX format: {output_path}")
        model.export(format="onnx", opset=12)
        
        # Check if export was successful
        expected_path = os.path.splitext(model_path)[0] + ".onnx"
        if os.path.exists(expected_path):
            print(f"ONNX model export successful: {expected_path}")
            # If user specified a different output path, rename the file
            if expected_path != output_path:
                os.rename(expected_path, output_path)
                print(f"Model renamed to: {output_path}")
            return True
        else:
            print(f"Error: ONNX model export failed")
            return False
            
    except Exception as e:
        print(f"Error during ONNX export: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_single_pt.py model_path [image_path] [--export-onnx] [--no-open]")
        print("Example: python check_single_pt.py V02_Musa_Violence/Yolo_nano_weights.pt test_image/1.jpg")
        print("Example: python check_single_pt.py V02_Musa_Violence/Yolo_nano_weights.pt --export-onnx")
        return
    
    # Parse command line arguments    
    model_path = sys.argv[1]
    
    # Check if ONNX export is needed
    if "--export-onnx" in sys.argv:
        export_to_onnx(model_path)
        return
        
    # Determine image path
    image_path = DEFAULT_TEST_IMAGE
    for arg in sys.argv[2:]:
        if not arg.startswith("--") and os.path.exists(arg):
            image_path = arg
            break
    
    # Determine save path
    save_path = None
    save_idx = sys.argv.index("--save") if "--save" in sys.argv else -1
    if save_idx > 0 and save_idx + 1 < len(sys.argv):
        save_path = sys.argv[save_idx + 1]
    
    # Whether to auto open image
    auto_open = "--no-open" not in sys.argv
    
    # Execute test
    test_model(model_path, image_path, save_path, auto_open=auto_open)

if __name__ == "__main__":
    main() 