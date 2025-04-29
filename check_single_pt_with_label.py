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
    print("成功导入ultralytics库")
except ImportError:
    print("错误：无法导入ultralytics库")
    print("请安装: pip install ultralytics")
    sys.exit(1)

def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

def open_image(image_path):
    """Open image with default system program"""
    try:
        if sys.platform == "darwin":  # macOS
            subprocess.call(["open", image_path])
        elif sys.platform == "win32":  # Windows
            os.startfile(image_path)
        else:  # Linux or others
            subprocess.call(["xdg-open", image_path])
        print(f"已打开图像: {image_path}")
    except Exception as e:
        print(f"无法打开图像: {e}")

def test_model(model_path, image_path, save_path=None, model_name=None, auto_open=True, warm_up=True, warm_up_rounds=2):
    """Test YOLO model"""
    try:
        # If model name not provided, extract from path
        if model_name is None:
            model_name = os.path.basename(model_path)
            
        # Check if files exist
        if not os.path.exists(model_path):
            print(f"错误：模型文件不存在 - {model_path}")
            return
            
        if not os.path.exists(image_path):
            print(f"错误：图像文件不存在 - {image_path}")
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
        print(f"加载模型: {model_name} ({model_path})")
        load_start_time = time.time()
        model = YOLO(model_path)
        load_time = time.time() - load_start_time
        print(f"模型加载时间: {load_time:.2f} 秒")
        
        # Read image
        print(f"读取图像: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误：无法读取图像 {image_path}")
            return
        
        # 预热模型
        if warm_up:
            print("\n===== 预热模型 =====")
            warm_up_times = []
            
            for i in range(warm_up_rounds):
                print(f"预热轮次 {i+1}/{warm_up_rounds}...")
                warm_up_start = time.time()
                model(image)
                warm_up_time = time.time() - warm_up_start
                warm_up_times.append(warm_up_time)
                print(f"预热轮次 {i+1} 耗时: {warm_up_time:.2f} 秒")
            
            avg_warm_up_time = sum(warm_up_times) / len(warm_up_times)
            print(f"平均预热时间: {avg_warm_up_time:.2f} 秒")
            
        # Run detection - Only measure the model inference time
        print("\n开始检测...")
        detect_start_time = time.time()
        results = model(image)
        detect_time = time.time() - detect_start_time
        print(f"检测时间: {detect_time:.2f} 秒")
        
        # Process detection results
        has_violence = False
        violence_count = 0
        nonviolence_count = 0
        
        # Process results
        print("\n===== 检测结果 =====")
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
                print(f"对象 {i+1}: {class_name} (置信度: {conf:.2f}), 位置: [{x1}, {y1}, {x2}, {y2}]")
                
                # Violence detection logic
                if cls_id == 1 or class_name.lower() in ['violence', 'fight', '暴力']:
                    has_violence = True
                    violence_count += 1
                else:
                    nonviolence_count += 1
                    
        # Print detection summary
        print("\n===== 检测摘要 =====")
        if violence_count == 0 and nonviolence_count == 0:
            print("未检测到任何对象")
        elif has_violence:
            print(f"⚠️  警告：图像中检测到暴力内容！")
            print(f"检测到 {violence_count} 个暴力对象，{nonviolence_count} 个非暴力对象")
        else:
            print("✓ 图像安全：未检测到暴力内容")
            print(f"检测到 {nonviolence_count} 个非暴力对象")
        
        # Print performance information
        print(f"\n===== 性能信息 =====")
        print(f"模型加载时间: {load_time:.2f} 秒")
        if warm_up:
            print(f"平均预热时间: {avg_warm_up_time:.2f} 秒")
        print(f"检测时间: {detect_time:.2f} 秒")
        print(f"总时间: {load_time + detect_time:.2f} 秒")
        
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
                    
                    # 将标签放到框内左上角
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    # 在框内绘制标签背景
                    cv2.rectangle(custom_result, (x1, y1), (x1+text_size[0], y1+text_size[1]+5), (0, 0, 255), -1)
                    # 在框内绘制标签文字
                    cv2.putText(custom_result, label, (x1, y1+text_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
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
            
            # 将信息放在右下角
            padding = 10
            text_height = 30
            line_height = text_height
            
            # 计算文本框位置（右下角）
            text_widths = [cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in text_lines]
            max_text_width = max(text_widths) + padding * 2
            
            # 计算背景框的位置
            bg_height = (len(text_lines)) * line_height + padding
            bg_x = custom_result.shape[1] - max_text_width
            bg_y = custom_result.shape[0] - bg_height
            
            # Add semi-transparent black background
            overlay = custom_result.copy()
            cv2.rectangle(overlay, (bg_x, bg_y), (custom_result.shape[1], custom_result.shape[0]), (0, 0, 0), -1)
            alpha = 0.6  # Transparency
            cv2.addWeighted(overlay, alpha, custom_result, 1 - alpha, 0, custom_result)
            
            # Add text
            for i, line in enumerate(text_lines):
                y_pos = bg_y + (i + 1) * line_height
                x_pos = bg_x + padding
                cv2.putText(custom_result, line, (x_pos, y_pos), font, font_scale, color, thickness)
            
            # Save custom annotated image
            cv2.imwrite(save_path, custom_result)
            
            # Calculate post-processing time
            postprocess_time = time.time() - postprocess_start_time
            print(f"后处理时间: {postprocess_time:.2f} 秒")
            print(f"总处理时间: {load_time + detect_time + postprocess_time:.2f} 秒")
            print(f"\n结果图像已保存至: {save_path}")
            
            # Auto open image
            if auto_open:
                open_image(save_path)
                
        return has_violence, results, load_time, detect_time, save_path
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, 0, 0, None

def export_to_onnx(model_path, output_path=None):
    """Convert PT model to ONNX format"""
    try:
        if not os.path.exists(model_path):
            print(f"错误：模型文件不存在 - {model_path}")
            return False
            
        # If output path not specified, use original filename + .onnx
        if output_path is None:
            output_path = os.path.splitext(model_path)[0] + ".onnx"
            
        print(f"加载模型: {model_path}")
        model = YOLO(model_path)
        
        print(f"导出模型为ONNX格式: {output_path}")
        model.export(format="onnx", opset=12)
        
        # Check if export was successful
        expected_path = os.path.splitext(model_path)[0] + ".onnx"
        if os.path.exists(expected_path):
            print(f"ONNX模型导出成功: {expected_path}")
            # If user specified a different output path, rename the file
            if expected_path != output_path:
                os.rename(expected_path, output_path)
                print(f"模型已重命名为: {output_path}")
            return True
        else:
            print(f"错误：ONNX模型导出失败")
            return False
            
    except Exception as e:
        print(f"ONNX导出过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) < 2:
        # 默认使用nano模型
        model_path = "V02_Musa_Violence/Yolo_nano_weights.pt"
        print(f"使用默认模型: {model_path}")
    else:
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
    
    # Whether to use warm-up
    warm_up = "--no-warm-up" not in sys.argv
    warm_up_rounds = 2
    
    if "--warm-up-rounds" in sys.argv:
        try:
            rounds_idx = sys.argv.index("--warm-up-rounds")
            if rounds_idx + 1 < len(sys.argv):
                warm_up_rounds = int(sys.argv[rounds_idx + 1])
        except (ValueError, IndexError):
            pass
    
    # Execute test
    test_model(model_path, image_path, save_path, auto_open=auto_open, warm_up=warm_up, warm_up_rounds=warm_up_rounds)

if __name__ == "__main__":
    main() 