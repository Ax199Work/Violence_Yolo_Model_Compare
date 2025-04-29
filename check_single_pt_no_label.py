#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Testing Script - Test performance and detection results of a single PyTorch model without visualization
Usage:
    python check_single_pt_no_label.py [model_path] [image_path]
"""

import os
import sys
import time
import cv2
from pathlib import Path

# Default test image
DEFAULT_TEST_IMAGE = "test_image/1.jpg"

# Import ultralytics library
try:
    from ultralytics import YOLO
    print("成功导入ultralytics库")
except ImportError:
    print("错误：无法导入ultralytics库")
    print("请安装: pip install ultralytics")
    sys.exit(1)

def test_model(model_path, image_path, model_name=None, warm_up=True, warm_up_rounds=2):
    """Test YOLO model without visualization"""
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
            
        # Run detection
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
                
        return has_violence, results, load_time, detect_time
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, 0, 0

def main():
    if len(sys.argv) < 2:
        # 默认使用nano模型
        model_path = "V02_Musa_Violence/Yolo_nano_weights.pt"
        print(f"使用默认模型: {model_path}")
    else:
        model_path = sys.argv[1]
    
    # Determine image path
    image_path = DEFAULT_TEST_IMAGE
    for arg in sys.argv[2:]:
        if not arg.startswith("--") and os.path.exists(arg):
            image_path = arg
            break
    
    # Execute test
    test_model(model_path, image_path, warm_up=True, warm_up_rounds=2)

if __name__ == "__main__":
    main() 