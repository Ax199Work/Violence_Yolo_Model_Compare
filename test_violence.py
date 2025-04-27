#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型测试脚本 - 可测试V01和V02目录下的模型
可同时测试V01和V02的small和nano模型并比较性能
"""

import os
import cv2
import sys
import time
from pathlib import Path

# V02模型路径
SMALL_MODEL_PATH = "V02_Musa_Violence/yolo_small_weights.pt"  # small模型
NANO_MODEL_PATH = "V02_Musa_Violence/Yolo_nano_weights.pt"  # nano模型
# V01模型路径
V01_MODEL_PATH = "V01_Yasser_Violence/weight/best.pt"  # V01模型
TEST_IMAGE = "test_image/1.jpg"

try:
    from ultralytics import YOLO
    print("成功导入ultralytics库")
except ImportError:
    print("错误: 无法导入ultralytics库")
    print("请安装: pip install ultralytics")
    sys.exit(1)

def test_model(model_path, image_path, save_path=None, model_name="未命名"):
    """测试YOLO模型"""
    try:
        # 检查文件是否存在
        if not os.path.exists(model_path):
            print(f"错误: 模型文件不存在 - {model_path}")
            return
            
        if not os.path.exists(image_path):
            print(f"错误: 图像文件不存在 - {image_path}")
            return
            
        # 加载模型
        print(f"正在加载模型: {model_name} ({model_path})")
        load_start_time = time.time()
        model = YOLO(model_path)
        load_time = time.time() - load_start_time
        print(f"模型加载时间: {load_time:.2f} 秒")
        
        # 读取图像
        print(f"正在读取图像: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误: 无法读取图像 {image_path}")
            return
            
        # 运行检测
        print("开始检测...")
        detect_start_time = time.time()
        results = model(image)
        detect_time = time.time() - detect_start_time
        print(f"检测时间: {detect_time:.2f} 秒")
        
        # 处理检测结果
        has_violence = False
        violence_count = 0
        nonviolence_count = 0
        
        # 处理结果
        print("\n===== 检测结果 =====")
        for r in results:
            boxes = r.boxes
            for i, box in enumerate(boxes):
                # 获取类别ID和置信度
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 获取类别名称（如果可用）
                if hasattr(r, 'names') and cls_id in r.names:
                    class_name = r.names[cls_id]
                else:
                    class_name = f"class_{cls_id}"
                
                # 打印检测结果
                print(f"对象 {i+1}: {class_name} (置信度: {conf:.2f}), 位置: [{x1}, {y1}, {x2}, {y2}]")
                
                # 暴力检测逻辑
                if cls_id == 1 or class_name.lower() in ['violence', 'fight', '暴力']:
                    has_violence = True
                    violence_count += 1
                else:
                    nonviolence_count += 1
                    
        # 打印检测总结
        print("\n===== 检测总结 =====")
        if violence_count == 0 and nonviolence_count == 0:
            print("未检测到任何对象")
        elif has_violence:
            print(f"⚠️  警告: 图像中检测到暴力内容!")
            print(f"检测到 {violence_count} 个暴力对象, {nonviolence_count} 个非暴力对象")
        else:
            print("✓ 图像安全: 未检测到暴力内容")
            print(f"检测到 {nonviolence_count} 个非暴力对象")
        
        # 打印性能信息
        print(f"\n===== 性能信息 =====")
        print(f"模型加载时间: {load_time:.2f} 秒")
        print(f"检测时间: {detect_time:.2f} 秒")
        print(f"总时间: {load_time + detect_time:.2f} 秒")
            
        return has_violence, results, load_time, detect_time
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        return None, None, 0, 0

# 测试V02模型的简化函数
def test_v02_model(model_path, image_path, save_path=None):
    """测试V02模型 - 兼容旧版本调用"""
    model_name = "V02-Small" if "small" in model_path.lower() else "V02-Nano"
    return test_model(model_path, image_path, save_path, model_name)

# 测试V01模型的简化函数
def test_v01_model(image_path, save_path=None):
    """测试V01模型"""
    return test_model(V01_MODEL_PATH, image_path, save_path, "V01-Yasser")

def compare_models(image_path=TEST_IMAGE, include_v01=True):
    """比较所有模型的性能"""
    print("\n" + "="*50)
    print("开始模型比较测试...")
    print("="*50 + "\n")
    
    results = {}
    
    # 测试V01模型
    if include_v01:
        print("\n" + "*"*30)
        print("测试 V01 模型")
        print("*"*30)
        v01_result = test_v01_model(image_path)
        if v01_result:
            _, _, v01_load_time, v01_detect_time = v01_result
            results["V01"] = {"load": v01_load_time, "detect": v01_detect_time, "total": v01_load_time + v01_detect_time}
        
        print("\n" + "="*50 + "\n")
    
    # 测试V02-small模型
    print("\n" + "*"*30)
    print("测试 V02-SMALL 模型")
    print("*"*30)
    small_result = test_v02_model(SMALL_MODEL_PATH, image_path)
    if small_result:
        _, _, small_load_time, small_detect_time = small_result
        results["V02-SMALL"] = {"load": small_load_time, "detect": small_detect_time, "total": small_load_time + small_detect_time}
    
    print("\n" + "="*50 + "\n")
    
    # 测试V02-nano模型
    print("\n" + "*"*30)
    print("测试 V02-NANO 模型")
    print("*"*30)
    nano_result = test_v02_model(NANO_MODEL_PATH, image_path)
    if nano_result:
        _, _, nano_load_time, nano_detect_time = nano_result
        results["V02-NANO"] = {"load": nano_load_time, "detect": nano_detect_time, "total": nano_load_time + nano_detect_time}
    
    # 比较结果
    if results:
        print("\n" + "="*70)
        print("模型性能比较")
        print("="*70)
        print(f"{'模型类型':<12} {'加载时间(秒)':<15} {'检测时间(秒)':<15} {'总时间(秒)':<15}")
        print("-"*70)
        
        for model, times in results.items():
            print(f"{model:<12} {times['load']:<15.2f} {times['detect']:<15.2f} {times['total']:<15.2f}")
        
        print("-"*70)
        
        # 如果有V01模型数据，计算与其他模型的比较
        if include_v01 and "V01" in results:
            v01_total = results["V01"]["total"]
            print(f"\nV01模型与其他模型的性能比较:")
            for model in ["V02-SMALL", "V02-NANO"]:
                if model in results:
                    speedup = v01_total / results[model]["total"] if results[model]["total"] > 0 else 0
                    if speedup > 1:
                        print(f"{model} 比 V01 快 {speedup:.2f}x")
                    else:
                        print(f"{model} 比 V01 慢 {1/speedup:.2f}x")
        
        # V02 small和nano的比较
        if "V02-SMALL" in results and "V02-NANO" in results:
            small_total = results["V02-SMALL"]["total"]
            nano_total = results["V02-NANO"]["total"]
            speedup = small_total / nano_total if nano_total > 0 else 0
            
            print(f"\nV02-NANO模型比V02-SMALL模型:")
            if speedup > 1:
                print(f"总体速度快 {speedup:.2f}x")
            else:
                print(f"总体速度慢 {1/speedup:.2f}x")

def main():
    if len(sys.argv) > 1:
        # 解析命令行参数
        if sys.argv[1] == "--compare":
            # 比较模式
            image_path = sys.argv[2] if len(sys.argv) > 2 else TEST_IMAGE
            include_v01 = True
            if len(sys.argv) > 3 and sys.argv[3] == "--no-v01":
                include_v01 = False
            compare_models(image_path, include_v01)
        elif sys.argv[1] == "--v01":
            # 仅测试V01模型
            image_path = sys.argv[2] if len(sys.argv) > 2 else TEST_IMAGE
            test_v01_model(image_path)
        elif sys.argv[1] == "--v02-small":
            # 仅测试V02-Small模型
            image_path = sys.argv[2] if len(sys.argv) > 2 else TEST_IMAGE
            test_v02_model(SMALL_MODEL_PATH, image_path)
        elif sys.argv[1] == "--v02-nano":
            # 仅测试V02-Nano模型
            image_path = sys.argv[2] if len(sys.argv) > 2 else TEST_IMAGE
            test_v02_model(NANO_MODEL_PATH, image_path)
        else:
            # 使用指定的模型文件
            model_path = sys.argv[1]
            image_path = sys.argv[2] if len(sys.argv) > 2 else TEST_IMAGE
            test_model(model_path, image_path, model_name="自定义模型")
    else:
        # 默认执行比较
        compare_models()

if __name__ == "__main__":
    main() 