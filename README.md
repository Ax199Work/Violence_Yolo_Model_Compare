# 暴力检测模型比较框架

这个项目用于测试和比较不同的基于YOLOv8的暴力检测模型。目前集成了两个模型：

1. V01 (Yasser) - 来自 [Violence_detection_YOLOv8](https://github.com/YasserElj/Violence_detection_YOLOv8)
2. V02 (Musa) - 来自 [Fight-Violence-detection-yolov8](https://github.com/Musawer1214/Fight-Violence-detection-yolov8)

## 项目结构

项目使用Git子模块来管理依赖的模型：

```
Violence_Yolo/
├── V01_Yasser_Violence/    # Yasser的暴力检测模型（子模块）
├── V02_Musa_Violence/      # Musa的暴力检测模型（子模块）
├── test_violence.py        # 模型测试和比较脚本
└── test_image/             # 测试图像目录
```

## 安装与设置

### 克隆项目及其子模块

```bash
# 克隆主项目和所有子模块
git clone --recurse-submodules https://your-repo-url.git
cd Violence_Yolo

# 或者，如果你已经克隆了项目但没有子模块
git submodule update --init --recursive
```

### 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 测试单个模型

```bash
# 测试V01模型
python test_violence.py --v01 [图像路径]

# 测试V02 Small模型
python test_violence.py --v02-small [图像路径]

# 测试V02 Nano模型
python test_violence.py --v02-nano [图像路径]
```

### 比较所有模型

```bash
# 比较所有模型
python test_violence.py --compare [图像路径]

# 比较所有模型但不包括V01
python test_violence.py --compare [图像路径] --no-v01
```

### 默认模式

直接运行脚本将默认执行所有模型的比较：

```bash
python test_violence.py
```

## 子模块管理

### 更新子模块

```bash
# 更新所有子模块到最新的远程提交
git submodule update --remote

# 更新特定子模块
git submodule update --remote V01_Yasser_Violence
```

### 添加新的子模块

```bash
git submodule add [仓库URL] [本地路径]
```

## 注意事项

1. 模型文件较大，首次克隆可能需要较长时间。
2. V01 模型预期的权重文件位于 `V01_Yasser_Violence/weight/best.pt`。
3. V02 模型预期的权重文件位于 `V02_Musa_Violence/yolo_small_weights.pt` 和 `V02_Musa_Violence/Yolo_nano_weights.pt`。 