# 多任务学习项目 - 任务开发指南

## 项目介绍
本项目实现了一个基于共享编码器的多任务学习框架，能够同时处理多种自然语言处理任务。框架的核心是采用共享编码器架构，让不同类型的NLP任务共享底层语义表示，同时通过任务特定的适配器处理差异化需求。

## 项目文件结构

以下是完整的文件结构说明，包括每个代码部分应该放在哪个文件中，以及文件应该位于哪个文件夹：

```
C:\VSCode\MachineLearning\
│
├── src/                             # 源代码根目录
│   ├── models/                      # 模型相关代码
│   │   ├── __init__.py
│   │   └── core_seq2seq.py          # 共享编码器模型实现
│   │
│   ├── prompts/                     # 提示模板相关代码
│   │   ├── __init__.py
│   │   └── template_manager.py      # 提示模板管理系统
│   │
│   ├── tasks/                       # 任务处理器代码
│   │   ├── __init__.py
│   │   ├── base_task.py             # 任务基类
│   │   ├── classification_task.py   # 分类任务基类
│   │   ├── nli_task.py              # NLI任务基类
│   │   ├── mrc_task.py              # 阅读理解任务基类
│   │   ├── task_manager.py          # 任务管理器
│   │   │
│   │   └── examples/                # 任务实现示例
│   │       ├── __init__.py
│   │       ├── tnews_task.py        # TNews任务示例
│   │       ├── ocnli_task.py        # OCNLI任务示例
│   │       └── chid_task.py         # CHID任务示例
│   │
│   ├── trainers/                    # 训练器代码
│   │   ├── __init__.py
│   │   └── multitask_trainer.py     # 多任务训练器
│   │
│   ├── tests/                       # 测试代码
│   │   ├── __init__.py
│   │   ├── test_encoder.py          # 编码器测试
│   │   ├── mock_data_loader.py      # 模拟数据加载器
│   │   ├── mock_trainer.py          # 模拟训练器
│   │   └── integrated_test.py       # 集成测试
│   │
│   ├── config/                      # 配置文件
│   │   ├── model_arch_config.yaml        # 模型架构配置
│   │   └── prompt_templates_config.yaml  # 任务模板配置
│   │
│   ├── utils/                       # 工具函数
│   │   ├── __init__.py
│   │   └── common_utils.py          # 通用工具函数
│   │
│   ├── run_multitask.py             # 多任务运行脚本
│   └── __init__.py
│
├── configs/                         # 项目配置文件
│   └── multitask_config.yaml        # 多任务配置
│
├── data/                            # 数据目录
│   ├── processed/                   # 处理后的数据
│   │   ├── data_stats.json         # 数据统计信息
│   │   └── pCLUE_dev.json          # 处理后的开发集
│   │
│   └── raw/                         # 原始数据
│       ├── pCLUE_dev.json          # 开发集
│       ├── pCLUE_test_1.json       # 测试集部分1
│       ├── pCLUE_test_2.json       # 测试集部分2
│       ├── pCLUE_test_public_1.json # 公开测试集部分1
│       ├── pCLUE_test_public_2.json # 公开测试集部分2
│       ├── pCLUE_train_1.json      # 训练集部分1 (包含tnews任务)
│       ├── pCLUE_train_3.json      # 训练集部分3 (包含ocnli任务)
│       └── pCLUE_train_9.json      # 训练集部分3 (包含chid任务)
│
├── outputs/                         # 输出目录
│   └── README.md                    # 输出目录说明
│
├── README.md                        # 项目主README
└── requirements.txt                 # 依赖项
```

### 各代码文件对应关系

| 代码部分 | 文件路径 |
|---------|---------|
| 任务处理接口 | `src/tasks/base_task.py` | 定义所有任务的基本接口 |
| 多任务训练器 | `src/trainers/multitask_trainer.py` | 处理多任务训练逻辑 |
| 任务注册和管理器 | `src/tasks/task_manager.py` | 负责发现、注册和管理任务 |
| 任务实现示例 | `src/tasks/examples/` | 包含已实现的任务示例 |
| 主程序脚本 | `src/run_multitask.py` | 主程序入口 |
| 示例配置文件 | `configs/multitask_config.yaml` | 定义多任务训练的配置 |
| 共享编码器模型 | `core_seq2seq.py` | 包含共享编码器的核心实现 |
| 分类任务基类 | `src/tasks/classification_task.py` |
| NLI任务基类 | `src/tasks/nli_task.py` |
| 阅读理解任务基类 | `src/tasks/mrc_task.py` |
| 提示模板管理系统 | `template_manager.py `|
| 编码器测试 | `test_encoder.py` |
| 模拟数据加载器 | `mock_data_loader.py` |
| 模拟训练器 | `mock_trainer.py` |
| 集成测试 | `integrated_test.py` |


## 使用共享编码器架构的步骤

### 1. 实现新任务类

要添加新的任务，需要创建一个新的任务类并继承适当的基类：

```python
# 在src/tasks/examples/目录下创建任务文件，例如custom_task.py
from src.tasks.classification_task import ClassificationTask  # 或其他合适的基类

class CustomTask(ClassificationTask):
    """自定义任务的实现"""
    
    @staticmethod
    def _get_task_type_static() -> str:
        """返回任务类型标识，用于任务注册"""
        return "custom_task"  # 这个标识将在配置文件中使用
    
    def _get_task_type(self) -> str:
        """返回任务类型标识"""
        return "custom_task"
    
    def _extract_text(self, item):
        """根据数据结构提取文本内容"""
        # 示例实现
        return item.get("text", "") or item.get("content", "")
    
    def _extract_label(self, item):
        """处理任务标签"""
        # 示例实现
        label = item.get("label")
        if label in self.label_map:
            return self.label_map[label]
        return None
```

### 2. 配置新任务

在配置文件中添加新任务的配置信息：

```yaml
# 在configs/multitask_config.yaml中添加
tasks:
  # 已有任务...
  
  custom_task_name:  # 任务名称
    type: "custom_task"  # 必须与_get_task_type_static()返回值匹配
    data_subdir: "custom_data"  # 可选，指定数据子目录
    description: "自定义任务描述"
    metrics:  # 评估指标
      - "accuracy"
      - "f1"
    params:  # 任务特定参数
      max_seq_length: 256
      eval_batch_size: 32
```

### 3. 准备数据

确保数据格式正确，并放置在适当的位置：

- 使用统一数据格式：数据项需要包含`type`字段，用于识别任务类型
- 文件位置：将数据文件放在raw目录下
- 数据分割：确保训练集、验证集和测试集准备就绪

对于使用共享数据文件的情况，任务类会根据`type`字段筛选属于该任务的数据样本。

### 4. 运行多任务训练

使用以下命令运行多任务训练：

```bash
# 训练所有任务
python -m src.run_multitask --config ./configs/multitask_config.yaml --do_train --do_eval

# 只训练特定任务
python -m src.run_multitask --config ./configs/multitask_config.yaml --do_train --task custom_task_name

# 指定输出目录
python -m src.run_multitask --config ./configs/multitask_config.yaml --do_train --output_dir ./outputs/experiment_name
```

### 5. 评估模型性能

```bash
# 评估所有任务
python -m src.run_multitask --config ./configs/multitask_config.yaml --do_eval

# 评估特定任务
python -m src.run_multitask --config ./configs/multitask_config.yaml --do_eval --task custom_task_name
```

## 优化共享编码器性能

如果某些任务性能不如预期，可以尝试以下优化方法：

### 调整任务适配器大小

在配置文件中修改适配器容量：

```yaml
model:
  # 基本配置...
  task_specific_params:
    adapter_size: 64  # 默认适配器大小
    task_adapters:
      nli: 128  # 为NLI任务设置更大的适配器
      custom_task: 96  # 为自定义任务设置适配器大小
```

### 调整任务采样权重

某些任务可能需要更多关注，可以调整采样权重：

```yaml
training:
  # 其他训练参数...
  task_sampling_weights:
    tnews: 1.0
    ocnli: 1.5  # 增加NLI任务的采样权重
    custom_task: 0.8
```

### 分层冻结策略

可以尝试在不同层采用不同的参数冻结策略，修改编码器配置：

```yaml
model:
  # 其他配置...
  freeze_layers:
    - 0  # 冻结第一层
    - 1  # 冻结第二层
  task_specific_unfrozen:
    nli: [10, 11]  # NLI任务专门解冻顶层
```

## 测试共享编码器实现

使用内置的测试工具验证共享编码器的功能：

```bash
# 运行编码器单元测试
python -m src.tests.test_encoder

# 运行集成测试
python -m src.tests.integrated_test
```

测试将验证编码器能够正确处理不同任务，并产生有效的共享表示。

通过这套框架，可以轻松地实现不同类型的NLP任务，并利用共享编码器架构进行高效的多任务学习。共享编码器能够捕获通用的语言表示，而任务特定的适配器则处理每个任务的独特需求，从而在保持模型规模合理的同时提升各任务的性能。