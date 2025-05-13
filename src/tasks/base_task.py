from abc import ABC, abstractmethod
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional, Union


class BaseTask(ABC):
    """
    任务基础抽象类，定义了所有任务的通用接口
    """

    def __init__(self, data_dir: str, model_config: Dict[str, Any], task_config: Dict[str, Any]):
        """
        初始化任务处理器
        
        Args:
            data_dir: 数据目录路径
            model_config: 模型配置参数
            task_config: 任务特定配置参数
        """
        self.data_dir = data_dir
        self.model_config = model_config
        self.task_config = task_config
        self.initialize()
    
    def initialize(self):
        """
        任务初始化逻辑，可在子类中重写
        """
        pass
    
    @staticmethod
    @abstractmethod
    def _get_task_type_static() -> str:
        """
        静态方法返回任务类型，用于任务注册
        """
        pass
    
    @abstractmethod
    def _get_task_type(self) -> str:
        """
        返回任务类型标识
        """
        pass
    
    @abstractmethod
    def preprocess_data(self, split: str) -> Dataset:
        """
        预处理数据并返回Dataset对象
        
        Args:
            split: 数据集划分，如"train", "dev", "test"
        
        Returns:
            处理好的Dataset对象
        """
        pass
    
    @abstractmethod
    def get_dataloader(self, split: str, batch_size: int, shuffle: bool = True) -> DataLoader:
        """
        获取数据加载器
        
        Args:
            split: 数据集划分，如"train", "dev", "test"
            batch_size: 批次大小
            shuffle: 是否打乱数据
        
        Returns:
            DataLoader对象
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            predictions: 模型的预测结果
            labels: 真实标签
        
        Returns:
            包含各项评估指标的字典
        """
        pass
    
    @abstractmethod
    def prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        准备传入模型的输入
        
        Args:
            batch: 数据批次
        
        Returns:
            处理后的模型输入
        """
        pass
    
    @abstractmethod
    def compute_loss(self, model_outputs: Any, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算任务特定的损失函数
        
        Args:
            model_outputs: 模型输出
            batch: 数据批次
        
        Returns:
            损失值
        """
        pass
    
    def get_dataset_path(self, split: str) -> str:
        """
        获取数据集文件路径
        
        Args:
            split: 数据集划分，如"train", "dev", "test"
        
        Returns:
            数据文件的完整路径
        """
        return os.path.join(self.data_dir, f"{split}.json")
    
    def load_json_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载JSON格式的数据文件
        
        Args:
            file_path: JSON文件路径
        
        Returns:
            加载的数据列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在：{file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def save_predictions(self, predictions: List[Any], output_file: str):
        """
        保存预测结果到文件
        
        Args:
            predictions: 预测结果列表
            output_file: 输出文件路径
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    def log_metrics(self, metrics: Dict[str, float], split: str):
        """
        记录评估指标
        
        Args:
            metrics: 评估指标字典
            split: 数据集划分
        """
        print(f"--- {self._get_task_type()} {split} metrics ---")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")