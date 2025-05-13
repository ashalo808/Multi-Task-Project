import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.tasks.base_task import BaseTask


class NLIDataset(Dataset):
    """自然语言推理任务的数据集类"""
    
    def __init__(self, premises: List[str], hypotheses: List[str], labels: List[int], tokenizer, max_length: int):
        """
        初始化NLI数据集
        
        Args:
            premises: 前提句子列表
            hypotheses: 假设句子列表
            labels: 标签列表(蕴含、矛盾、中性)
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.premises)
    
    def __getitem__(self, idx):
        premise = self.premises[idx]
        hypothesis = self.hypotheses[idx]
        label = self.labels[idx]
        
        # 同时编码前提和假设
        encoding = self.tokenizer(
            premise,
            hypothesis,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 移除批次维度
        for key in encoding:
            encoding[key] = encoding[key].squeeze(0)
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class NLITask(BaseTask):
    """自然语言推理任务的基类"""
    
    def initialize(self):
        """初始化NLI任务"""
        super().initialize()
        
        # 加载标签映射
        self.label_map = self._load_label_map()
        self.num_labels = len(self.label_map)
        
        # 设置最大序列长度
        self.max_seq_length = self.task_config.get("max_seq_length", 256)
        
        # 获取tokenizer
        self.tokenizer = self._get_tokenizer()
    
    def _load_label_map(self) -> Dict[str, int]:
        """
        加载标签映射
        
        Returns:
            标签到索引的映射字典
        """
        label_map_path = os.path.join(self.data_dir, "label_map.json")
        if os.path.exists(label_map_path):
            return self.load_json_data(label_map_path)
        else:
            # 如果没有找到标签映射文件，使用默认的NLI标签映射
            return {
                "entailment": 0,   # 蕴含
                "contradiction": 1, # 矛盾
                "neutral": 2        # 中性
            }
    
    def _get_tokenizer(self):
        """
        获取tokenizer，应在子类实现
        
        Returns:
            适用于当前任务的tokenizer
        """
        # 默认实现，实际应在子类中实现
        raise NotImplementedError("必须在子类中实现_get_tokenizer方法")
    
    def preprocess_data(self, split: str) -> Dataset:
        """
        预处理NLI数据
        
        Args:
            split: 数据集划分类型 ('train', 'dev', 'test')
        
        Returns:
            处理好的Dataset对象
        """
        # 根据划分确定文件名
        if split == 'train':
            # 对于train可能需要加载多个文件并合并
            files = [
                os.path.join(self.data_dir, 'raw', f'pCLUE_train_{i}.json') 
                for i in [1, 3, 9]  # 假设这些是需要的文件
            ]
        elif split == 'dev':
            files = [os.path.join(self.data_dir, 'raw', 'pCLUE_dev.json')]
        else:  # test
            files = [
                os.path.join(self.data_dir, 'raw', f'pCLUE_test_{i}.json') 
                for i in [1, 2]
            ]
        
        # 加载并合并数据
        all_data = []
        for file_path in files:
            if os.path.exists(file_path):
                data = self.load_json_data(file_path)
                all_data.extend(data)
        
        # 筛选特定任务类型的数据
        task_type = self._get_task_type()  # 如'ocnli'
        task_data = [item for item in all_data if item.get('type') == task_type]
        
        premises = []
        hypotheses = []
        labels = []
        
        for item in task_data:
            premise = self._extract_premise(item)
            hypothesis = self._extract_hypothesis(item)
            label = self._extract_label(item)
            
            if premise and hypothesis and label is not None:
                premises.append(premise)
                hypotheses.append(hypothesis)
                labels.append(label)
        
        return NLIDataset(premises, hypotheses, labels, self.tokenizer, self.max_seq_length)
    
    def _extract_premise(self, item: Dict[str, Any]) -> str:
        """
        从数据条目中提取前提句子，子类可重写此方法
        
        Args:
            item: 数据条目
        
        Returns:
            提取的前提句子
        """
        # 默认实现尝试常见字段名
        premise = item.get("premise") or item.get("sentence1") or item.get("text1") or ""
        return premise
    
    def _extract_hypothesis(self, item: Dict[str, Any]) -> str:
        """
        从数据条目中提取假设句子，子类可重写此方法
        
        Args:
            item: 数据条目
        
        Returns:
            提取的假设句子
        """
        # 默认实现尝试常见字段名
        hypothesis = item.get("hypothesis") or item.get("sentence2") or item.get("text2") or ""
        return hypothesis
    
    def _extract_label(self, item: Dict[str, Any]) -> int:
        """
        从数据条目中提取并转换标签，子类可重写此方法
        
        Args:
            item: 数据条目
        
        Returns:
            转换后的数值标签
        """
        label_str = item.get("label")
        if label_str is None:
            return None
        
        # 如果是字符串标签，转换为数值
        if isinstance(label_str, str):
            if label_str in self.label_map:
                return self.label_map[label_str]
            else:
                raise ValueError(f"未知标签: {label_str}")
        else:
            # 如果已经是数值，直接返回
            return label_str
    
    def get_dataloader(self, split: str, batch_size: int, shuffle: bool = True) -> DataLoader:
        """
        获取数据加载器
        
        Args:
            split: 数据集划分
            batch_size: 批次大小
            shuffle: 是否打乱数据
        
        Returns:
            DataLoader对象
        """
        dataset = self.preprocess_data(split)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.task_config.get("num_workers", 0)
        )
    
    def prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        准备模型输入
        
        Args:
            batch: 数据批次
        
        Returns:
            模型输入字典
        """
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        
        # 如果有token_type_ids则加入
        if "token_type_ids" in batch:
            inputs["token_type_ids"] = batch["token_type_ids"]
        
        return inputs
    
    def compute_loss(self, model_outputs: Any, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算NLI损失
        
        Args:
            model_outputs: 模型输出
            batch: 数据批次
        
        Returns:
            损失张量
        """
        logits = model_outputs.logits
        labels = batch["labels"]
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
    
    def compute_metrics(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        计算NLI评估指标
        
        Args:
            predictions: 预测结果
            labels: 真实标签
        
        Returns:
            包含评估指标的字典
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # 对于logits输入，取最大值的索引作为预测标签
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            preds = np.argmax(predictions, axis=1)
        else:
            preds = predictions
        
        # 计算准确率
        accuracy = accuracy_score(labels, preds)
        
        # 计算精确率、召回率和F1值
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, 
            preds, 
            average='weighted'
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }