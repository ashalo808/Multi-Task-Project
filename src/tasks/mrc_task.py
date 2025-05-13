import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import Counter

from src.tasks.base_task import BaseTask


class MRCDataset(Dataset):
    """机器阅读理解任务的数据集类"""
    
    def __init__(
        self, 
        contexts: List[str], 
        questions: List[str], 
        answers: List[Dict[str, Any]],
        tokenizer,
        max_length: int,
        doc_stride: int = 128,
        max_question_length: int = 64,
        is_training: bool = True
    ):
        """
        初始化MRC数据集
        
        Args:
            contexts: 上下文文本列表
            questions: 问题列表
            answers: 答案列表，每个答案包含开始位置、结束位置和文本
            tokenizer: 分词器
            max_length: 最大序列长度
            doc_stride: 文档滑动窗口步长
            max_question_length: 问题最大长度
            is_training: 是否为训练模式
        """
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.max_question_length = max_question_length
        self.is_training = is_training
        
        # 预处理特征
        self.features = self._convert_to_features()
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        
        return {
            "input_ids": feature["input_ids"],
            "attention_mask": feature["attention_mask"],
            "token_type_ids": feature.get("token_type_ids", torch.zeros_like(feature["input_ids"])),
            "start_positions": feature.get("start_positions", torch.tensor(0)),
            "end_positions": feature.get("end_positions", torch.tensor(0))
        }
    
    def _convert_to_features(self):
        """将原始数据转换为模型输入特征"""
        features = []
        
        for i, (context, question, answer) in enumerate(zip(self.contexts, self.questions, self.answers)):
            # 对问题和上下文进行编码
            tokenized_example = self.tokenizer(
                question,
                context,
                truncation="only_second",
                max_length=self.max_length,
                stride=self.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # 如果有多个特征（长文本会被切分）
            sample_mapping = tokenized_example.pop("overflow_to_sample_mapping")
            offset_mapping = tokenized_example.pop("offset_mapping")
            
            # 对于训练模式，我们需要标记答案的位置
            if self.is_training:
                answer_text = answer.get("text", "")
                answer_start_char = answer.get("answer_start", -1)
                answer_end_char = answer_start_char + len(answer_text) if answer_start_char != -1 else -1
                
                for feature_idx, sample_idx in enumerate(sample_mapping):
                    # 跳过不是当前样本的特征
                    if sample_idx != i:
                        continue
                    
                    sequence_ids = tokenized_example.sequence_ids(feature_idx)
                    
                    # 查找上下文的起始和结束位置
                    context_start = 0
                    while sequence_ids[context_start] != 1:
                        context_start += 1
                    context_end = len(sequence_ids) - 1
                    while sequence_ids[context_end] != 1:
                        context_end -= 1
                    
                    # 如果答案不在当前特征的上下文范围内，则将标签设为0
                    if (offset_mapping[feature_idx][context_start][0] > answer_end_char or
                            offset_mapping[feature_idx][context_end][1] < answer_start_char):
                        start_position = 0
                        end_position = 0
                    else:
                        # 否则，找到答案的开始和结束位置
                        idx = context_start
                        while idx <= context_end and offset_mapping[feature_idx][idx][0] <= answer_start_char:
                            idx += 1
                        start_position = idx - 1
                        
                        idx = context_end
                        while idx >= context_start and offset_mapping[feature_idx][idx][1] >= answer_end_char:
                            idx -= 1
                        end_position = idx + 1
                    
                    feature = {
                        "input_ids": tokenized_example["input_ids"][feature_idx],
                        "attention_mask": tokenized_example["attention_mask"][feature_idx],
                        "start_positions": torch.tensor(start_position),
                        "end_positions": torch.tensor(end_position)
                    }
                    
                    if "token_type_ids" in tokenized_example:
                        feature["token_type_ids"] = tokenized_example["token_type_ids"][feature_idx]
                    
                    features.append(feature)
            else:
                # 对于评估/预测模式，我们保留偏移映射以便后处理
                for feature_idx, sample_idx in enumerate(sample_mapping):
                    # 跳过不是当前样本的特征
                    if sample_idx != i:
                        continue
                    
                    feature = {
                        "input_ids": tokenized_example["input_ids"][feature_idx],
                        "attention_mask": tokenized_example["attention_mask"][feature_idx],
                        "offset_mapping": offset_mapping[feature_idx],
                        "example_id": i
                    }
                    
                    if "token_type_ids" in tokenized_example:
                        feature["token_type_ids"] = tokenized_example["token_type_ids"][feature_idx]
                    
                    features.append(feature)
        
        return features


class MRCTask(BaseTask):
    """阅读理解任务的基类"""
    
    def initialize(self):
        """初始化MRC任务"""
        super().initialize()
        
        # 设置最大序列长度
        self.max_seq_length = self.task_config.get("max_seq_length", 384)
        
        # 设置文档滑动窗口步长
        self.doc_stride = self.task_config.get("doc_stride", 128)
        
        # 设置问题最大长度
        self.max_question_length = self.task_config.get("max_question_length", 64)
        
        # 获取tokenizer
        self.tokenizer = self._get_tokenizer()
    
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
        预处理MRC数据
        
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
        task_type = self._get_task_type()  # 如'chid'
        task_data = [item for item in all_data if item.get('type') == task_type]
        
        contexts = []
        questions = []
        answers = []
        
        for item in task_data:
            context = self._extract_context(item)
            question = self._extract_question(item)
            answer = self._extract_answer(item)
            
            if context and question and answer:
                contexts.append(context)
                questions.append(question)
                answers.append(answer)
        
        # 创建数据集，训练集使用带标签的处理，其他使用不带标签的处理
        return MRCDataset(
            contexts=contexts,
            questions=questions,
            answers=answers,
            tokenizer=self.tokenizer,
            max_length=self.max_seq_length,
            doc_stride=self.doc_stride,
            max_question_length=self.max_question_length,
            is_training=(split == 'train')
        )
    
    def _extract_context(self, item: Dict[str, Any]) -> str:
        """
        从数据条目中提取上下文，子类可重写此方法
        
        Args:
            item: 数据条目
        
        Returns:
            提取的上下文文本
        """
        # 默认实现尝试常见字段名
        context = item.get("context") or item.get("passage") or item.get("content") or ""
        return context
    
    def _extract_question(self, item: Dict[str, Any]) -> str:
        """
        从数据条目中提取问题，子类可重写此方法
        
        Args:
            item: 数据条目
        
        Returns:
            提取的问题文本
        """
        # 默认实现尝试常见字段名
        question = item.get("question") or item.get("query") or ""
        return question
    
    def _extract_answer(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        从数据条目中提取答案，子类可重写此方法
        
        Args:
            item: 数据条目
        
        Returns:
            包含答案文本和位置的字典
        """
        # 默认实现尝试常见字段名
        answer = item.get("answer") or item.get("answers", [{}])[0]
        
        # 如果答案是字符串，则尝试在上下文中找到它
        if isinstance(answer, str):
            context = self._extract_context(item)
            answer_start = context.find(answer)
            return {"text": answer, "answer_start": answer_start}
        
        # 如果答案已经是字典形式，则直接返回
        elif isinstance(answer, dict):
            # 确保字典中包含text和answer_start字段
            if "text" not in answer:
                answer["text"] = ""
            if "answer_start" not in answer:
                # 尝试在上下文中找到答案位置
                context = self._extract_context(item)
                answer["answer_start"] = context.find(answer["text"])
            return answer
        
        # 其他情况返回空答案
        return {"text": "", "answer_start": -1}
    
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
        
        # 如果是训练模式，需要加入答案位置
        if "start_positions" in batch and "end_positions" in batch:
            inputs["start_positions"] = batch["start_positions"]
            inputs["end_positions"] = batch["end_positions"]
        
        return inputs
    
    def compute_loss(self, model_outputs: Any, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算MRC损失
        
        Args:
            model_outputs: 模型输出
            batch: 数据批次
        
        Returns:
            损失张量
        """
        start_logits = model_outputs.start_logits
        end_logits = model_outputs.end_logits
        
        # 如果没有提供开始和结束位置，则返回0损失（用于预测）
        if "start_positions" not in batch or "end_positions" not in batch:
            return torch.tensor(0.0, device=start_logits.device)
        
        start_positions = batch["start_positions"]
        end_positions = batch["end_positions"]
        
        # 有时批次大小可能为1，这会导致维度问题
        if len(start_positions.shape) == 0:
            start_positions = start_positions.unsqueeze(0)
        if len(end_positions.shape) == 0:
            end_positions = end_positions.unsqueeze(0)
        
        # 计算交叉熵损失
        loss_fct = torch.nn.CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        
        # 总损失为开始和结束位置损失的平均值
        total_loss = (start_loss + end_loss) / 2
        return total_loss
    
    def compute_metrics(self, predictions: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        计算MRC评估指标
        
        Args:
            predictions: 预测结果，包含start_logits和end_logits
            labels: 真实标签，包含start_positions和end_positions
        
        Returns:
            包含评估指标的字典
        """
        all_start_logits = predictions["start_logits"]
        all_end_logits = predictions["end_logits"]
        
        # 将所有数据移到CPU
        if isinstance(all_start_logits, torch.Tensor):
            all_start_logits = all_start_logits.detach().cpu().numpy()
        if isinstance(all_end_logits, torch.Tensor):
            all_end_logits = all_end_logits.detach().cpu().numpy()
        
        # 对于每个样本，找到最可能的开始和结束位置
        start_indices = np.argmax(all_start_logits, axis=1)
        end_indices = np.argmax(all_end_logits, axis=1)
        
        # 计算准确率 - 这里简化处理，实际MRC通常需要更复杂的指标
        if "start_positions" in labels and "end_positions" in labels:
            start_positions = labels["start_positions"]
            end_positions = labels["end_positions"]
            
            if isinstance(start_positions, torch.Tensor):
                start_positions = start_positions.detach().cpu().numpy()
            if isinstance(end_positions, torch.Tensor):
                end_positions = end_positions.detach().cpu().numpy()
            
            # 计算开始位置和结束位置的准确率
            start_accuracy = (start_indices == start_positions).mean()
            end_accuracy = (end_indices == end_positions).mean()
            
            # 完全匹配的准确率
            exact_match = ((start_indices == start_positions) & (end_indices == end_positions)).mean()
            
            return {
                "start_accuracy": float(start_accuracy),
                "end_accuracy": float(end_accuracy),
                "exact_match": float(exact_match)
            }
        else:
            # 如果没有标签（预测模式），则返回空指标
            return {}
    
    def postprocess_predictions(
        self, 
        examples: List[Dict[str, Any]], 
        features: List[Dict[str, Any]],
        predictions: Dict[str, np.ndarray]
    ) -> List[Dict[str, str]]:
        """
        后处理预测结果
        
        Args:
            examples: 原始样例列表
            features: 处理后的特征列表
            predictions: 模型预测结果
        
        Returns:
            包含预测答案的列表
        """
        all_start_logits = predictions["start_logits"]
        all_end_logits = predictions["end_logits"]
        
        # 构建example_id到其对应features的映射
        example_to_features = {}
        for feature in features:
            example_id = feature["example_id"]
            if example_id not in example_to_features:
                example_to_features[example_id] = []
            example_to_features[example_id].append(feature)
        
        # 对每个example，从所有候选答案中选择最佳答案
        final_predictions = []
        for example in examples:
            example_id = example["id"]
            example_features = example_to_features[example_id]
            context = example["context"]
            
            # 获取该example的所有预测
            valid_answers = []
            for feature_index, feature in enumerate(example_features):
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                offset_mapping = feature["offset_mapping"]
                
                # 获取前20个可能的答案
                start_indexes = np.argsort(start_logits)[-20:][::-1].tolist()
                end_indexes = np.argsort(end_logits)[-20:][::-1].tolist()
                
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # 跳过无效的答案位置
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                            or end_index < start_index
                        ):
                            continue
                        
                        # 从原始上下文中提取答案
                        answer_start = offset_mapping[start_index][0]
                        answer_end = offset_mapping[end_index][1]
                        answer_text = context[answer_start:answer_end]
                        
                        # 计算得分
                        score = start_logits[start_index] + end_logits[end_index]
                        
                        valid_answers.append({
                            "text": answer_text,
                            "score": score
                        })
            
            # 如果有有效答案，选择得分最高的
            if valid_answers:
                valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)
                final_predictions.append({
                    "id": example_id,
                    "answer": valid_answers[0]["text"]
                })
            else:
                # 如果没有有效答案，返回空答案
                final_predictions.append({
                    "id": example_id,
                    "answer": ""
                })
        
        return final_predictions