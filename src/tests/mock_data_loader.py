import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os
import json
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.prompts.template_manager import PromptManager

# 添加自定义collate函数
def collate_batch(batch):
    """
    为不同长度的序列创建批次
    
    参数:
        batch: 批次数据列表
        
    返回:
        批次字典，包含填充后的序列
    """
    # 分离batch中的不同字段
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    
    # 填充序列到相同长度
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    # 收集其他非张量字段
    task_types = [item['task_type'] for item in batch]
    input_texts = [item['input_text'] for item in batch]
    target_texts = [item['target_text'] for item in batch]
    
    return {
        'input_ids': input_ids_padded,
        'target_ids': target_ids_padded,
        'attention_mask': attention_masks_padded,
        'task_type': task_types,
        'input_text': input_texts,
        'target_text': target_texts
    }

class MultiTaskMockDataset(Dataset):
    """多任务模拟数据集"""
    def __init__(self, prompt_manager, task_type, size=100, max_len=40):
        self.size = size
        self.max_len = max_len
        self.prompt_manager = prompt_manager
        self.task_type = task_type
        
        # 为不同任务创建不同的样本特征
        self.task_features = self._create_task_features()
        
    def _create_task_features(self):
        """为不同任务创建特定的样本特征"""
        if self.task_type == "classify":
            # 文本分类任务特征
            categories = ["体育", "财经", "科技", "娱乐", "教育"]
            return {
                "categories": categories,
                "avg_len": 15,
                "examples": [
                    {"input": "股市大盘今日上涨2%，创近期新高。", "target": "财经"},
                    {"input": "新款手机处理器性能提升30%，续航大幅改善。", "target": "科技"}
                ]
            }
        elif self.task_type == "nli":
            # 自然语言推理任务特征
            relations = ["蕴含", "矛盾", "中立"]
            return {
                "relations": relations,
                "avg_len": 25,
                "examples": [
                    {"premise": "这家餐厅的食物很美味。", "hypothesis": "食物的味道很好。", "target": "蕴含"},
                    {"premise": "他是一名优秀的医生。", "hypothesis": "他不是医生。", "target": "矛盾"}
                ]
            }
        elif self.task_type == "mrc":
            # 阅读理解任务特征
            return {
                "avg_len": 35,
                "examples": [
                    {
                        "context": "人工智能是研究如何使计算机实现人类智能的一门科学。",
                        "question": "人工智能研究的核心是什么？", 
                        "target": "如何使计算机实现人类智能"
                    }
                ]
            }
        else:
            # 默认任务特征
            return {"avg_len": 20}
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 根据任务类型生成不同的样本
        if self.task_type == "classify":
            return self._generate_classification_sample(idx)
        elif self.task_type == "nli":
            return self._generate_nli_sample(idx)
        elif self.task_type == "mrc":
            return self._generate_mrc_sample(idx)
        else:
            return self._generate_default_sample(idx)
    
    def _generate_classification_sample(self, idx):
        # 随机选择类别
        categories = self.task_features["categories"]
        target = random.choice(categories)
        
        # 为分类任务创建模拟样本
        example = {
            "input_text": f"任务{idx}: 这是一段关于{target}的模拟文本。",
            "answer_choices": categories,
            "target": target,
            "type": "classify"
        }
        
        # 使用PromptManager处理样本
        processed = self.prompt_manager.process_example(example, "classify")
        
        # 生成固定长度的输入序列
        input_length = 20  # 为分类任务设置固定长度
        input_ids = torch.randint(1, 1000, (input_length,))
        
        # 生成固定长度的目标序列
        target_length = 5  # 分类标签通常较短
        target_ids = torch.randint(1, 1000, (target_length,))
        
        # 创建注意力掩码
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': attention_mask,
            'task_type': self.task_type,
            'input_text': processed['input_prompt'],
            'target_text': processed['target_text']
        }
    
    def _generate_nli_sample(self, idx):
        # 随机选择关系
        relations = self.task_features["relations"]
        target = random.choice(relations)
        
        # 为NLI任务创建模拟样本
        example = {
            "input_text": f"前提：这是任务{idx}的前提。假设：这是相应的假设。",
            "answer_choices": relations,
            "target": target,
            "type": "nli"
        }
        
        # 使用PromptManager处理样本
        processed = self.prompt_manager.process_example(example, "nli")
        
        # 生成固定长度的输入序列
        input_length = 25  # 为NLI任务设置固定长度
        input_ids = torch.randint(1, 1000, (input_length,))
        
        # 生成固定长度的目标序列
        target_length = 5  # NLI目标通常很短
        target_ids = torch.randint(1, 1000, (target_length,))
        
        # 创建注意力掩码
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': attention_mask,
            'task_type': self.task_type,
            'input_text': processed['input_prompt'],
            'target_text': processed['target_text']
        }
    
    def _generate_mrc_sample(self, idx):
        # 为MRC任务创建模拟样本
        example = {
            "context": f"这是任务{idx}的上下文段落，包含一些信息。",
            "question": "这是一个问题？",
            "target": "这是问题的答案",
            "type": "mrc"
        }
        
        # 使用PromptManager处理样本
        processed = self.prompt_manager.process_example(example, "mrc")
        
        # 生成固定长度的输入序列
        input_length = 30  # 为MRC任务设置固定长度
        input_ids = torch.randint(1, 1000, (input_length,))
        
        # 生成固定长度的目标序列
        target_length = 8  # MRC答案长度适中
        target_ids = torch.randint(1, 1000, (target_length,))
        
        # 创建注意力掩码
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': attention_mask,
            'task_type': self.task_type,
            'input_text': processed['input_prompt'],
            'target_text': processed['target_text']
        }
    
    def _generate_default_sample(self, idx):
        # 通用样本生成
        input_length = 20  # 为默认任务设置固定长度
        target_length = 10
        
        input_ids = torch.randint(1, 1000, (input_length,))
        target_ids = torch.randint(1, 1000, (target_length,))
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': attention_mask,
            'task_type': 'default',
            'input_text': f"默认任务样本 {idx}",
            'target_text': f"默认目标 {idx}"
        }
    
    def _pad_or_truncate_sequence(self, sequence, target_length, pad_value=0):
        """填充或截断序列到固定长度
        
        参数:
            sequence: 输入序列张量
            target_length: 目标长度
            pad_value: 填充值，默认为0
            
        返回:
            填充或截断后的序列张量
        """
        current_length = sequence.size(0)
        
        if current_length < target_length:
            # 需要填充
            padding = torch.full((target_length - current_length,), pad_value, dtype=sequence.dtype)
            padded_sequence = torch.cat([sequence, padding], dim=0)
            return padded_sequence
        elif current_length > target_length:
            # 需要截断
            return sequence[:target_length]
        else:
            # 长度正好
            return sequence

class MultiTaskMockDataLoader:
    """多任务模拟数据加载器"""
    def __init__(self, data_dir, prompt_manager):
        self.data_dir = data_dir
        self.prompt_manager = prompt_manager
        self.task_types = ["classify", "nli", "mrc"]
    
    def _collate_fn(self, batch):
        """自定义的批处理整理函数，处理不同长度的序列"""
        # 找出批次中最长的序列长度
        max_input_len = max([x['input_ids'].size(0) for x in batch])
        max_target_len = max([x['target_ids'].size(0) for x in batch])
        
        # 填充所有序列到相同长度
        for item in batch:
            item['input_ids'] = self._pad_sequence(item['input_ids'], max_input_len)
            item['attention_mask'] = self._pad_sequence(item['attention_mask'], max_input_len)
            item['target_ids'] = self._pad_sequence(item['target_ids'], max_target_len)
        
        # 将列表转换为批次张量
        input_ids = torch.stack([x['input_ids'] for x in batch])
        target_ids = torch.stack([x['target_ids'] for x in batch])
        attention_mask = torch.stack([x['attention_mask'] for x in batch])
        
        # 收集非张量数据
        task_types = [x['task_type'] for x in batch]
        input_texts = [x['input_text'] for x in batch]
        target_texts = [x['target_text'] for x in batch]
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': attention_mask,
            'task_type': task_types,
            'input_text': input_texts,
            'target_text': target_texts
        }
    
    def _pad_sequence(self, sequence, length, padding_value=0):
        """填充序列到指定长度"""
        padded = torch.ones(length) * padding_value
        padded[:sequence.size(0)] = sequence
        return padded.long()  # 确保返回LongTensor
        
    def get_task_dataloader(self, task_type, batch_size=8, is_train=True):
        """获取特定任务的数据加载器"""
        dataset = MultiTaskMockDataset(
            self.prompt_manager, 
            task_type, 
            size=80 if is_train else 20
        )
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=is_train,
            collate_fn=self._collate_fn  # 使用自定义的整理函数
        )
    
    def get_all_train_dataloaders(self, batch_size=8):
        """获取所有任务的训练数据加载器"""
        return {
            task: self.get_task_dataloader(task, batch_size, is_train=True)
            for task in self.task_types
        }
        
    def get_all_eval_dataloaders(self, batch_size=8):
        """获取所有任务的评估数据加载器"""
        return {
            task: self.get_task_dataloader(task, batch_size, is_train=False)
            for task in self.task_types
        }


# 测试代码
if __name__ == "__main__":
    # 初始化提示管理器
    config_path = os.path.join(project_root, "prompt_templates_config.yaml")
    prompt_manager = PromptManager(config_path)
    
    # 创建多任务数据加载器
    data_loader = MultiTaskMockDataLoader("./data", prompt_manager)
    
    # 测试所有任务的数据加载
    for task in data_loader.task_types:
        print(f"\n测试任务: {task}")
        loader = data_loader.get_task_dataloader(task)
        
        for batch in loader:
            print(f"  输入形状: {batch['input_ids'].shape}")
            print(f"  目标形状: {batch['target_ids'].shape}")
            print(f"  任务类型: {batch['task_type'][0]}")
            print(f"  输入文本示例: {batch['input_text'][0][:50]}...")
            print(f"  目标文本示例: {batch['target_text'][0][:30]}...")
            break
    
    print("\n多任务数据加载器测试完成!")