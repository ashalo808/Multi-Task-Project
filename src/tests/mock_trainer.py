import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from tqdm import tqdm
import random
import numpy as np
import time
from collections import defaultdict

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.models.core_seq2seq import Seq2SeqTransformer
from src.tests.mock_data_loader import MultiTaskMockDataLoader
from src.prompts.template_manager import PromptManager

class MultiTaskTrainer:
    """多任务训练器"""
    
    def __init__(self, model, data_loader, config, output_dir, device):
        """
        初始化多任务训练器
        
        参数:
            model: 模型实例（共享编码器）
            data_loader: 多任务数据加载器
            config: 训练配置
            output_dir: 输出目录
            device: 设备
        """
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.output_dir = output_dir
        self.device = device
        
        # 多任务训练配置
        self.task_types = data_loader.task_types
        
        # 设置损失函数和优化器
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 假设0是填充标记
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.get('learning_rate', 0.0001),
            weight_decay=config.get('weight_decay', 0.0001)
        )
        
        # 多任务训练统计
        self.task_losses = {task: [] for task in self.task_types}
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
    def train_single_task(self, task_type, dataloader, epoch, task_steps=20):
        """
        训练单个任务
        
        参数:
            task_type: 任务类型
            dataloader: 任务数据加载器
            epoch: 当前轮次
            task_steps: 每个任务的训练步数
        
        返回:
            平均损失值
        """
        self.model.train()
        task_loss = 0.0
        steps = 0
        
        # 使用进度条跟踪训练进度
        with tqdm(total=task_steps, desc=f"Epoch {epoch} - 任务: {task_type}", unit="batch") as pbar:
            for batch in dataloader:
                if steps >= task_steps:
                    break
                    
                # 移动数据到设备
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # 创建目标掩码 - 忽略填充标记
                target_mask = (target_ids != 0)
                
                # 前向传播 - 使用注意力掩码
                outputs = self.model(
                    input_ids, 
                    target_ids[:, :-1],
                    src_padding_mask=(input_ids == 0),  # 输入填充掩码
                    tgt_padding_mask=(target_ids[:, :-1] == 0)  # 目标填充掩码
                )
                
                # 计算损失 (reshape输出以适应损失函数)
                outputs = outputs.reshape(-1, outputs.size(-1))
                target_ids = target_ids[:, 1:].reshape(-1)
                
                # 仅计算非填充标记的损失
                loss = self.criterion(outputs, target_ids)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('gradient_clip_val', 1.0)
                )
                
                self.optimizer.step()
                
                # 更新统计数据
                task_loss += loss.item()
                steps += 1
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # 计算平均损失
        avg_loss = task_loss / steps if steps > 0 else 0
        self.task_losses[task_type].append(avg_loss)
        
        return avg_loss
    
    def train_multitask(self, num_epochs=3, task_sampling="round_robin"):
        """
        执行多任务训练
        
        参数:
            num_epochs: 训练轮数
            task_sampling: 任务采样策略
                - "round_robin": 轮流训练每个任务
                - "random": 随机选择任务
                - "proportional": 按任务数据比例采样
        """
        print(f"开始多任务训练 - 采样策略: {task_sampling}")
        
        # 获取所有任务的数据加载器
        task_dataloaders = self.data_loader.get_all_train_dataloaders(
            batch_size=self.config.get('batch_size', 8)
        )
        
        # 设置每个任务每轮的训练步数
        task_steps = 20  # 可以根据需要调整
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            epoch_losses = []
            
            if task_sampling == "round_robin":
                # 轮流训练每个任务
                for task in self.task_types:
                    avg_loss = self.train_single_task(
                        task, task_dataloaders[task], epoch+1, task_steps
                    )
                    epoch_losses.append(avg_loss)
                    print(f"Epoch {epoch+1} - 任务 {task} 平均损失: {avg_loss:.4f}")
            
            elif task_sampling == "random":
                # 随机选择训练任务的顺序
                random_tasks = self.task_types.copy()
                random.shuffle(random_tasks)
                for task in random_tasks:
                    avg_loss = self.train_single_task(
                        task, task_dataloaders[task], epoch+1, task_steps
                    )
                    epoch_losses.append(avg_loss)
                    print(f"Epoch {epoch+1} - 任务 {task} 平均损失: {avg_loss:.4f}")
            
            elif task_sampling == "proportional":
                # 按比例采样训练任务
                # 这里简化实现，实际应用中可能需要更复杂的采样策略
                for _ in range(len(self.task_types)):
                    task = random.choice(self.task_types)
                    avg_loss = self.train_single_task(
                        task, task_dataloaders[task], epoch+1, task_steps//len(self.task_types)
                    )
                    epoch_losses.append(avg_loss)
                    print(f"Epoch {epoch+1} - 任务 {task} 平均损失: {avg_loss:.4f}")
            
            # 计算并打印整体平均损失
            epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1} 完成 - 总平均损失: {epoch_avg_loss:.4f}, 用时: {epoch_time:.2f}秒")
            
            # 可以添加验证步骤和早停逻辑
        
        print("多任务训练完成!")
        return self.task_losses
    
    def evaluate_multitask(self):
        """
        对所有任务进行评估
        
        返回:
            包含每个任务评估指标的字典
        """
        print("开始多任务评估...")
        self.model.eval()
        
        # 获取所有任务的评估数据加载器
        task_dataloaders = self.data_loader.get_all_eval_dataloaders(
            batch_size=self.config.get('batch_size', 8)
        )
        
        task_metrics = {}
        
        with torch.no_grad():
            for task in self.task_types:
                print(f"评估任务: {task}")
                task_loss = 0.0
                steps = 0
                
                for batch in task_dataloaders[task]:
                    # 移动数据到设备
                    input_ids = batch['input_ids'].to(self.device)
                    target_ids = batch['target_ids'].to(self.device)
                    
                    # 前向传播
                    outputs = self.model(input_ids, target_ids[:, :-1])
                    
                    # 计算损失
                    outputs = outputs.reshape(-1, outputs.size(-1))
                    target_ids = target_ids[:, 1:].reshape(-1)
                    
                    loss = self.criterion(outputs, target_ids)
                    task_loss += loss.item()
                    steps += 1
                
                # 计算平均损失和任务特定指标
                avg_loss = task_loss / steps if steps > 0 else 0
                
                # 在实际应用中，这里可以计算任务特定的评估指标，如准确率、F1等
                task_metrics[task] = {
                    "loss": avg_loss,
                    # 可以添加更多特定于任务的指标
                }
                
                print(f"  任务 {task} 评估损失: {avg_loss:.4f}")
        
        return task_metrics
    
    def plot_training_progress(self):
        """绘制训练进度图表"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            for task in self.task_types:
                plt.plot(self.task_losses[task], label=f"任务: {task}")
            
            plt.xlabel("训练步骤")
            plt.ylabel("损失")
            plt.title("多任务训练损失")
            plt.legend()
            plt.grid(True)
            
            plot_path = os.path.join(self.output_dir, "multitask_training_loss.png")
            plt.savefig(plot_path)
            print(f"训练损失图表保存到: {plot_path}")
        except ImportError:
            print("需要安装matplotlib才能绘制图表")
            pass


# 测试代码
def test_multitask_training():
    """测试多任务训练流程"""
    # 设置随机种子以便结果可复现
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # 初始化提示管理器
    config_path = os.path.join(project_root, "prompt_templates_config.yaml")
    prompt_manager = PromptManager(config_path)
    
    # 创建多任务数据加载器
    data_loader = MultiTaskMockDataLoader("./data", prompt_manager)
    
    # 创建模型
    model = Seq2SeqTransformer(
        vocab_size=1000,
        d_model=64,
        nhead=4,
        num_encoder_layers=3,  # 共享的编码器层
        num_decoder_layers=3,
        dim_feedforward=128,
        dropout=0.1,
        max_seq_length=50
    )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 创建训练配置
    config = {
        'batch_size': 4,
        'learning_rate': 0.0005,
        'weight_decay': 0.0001,
        'gradient_clip_val': 1.0
    }
    
    # 创建输出目录
    output_dir = os.path.join(project_root, "test_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化多任务训练器
    trainer = MultiTaskTrainer(model, data_loader, config, output_dir, device)
    
    # 开始多任务训练
    print("\n开始多任务训练...")
    trainer.train_multitask(num_epochs=2, task_sampling="round_robin")
    
    # 评估多任务性能
    print("\n开始多任务评估...")
    task_metrics = trainer.evaluate_multitask()
    
    # 绘制训练进度
    trainer.plot_training_progress()
    
    print("\n多任务训练测试完成!")


if __name__ == "__main__":
    test_multitask_training()