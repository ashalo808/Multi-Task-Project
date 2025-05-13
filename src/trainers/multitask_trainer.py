import os
import time
import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.tasks.task_manager import TaskManager


class MultitaskTrainer:
    """多任务训练器，负责协调多个任务的训练、评估和推理"""
    
    def __init__(
        self, 
        model, 
        task_manager: TaskManager,
        output_dir: str,
        device: str = None,
        log_steps: int = 10,
        save_steps: int = 1000,
        eval_steps: int = 500
    ):
        """
        初始化多任务训练器
        
        Args:
            model: 共享的基础模型
            task_manager: 任务管理器实例
            output_dir: 输出目录
            device: 运行设备
            log_steps: 日志记录间隔步数
            save_steps: 模型保存间隔步数
            eval_steps: 评估间隔步数
        """
        self.model = model
        self.task_manager = task_manager
        self.output_dir = output_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_steps = log_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        
        # 将模型移到指定设备
        self.model.to(self.device)
        
        # 设置日志
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
    
    def train(
        self,
        num_epochs: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        batch_size: int = 32,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        early_stopping_patience: int = 0,
        task_sampling_weights: Optional[Dict[str, float]] = None
    ):
        """
        训练多任务模型
        
        Args:
            num_epochs: 训练轮数
            optimizer: 优化器
            scheduler: 学习率调度器
            batch_size: 批次大小
            gradient_accumulation_steps: 梯度累积步数
            max_grad_norm: 梯度裁剪最大范数
            early_stopping_patience: 早停耐心值，0表示不使用早停
            task_sampling_weights: 任务采样权重
        """
        self.logger.info("开始多任务训练...")
        
        # 初始化数据加载器
        task_loaders = self._initialize_task_loaders("train", batch_size)
        
        # 确保有任务
        if not task_loaders:
            self.logger.error("没有可用的训练任务")
            return
        
        # 设置模型为训练模式
        self.model.train()
        
        # 跟踪最佳模型性能
        best_avg_metric = -float('inf')
        no_improvement_count = 0
        
        # 构建任务采样权重
        tasks = list(task_loaders.keys())
        if task_sampling_weights:
            weights = [task_sampling_weights.get(task, 1.0) for task in tasks]
        else:
            weights = [1.0] * len(tasks)
        
        # 归一化权重
        weights = np.array(weights) / sum(weights)
        
        # 统计总步数和任务步数
        global_step = 0
        task_steps = {task: 0 for task in tasks}
        
        self.logger.info(f"训练任务: {', '.join(tasks)}")
        
        # 记录开始时间
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.logger.info(f"开始 Epoch {epoch+1}/{num_epochs}")
            
            epoch_loss = 0
            epoch_steps = 0
            
            # 创建进度条
            progress_bar = tqdm(total=sum(len(loader) for loader in task_loaders.values()),
                               desc=f"Epoch {epoch+1}")
            
            # 初始化任务迭代器字典
            task_iterators = {task: iter(loader) for task, loader in task_loaders.items()}
            active_tasks = set(tasks)
            
            # 循环直到所有任务的数据都处理完毕
            while active_tasks:
                # 根据权重采样任务
                task = np.random.choice(tasks, p=weights)
                
                # 如果该任务已处理完，跳过
                if task not in active_tasks:
                    continue
                
                # 尝试获取下一个批次
                try:
                    batch = next(task_iterators[task])
                except StopIteration:
                    active_tasks.remove(task)
                    continue
                
                # 处理当前任务的批次
                loss = self._train_step(task, batch, optimizer, scheduler, 
                                        gradient_accumulation_steps, max_grad_norm)
                
                epoch_loss += loss
                epoch_steps += 1
                global_step += 1
                task_steps[task] += 1
                
                progress_bar.update(1)
                
                # 更新进度条信息
                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'task': task,
                    'lr': f"{optimizer.param_groups[0]['lr']:.3e}"
                })
                
                # 记录日志
                if global_step % self.log_steps == 0:
                    self.logger.info(
                        f"Step: {global_step}, Task: {task}, Loss: {loss:.4f}, "
                        f"LR: {optimizer.param_groups[0]['lr']:.3e}, "
                        f"Time: {(time.time() - start_time) / 60:.2f}min"
                    )
                
                # 定期评估
                if global_step % self.eval_steps == 0:
                    avg_metric = self._evaluate_all_tasks(batch_size)
                    
                    # 检查是否有改进
                    if avg_metric > best_avg_metric:
                        best_avg_metric = avg_metric
                        no_improvement_count = 0
                        # 保存最佳模型
                        self._save_model("best_model")
                        self.logger.info(f"保存最佳模型，平均指标: {avg_metric:.4f}")
                    else:
                        no_improvement_count += 1
                    
                    # 早停检查
                    if early_stopping_patience > 0 and no_improvement_count >= early_stopping_patience:
                        self.logger.info(f"触发早停，{early_stopping_patience}次评估没有改进")
                        break
                    
                    # 恢复训练模式
                    self.model.train()
                
                # 定期保存模型
                if global_step % self.save_steps == 0:
                    self._save_model(f"checkpoint-{global_step}")
            
            progress_bar.close()
            
            # 每个epoch结束后的平均损失
            avg_epoch_loss = epoch_loss / max(1, epoch_steps)
            self.logger.info(f"Epoch {epoch+1} 完成，平均损失: {avg_epoch_loss:.4f}")
            
            # 每个epoch结束后评估
            self._evaluate_all_tasks(batch_size)
        
        # 训练结束后保存最终模型
        self._save_model("final_model")
        self.logger.info(f"训练完成，总步数: {global_step}，耗时: {(time.time() - start_time) / 60:.2f}分钟")
        
        # 返回最佳平均指标
        return best_avg_metric
    
    def _train_step(
        self, 
        task_name: str, 
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        gradient_accumulation_steps: int,
        max_grad_norm: float
    ) -> float:
        """
        执行单个任务的训练步骤
        
        Args:
            task_name: 任务名称
            batch: 数据批次
            optimizer: 优化器
            scheduler: 学习率调度器
            gradient_accumulation_steps: 梯度累积步数
            max_grad_norm: 梯度裁剪最大范数
        
        Returns:
            当前步骤的损失值
        """
        task = self.task_manager.get_task(task_name)
        
        # 将数据移到设备上
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # 获取任务特定的模型输入
        inputs = task.prepare_inputs(batch)
        
        # 前向传播
        outputs = self.model(**inputs)
        
        # 计算损失
        loss = task.compute_loss(outputs, batch)
        
        # 如果使用梯度累积，则缩放损失
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        
        # 反向传播
        loss.backward()
        
        # 检查是否需要优化器步骤
        if (task_steps[task_name] + 1) % gradient_accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            # 参数更新
            optimizer.step()
            
            # 学习率调度
            if scheduler is not None:
                scheduler.step()
            
            # 梯度清零
            optimizer.zero_grad()
        
        return loss.item()
    
    def evaluate(self, task_name: str, batch_size: int = 32, split: str = "dev") -> Dict[str, float]:
        """
        评估单个任务
        
        Args:
            task_name: 任务名称
            batch_size: 批次大小
            split: 数据集划分
        
        Returns:
            包含评估指标的字典
        """
        task = self.task_manager.get_task(task_name)
        dataloader = task.get_dataloader(split, batch_size, shuffle=False)
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        total_eval_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"评估 {task_name}"):
                # 将数据移到设备上
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 获取任务特定的模型输入
                inputs = task.prepare_inputs(batch)
                
                # 前向传播
                outputs = self.model(**inputs)
                
                # 计算损失
                loss = task.compute_loss(outputs, batch)
                total_eval_loss += loss.item()
                
                # 收集预测和标签
                if hasattr(outputs, "logits"):
                    predictions = outputs.logits
                else:
                    predictions = outputs
                
                all_predictions.append(predictions.detach().cpu())
                all_labels.append(batch["labels"].detach().cpu())
        
        # 合并所有批次的预测和标签
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # 计算评估指标
        metrics = task.compute_metrics(all_predictions, all_labels)
        
        # 添加平均损失到指标中
        metrics["eval_loss"] = total_eval_loss / len(dataloader)
        
        # 记录评估结果
        self.logger.info(f"任务 {task_name} ({split}) 评估指标:")
        for k, v in metrics.items():
            self.logger.info(f"  {k}: {v:.4f}")
        
        return metrics
    
    def _evaluate_all_tasks(self, batch_size: int = 32) -> float:
        """
        评估所有任务
        
        Args:
            batch_size: 批次大小
        
        Returns:
            所有任务的平均主要指标值
        """
        self.logger.info("评估所有任务...")
        
        self.model.eval()
        
        all_metrics = {}
        primary_metrics = []
        
        for task_name in self.task_manager.get_all_task_names():
            metrics = self.evaluate(task_name, batch_size)
            all_metrics[task_name] = metrics
            
            # 收集主要指标 (默认使用准确率或F1)
            primary_metric = metrics.get("accuracy", metrics.get("f1", 0.0))
            primary_metrics.append(primary_metric)
        
        # 计算平均主要指标
        avg_metric = sum(primary_metrics) / max(1, len(primary_metrics))
        self.logger.info(f"所有任务平均指标: {avg_metric:.4f}")
        
        return avg_metric
    
    def predict(self, task_name: str, data: List[Dict[str, Any]], batch_size: int = 32) -> List[Any]:
        """
        使用模型进行预测
        
        Args:
            task_name: 任务名称
            data: 要预测的数据列表
            batch_size: 批次大小
        
        Returns:
            预测结果列表
        """
        task = self.task_manager.get_task(task_name)
        
        self.model.eval()
        
        # 待实现: 将原始数据转换为可用于预测的格式
        # 这部分通常需要根据具体任务定制
        
        return []
    
    def _initialize_task_loaders(self, split: str, batch_size: int) -> Dict[str, DataLoader]:
        """
        初始化所有任务的数据加载器
        
        Args:
            split: 数据集划分
            batch_size: 批次大小
        
        Returns:
            任务名称到数据加载器的映射字典
        """
        task_loaders = {}
        for task_name in self.task_manager.get_all_task_names():
            try:
                task = self.task_manager.get_task(task_name)
                loader = task.get_dataloader(split, batch_size)
                task_loaders[task_name] = loader
            except Exception as e:
                self.logger.warning(f"无法为任务 {task_name} 初始化数据加载器: {e}")
        
        return task_loaders
    
    def _save_model(self, name: str):
        """
        保存模型
        
        Args:
            name: 模型名称
        """
        save_path = os.path.join(self.output_dir, name)
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(save_path)
        
        # 记录保存信息
        self.logger.info(f"模型保存到 {save_path}")
    
    def load_model(self, model_path: str):
        """
        加载模型
        
        Args:
            model_path: 模型路径
        """
        self.model.from_pretrained(model_path)
        self.model.to(self.device)
        self.logger.info(f"已从 {model_path} 加载模型")