import sys
import os
import torch
import random
import numpy as np
import argparse
from collections import defaultdict

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.models.core_seq2seq import Seq2SeqTransformer, save_model
from src.prompts.template_manager import PromptManager
from src.tests.mock_data_loader import MultiTaskMockDataLoader
from src.tests.mock_trainer import MultiTaskTrainer

class MultiTaskTester:
    """多任务集成测试类"""
    
    def __init__(self, model, data_loader, prompt_manager, device):
        self.model = model
        self.data_loader = data_loader
        self.prompt_manager = prompt_manager
        self.device = device
        self.task_types = data_loader.task_types
    
    def test_shared_encoder(self):
        """测试共享编码器在多任务间的表现"""
        print("\n测试共享编码器在不同任务间的一致性...")
        
        # 获取每个任务的一个样本
        task_samples = {}
        for task in self.task_types:
            dataloader = self.data_loader.get_task_dataloader(task, batch_size=1)
            for batch in dataloader:
                task_samples[task] = batch
                break
        
        # 使用相同的编码器处理不同任务的输入
        encodings = {}
        self.model.eval()
        with torch.no_grad():
            for task, batch in task_samples.items():
                input_ids = batch['input_ids'].to(self.device)
                memory = self.model.encode(input_ids)
                encodings[task] = memory
                print(f"任务 {task} 编码输出形状: {memory.shape}")
        
        # 检查编码器共享特征
        print("\n验证编码器共享特性:")
        hidden_dims = [enc.shape[-1] for enc in encodings.values()]
        all_same_dim = all(dim == hidden_dims[0] for dim in hidden_dims)
        print(f"所有任务共享相同隐藏维度: {'是' if all_same_dim else '否'}")
        
        # 计算每个任务编码输出的统计特征
        for task, encoding in encodings.items():
            mean = torch.mean(encoding).item()
            std = torch.std(encoding).item()
            print(f"任务 {task} 编码统计 - 均值: {mean:.4f}, 标准差: {std:.4f}")
        
        return all_same_dim
    
    def test_cross_task_transfer(self):
        """测试跨任务知识迁移能力"""
        print("\n测试跨任务知识迁移能力...")
        
        # 获取每个任务的一个样本
        task_samples = {}
        for task in self.task_types:
            dataloader = self.data_loader.get_task_dataloader(task, batch_size=1)
            for batch in dataloader:
                task_samples[task] = batch
                break
        
        # 跨任务使用编码器和解码器
        self.model.eval()
        with torch.no_grad():
            # 对于每个任务的输入
            for input_task, input_batch in task_samples.items():
                input_ids = input_batch['input_ids'].to(self.device)
                
                # 编码输入
                memory = self.model.encode(input_ids)
                
                print(f"\n使用任务 {input_task} 的输入:")
                
                # 用其他任务的目标解码
                for target_task, target_batch in task_samples.items():
                    target_ids = target_batch['target_ids'].to(self.device)
                    
                    # 解码
                    output = self.model.decode(target_ids[:, :-1], memory)
                    
                    print(f"  与任务 {target_task} 的目标解码 - 输出形状: {output.shape}")
                    
                    # 在实际应用中，这里可以评估解码质量
        
        print("跨任务知识迁移测试完成")
    
    def test_multitask_inference(self):
        """测试多任务推理能力"""
        print("\n测试多任务推理能力...")
        
        # 获取每个任务的一个样本
        task_samples = {}
        for task in self.task_types:
            dataloader = self.data_loader.get_task_dataloader(task, batch_size=1)
            for batch in dataloader:
                task_samples[task] = batch
                break
        
        # 对每个任务进行推理
        self.model.eval()
        with torch.no_grad():
            for task, batch in task_samples.items():
                print(f"\n对任务 {task} 执行推理:")
                
                input_ids = batch['input_ids'].to(self.device)
                
                # 使用生成函数
                generated = self.model.generate(
                    input_ids,
                    max_len=15,
                    start_token_id=1,
                    end_token_id=2
                )
                
                print(f"  生成序列形状: {generated.shape}")
                print(f"  输入长度: {input_ids.shape[1]}")
                print(f"  生成长度: {generated.shape[1]}")


def run_integrated_test():
    """执行综合集成测试"""
    # 设置随机种子以便结果可复现
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    print("开始多任务共享编码器集成测试...")
    
    # 确保输出目录存在
    output_dir = os.path.join(project_root, "test_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化提示管理器
    config_path = os.path.join(project_root, "src", "config", "prompt_templates_config.yaml")
    prompt_manager = PromptManager(config_path)
    print(f"提示管理器初始化完成，使用配置路径: {config_path}")
    
    # 创建多任务数据加载器
    data_loader = MultiTaskMockDataLoader("./data", prompt_manager)
    print("多任务数据加载器初始化完成")
    
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
    model.to(device)
    print("共享编码器模型初始化完成")
    
    # 创建训练配置
    train_config = {
        'batch_size': 4,
        'learning_rate': 0.0005,
        'weight_decay': 0.0001,
        'gradient_clip_val': 1.0
    }
    
    # 初始化多任务训练器
    trainer = MultiTaskTrainer(model, data_loader, train_config, output_dir, device)
    
    # 第1阶段：多任务训练
    print("\n第1阶段：开始多任务训练...")
    trainer.train_multitask(num_epochs=2)
    
    # 保存模型
    save_path = os.path.join(output_dir, "multitask_model.pth")
    save_model(model, save_path)
    print(f"多任务模型已保存到: {save_path}")
    
    # 第2阶段：评估多任务性能
    print("\n第2阶段：评估多任务性能...")
    task_metrics = trainer.evaluate_multitask()
    
    # 第3阶段：执行特定的多任务测试
    print("\n第3阶段：执行特定的多任务测试...")
    tester = MultiTaskTester(model, data_loader, prompt_manager, device)
    
    # 测试共享编码器
    shared_encoder_works = tester.test_shared_encoder()
    
    # 测试跨任务知识迁移
    tester.test_cross_task_transfer()
    
    # 测试多任务推理
    tester.test_multitask_inference()
    
    # 绘制训练进度
    trainer.plot_training_progress()
    
    # 总结结果
    print("\n============ 多任务集成测试总结 ============")
    print(f"共享编码器表现: {'成功' if shared_encoder_works else '需要改进'}")
    print("任务评估指标:")
    for task, metrics in task_metrics.items():
        print(f"  {task}: 损失 = {metrics['loss']:.4f}")
    print("==========================================")
    
    print("\n多任务共享编码器集成测试完成!")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多任务共享编码器测试')
    
    parser.add_argument('--output_dir', type=str, default='./test_outputs',
                        help='输出目录')
    parser.add_argument('--device', type=str, default=None,
                        help='运行设备 (例如 "cuda:0" 或 "cpu")')
    parser.add_argument('--epochs', type=int, default=2,
                        help='训练轮数')
    
    return parser.parse_args()


if __name__ == "__main__":
    run_integrated_test()