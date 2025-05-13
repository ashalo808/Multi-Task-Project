import sys
import os
import torch

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.models.core_seq2seq import Seq2SeqTransformer

class MultiTaskTester:
    """多任务编码器测试类"""
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        
    def test_encoder_sharing(self):
        """测试编码器在多个任务间的共享特性"""
        print("测试编码器的跨任务共享能力...")
        
        # 创建模拟不同任务的测试数据
        tasks = {
            "classification": torch.randint(1, 1000, (4, 15)),
            "nli": torch.randint(1, 1000, (4, 25)),
            "mrc": torch.randint(1, 1000, (4, 35))
        }
        
        # 对每个任务使用相同的编码器进行编码
        task_embeddings = {}
        for task_name, input_ids in tasks.items():
            print(f"处理任务: {task_name}，输入形状: {input_ids.shape}")
            input_ids = input_ids.to(self.device)
            
            # 使用编码器编码任务输入
            with torch.no_grad():
                memory = self.model.encode(input_ids)
            
            task_embeddings[task_name] = memory
            print(f"  编码器输出形状: {memory.shape}")
            
        # 验证编码器输出特征表示的通用性
        self._verify_representation_sharing(task_embeddings)
        
        # 测试基于共享编码器的多任务解码
        self._test_multitask_decoding(tasks, task_embeddings)
        
    def _verify_representation_sharing(self, task_embeddings):
        """验证不同任务的编码表示共享相同的维度空间"""
        print("\n验证编码表示的共享性...")
        
        # 检查所有任务的隐藏维度是否一致
        hidden_dims = [emb.shape[-1] for emb in task_embeddings.values()]
        is_consistent = len(set(hidden_dims)) == 1
        print(f"  隐藏维度一致性: {'通过' if is_consistent else '失败'}")
        
        # 检查编码表示的统计特性
        for task_name, embedding in task_embeddings.items():
            mean = torch.mean(embedding).item()
            std = torch.std(embedding).item()
            print(f"  {task_name} - 均值: {mean:.4f}, 标准差: {std:.4f}")
    
    def _test_multitask_decoding(self, tasks, task_embeddings):
        """测试使用共享编码器的表示进行多任务解码"""
        print("\n测试基于共享编码器的多任务解码...")
        
        # 为每个任务创建相同形状的解码器输入
        decoder_input = torch.randint(1, 1000, (4, 10)).to(self.device)
        
        for task_name, memory in task_embeddings.items():
            print(f"测试任务: {task_name}")
            
            # 使用相同的解码器和任务特定的编码器记忆
            with torch.no_grad():
                output = self.model.decode(decoder_input, memory)
            
            print(f"  解码器输出形状: {output.shape}")
            
            # 测试完整的序列生成
            with torch.no_grad():
                generated = self.model.generate(
                    tasks[task_name].to(self.device),
                    max_len=12,
                    start_token_id=1,
                    end_token_id=2
                )
            
            print(f"  生成序列形状: {generated.shape}")


def main():
    # 创建一个适合多任务的模型
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
    
    # 实例化测试器并运行测试
    tester = MultiTaskTester(model)
    tester.test_encoder_sharing()
    
    print("\n共享编码器测试完成!")

if __name__ == "__main__":
    main()