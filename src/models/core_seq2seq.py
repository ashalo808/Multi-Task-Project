import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import yaml
import logging
from typing import Dict, List, Optional, Tuple, Union

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Seq2SeqTransformer(nn.Module):
    """
    基于Transformer架构的序列到序列模型
    """
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 512, 
                 nhead: int = 8, 
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, 
                 dim_feedforward: int = 2048, 
                 dropout: float = 0.1,
                 max_seq_length: int = 1024,
                 pad_idx: int = 0):
        super(Seq2SeqTransformer, self).__init__()
        
        self.model_type = "Transformer"
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.max_seq_length = max_seq_length
        
        # 创建词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # 创建Transformer编码器
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # 创建Transformer解码器
        decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, 
            src: torch.Tensor, 
            tgt: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None, 
            tgt_mask: Optional[torch.Tensor] = None,
            src_padding_mask: Optional[torch.Tensor] = None,
            tgt_padding_mask: Optional[torch.Tensor] = None,
            memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        模型前向传播
        
        参数:
            src: 源序列 [batch_size, src_seq_len]
            tgt: 目标序列 [batch_size, tgt_seq_len]
            src_mask: 源序列掩码，用于防止注意力看到某些位置
            tgt_mask: 目标序列掩码，通常用于防止注意力机制看到未来位置
            src_padding_mask: 源序列的填充掩码，True表示填充位置
            tgt_padding_mask: 目标序列的填充掩码，True表示填充位置
            memory_key_padding_mask: 记忆键的填充掩码
            
        返回:
            output: 模型输出 [batch_size, tgt_seq_len, vocab_size]
        """
        # 创建序列掩码，如果未提供
        if src_mask is None:
            src_mask = self._generate_square_subsequent_mask(src.size(1)).to(src.device)
            
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            
        # 创建填充掩码，如果未提供 (True表示需要掩码的位置)
        if src_padding_mask is None:
            src_padding_mask = (src == self.pad_idx).to(src.device)
            
        if tgt_padding_mask is None:
            tgt_padding_mask = (tgt == self.pad_idx).to(tgt.device)
        
        # 如果memory_key_padding_mask未提供，则使用src_padding_mask
        if memory_key_padding_mask is None:
            memory_key_padding_mask = src_padding_mask
        
        # 嵌入并添加位置编码
        src_embedded = self.positional_encoding(self.embedding(src))
        tgt_embedded = self.positional_encoding(self.embedding(tgt))
        
        # 编码器前向传播
        # 注意：src_mask通常是不需要的（没有自回归限制），但src_padding_mask是必要的
        memory = self.transformer_encoder(
            src_embedded, 
            mask=src_mask if src_mask is not None else None, 
            src_key_padding_mask=src_padding_mask
        )
        
        # 解码器前向传播
        # tgt_mask用于自回归，memory_key_padding_mask用于填充位置
        output = self.transformer_decoder(
            tgt_embedded, 
            memory, 
            tgt_mask=tgt_mask,
            memory_mask=None,  # 通常不需要memory_mask
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # 线性层输出
        output = self.output_layer(output)
        
        return output
    
    def encode(self, 
           src: torch.Tensor, 
           src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        仅执行编码器部分的前向传播
        
        参数:
            src: 源序列 [batch_size, src_seq_len]
            src_mask: 源序列掩码
            
        返回:
            memory: 编码器输出 [batch_size, src_seq_len, d_model]
        """
        # 创建填充掩码
        src_padding_mask = (src == self.pad_idx).to(src.device)
        
        # 嵌入并添加位置编码
        src_embedded = self.positional_encoding(self.embedding(src))
        
        # 编码器前向传播
        memory = self.transformer_encoder(
            src_embedded, 
            mask=src_mask, 
            src_key_padding_mask=src_padding_mask
        )
        
        return memory
    
    def decode(self, 
           tgt: torch.Tensor, 
           memory: torch.Tensor, 
           tgt_mask: Optional[torch.Tensor] = None,
           memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        仅执行解码器部分的前向传播
        
        参数:
            tgt: 目标序列 [batch_size, tgt_seq_len]
            memory: 来自编码器的memory [batch_size, src_seq_len, d_model]
            tgt_mask: 目标序列掩码
            memory_key_padding_mask: 记忆键的填充掩码
            
        返回:
            output: 解码器输出 [batch_size, tgt_seq_len, vocab_size]
        """
        # 创建目标掩码，如果未提供
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            
        # 创建填充掩码
        tgt_padding_mask = (tgt == self.pad_idx).to(tgt.device)
        
        # 嵌入并添加位置编码
        tgt_embedded = self.positional_encoding(self.embedding(tgt))
        
        # 解码器前向传播
        output = self.transformer_decoder(
            tgt_embedded, 
            memory, 
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # 线性层输出
        output = self.output_layer(output)
        
        return output
    
    def generate(self, 
            src: torch.Tensor, 
            max_len: int, 
            start_token_id: int, 
            end_token_id: int,
            beam_size: int = 1,
            length_penalty: float = 1.0) -> torch.Tensor:
        """
        生成序列
        
        参数:
            src: 源序列 [batch_size, src_seq_len]
            max_len: 生成序列的最大长度
            start_token_id: 起始标记ID
            end_token_id: 结束标记ID
            beam_size: 波束搜索大小，默认为1（贪婪搜索）
            length_penalty: 长度惩罚因子
            
        返回:
            生成的序列 [batch_size, seq_len]
        """
        device = src.device
        batch_size = src.size(0)
        
        # 创建源填充掩码
        src_padding_mask = (src == self.pad_idx).to(device)
        
        # 使用编码器处理输入
        memory = self.encode(src)
        
        # 存储记忆键填充掩码，供解码器使用
        memory_key_padding_mask = src_padding_mask
        
        if beam_size == 1:
            # 贪婪搜索
            decoder_input = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
            
            for _ in range(max_len - 1):
                # 解码预测下一个标记
                # 确保传递memory_key_padding_mask
                logits = self.decode(
                    decoder_input, 
                    memory, 
                    memory_key_padding_mask=memory_key_padding_mask
                )[:, -1]
                
                next_token = logits.argmax(dim=-1, keepdim=True)
                decoder_input = torch.cat([decoder_input, next_token], dim=-1)
                
                # 如果所有序列都生成了结束标记，则提前终止
                if (next_token == end_token_id).all():
                    break
            
            return decoder_input
        else:
            # 波束搜索 (将传递memory_key_padding_mask)
            return self._beam_search(
                memory, 
                max_len, 
                start_token_id, 
                end_token_id, 
                beam_size, 
                length_penalty,
                memory_key_padding_mask
            )
    
    def _beam_search(self, 
                memory: torch.Tensor, 
                max_len: int, 
                start_token_id: int, 
                end_token_id: int,
                beam_size: int = 5,
                length_penalty: float = 1.0,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        进行波束搜索生成序列
        
        参数:
            memory: 编码器输出 [batch_size, src_seq_len, d_model]
            max_len: 生成序列的最大长度
            start_token_id: 起始标记ID
            end_token_id: 结束标记ID
            beam_size: 波束搜索大小
            length_penalty: 长度惩罚因子
            memory_key_padding_mask: 记忆键的填充掩码
            
        返回:
            生成的序列 [batch_size, seq_len]
        """
        device = memory.device
        batch_size = memory.size(0)
        
        # 初始化
        beam_scores = torch.zeros(batch_size, beam_size, device=device)
        beam_tokens = torch.full((batch_size, beam_size, 1), start_token_id, dtype=torch.long, device=device)
        beam_indices = torch.arange(batch_size, device=device).view(-1, 1).repeat(1, beam_size)
        
        # 标记序列是否已完成
        done = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=device)
        
        # 扩展memory以匹配beam_size
        memory = memory.unsqueeze(1).expand(-1, beam_size, -1, -1)
        memory = memory.reshape(batch_size * beam_size, memory.size(2), memory.size(3))
        
        # 如果有填充掩码，也需要扩展它
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.unsqueeze(1).expand(-1, beam_size, -1)
            memory_key_padding_mask = memory_key_padding_mask.reshape(batch_size * beam_size, -1)
        
        for step in range(max_len - 1):
            # 准备当前输入
            decoder_input = beam_tokens.view(batch_size * beam_size, -1)
            
            # 解码预测下一个标记，传递填充掩码
            logits = self.decode(
                decoder_input, 
                memory,
                memory_key_padding_mask=memory_key_padding_mask
            )[:, -1]
            
            vocab_probs = F.log_softmax(logits, dim=-1)
            
            # 调整形状
            vocab_probs = vocab_probs.view(batch_size, beam_size, -1)
            
            # 计算新分数
            next_scores = beam_scores.unsqueeze(-1) + vocab_probs
            
            # 为已完成序列设置低分数，以便选择其他序列
            next_scores = next_scores.view(batch_size, -1)
            
            # 选择top-k
            next_scores, next_tokens = next_scores.topk(beam_size, dim=1)
            
            # 计算beam索引和token索引
            next_beam_indices = next_tokens // vocab_probs.size(-1)
            next_token_indices = next_tokens % vocab_probs.size(-1)
            
            # 更新tokens和scores
            beam_indices = torch.gather(beam_indices, 1, next_beam_indices)
            beam_tokens = torch.gather(beam_tokens, 1, 
                                    next_beam_indices.unsqueeze(-1).expand(-1, -1, step+1))
            beam_tokens = torch.cat([beam_tokens, next_token_indices.unsqueeze(-1)], dim=-1)
            beam_scores = next_scores
            
            # 更新完成状态
            done = done | (next_token_indices == end_token_id)
            
            # 如果所有序列都完成，则提前结束
            if done.all():
                break
        
        # 找到每个批次中得分最高的序列
        best_indices = beam_scores.argmax(dim=1)
        best_tokens = torch.gather(beam_tokens, 1, 
                                best_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, beam_tokens.size(-1)))
        
        return best_tokens.squeeze(1)
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        生成用于注意力机制的方形后续掩码
        
        参数:
            sz: 掩码大小
            
        返回:
            掩码矩阵 [sz, sz]
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    @classmethod
    def from_config(cls, config_path: str) -> 'Seq2SeqTransformer':
        """
        从配置文件加载模型
        
        参数:
            config_path: 配置文件路径
            
        返回:
            配置好的模型实例
        """
        # 加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 获取模型配置
        model_config = config.get('model', {})
        
        # 创建模型实例
        model = cls(
            vocab_size=model_config.get('vocab_size', 50000),
            d_model=model_config.get('embedding_dim', 512),
            nhead=model_config.get('num_heads', 8),
            num_encoder_layers=model_config.get('num_layers', 6),
            num_decoder_layers=model_config.get('num_layers', 6),
            dim_feedforward=model_config.get('hidden_dim', 2048),
            dropout=model_config.get('dropout', 0.1),
            max_seq_length=model_config.get('max_seq_length', 1024)
        )
        
        logger.info(f"Model initialized from config: {config_path}")
        return model


class PositionalEncoding(nn.Module):
    """
    位置编码层
    
    将位置信息添加到序列中，以便模型可以了解序列中的相对位置
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区而不是参数
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        将位置编码添加到输入嵌入
        
        参数:
            x: 输入嵌入 [batch_size, seq_len, d_model]
            
        返回:
            添加位置编码后的嵌入 [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def load_model(model_path: str, config_path: str = None) -> Seq2SeqTransformer:
    """
    加载预训练模型
    
    参数:
        model_path: 模型权重路径
        config_path: 配置文件路径，如果为None则使用相同目录下的model_arch_config.yaml
        
    返回:
        加载的模型
    """
    # 确定配置路径
    if config_path is None:
        config_path = os.path.join(os.path.dirname(model_path), 'model_arch_config.yaml')
    
    # 从配置创建模型
    model = Seq2SeqTransformer.from_config(config_path)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    logger.info(f"Model loaded from: {model_path}")
    return model


def save_model(model: Seq2SeqTransformer, save_path: str):
    """
    保存模型
    
    参数:
        model: 要保存的模型
        save_path: 保存路径
    """
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存模型
    torch.save(model.state_dict(), save_path)
    
    logger.info(f"Model saved to: {save_path}")