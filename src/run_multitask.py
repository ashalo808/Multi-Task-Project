import os
import sys
import argparse
import yaml
import logging
import torch
from typing import Dict, Any

from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.tasks.task_manager import TaskManager
from src.trainers.multitask_trainer import MultitaskTrainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="多任务学习训练脚本")
    
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/multitask_config.yaml",
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="输出目录"
    )
    
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="是否执行训练"
    )
    
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="是否执行评估"
    )
    
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="是否执行预测"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="指定要处理的任务名称，默认处理所有任务"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="运行设备，例如'cuda:0'或'cpu'"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging():
    """设置日志系统"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("multitask_run.log", mode="w")
        ]
    )
    logger = logging.getLogger(__name__)
    return logger


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置日志
    logger = setup_logging()
    logger.info(f"命令行参数: {args}")
    
    # 加载配置
    config = load_config(args.config)
    logger.info("已加载配置")
    
    # 设置输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定设备
    device = args.device
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    # 加载模型配置
    model_config = config.get("model", {})
    model_name = model_config.get("name", "bert-base-chinese")
    logger.info(f"使用模型: {model_name}")
    
    # 加载预训练模型和分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        logger.info(f"已加载预训练模型和分词器: {model_name}")
    except Exception as e:
        logger.error(f"加载模型时出错: {e}")
        return
    
    # 将分词器附加到模型，使任务可以访问
    model.tokenizer = tokenizer
    
    # 数据目录
    data_base_dir = config.get("data_dir", "./data")
    
    # 任务配置
    tasks_config = config.get("tasks", {})
    if not tasks_config:
        logger.error("配置中未找到任务定义")
        return
    
    # 初始化任务管理器
    task_manager = TaskManager(data_base_dir, model_config, tasks_config)
    logger.info("已初始化任务管理器")
    
    # 筛选任务
    selected_tasks = [args.task] if args.task else list(tasks_config.keys())
    filtered_tasks = {name: config for name, config in tasks_config.items() if name in selected_tasks}
    
    # 向任务管理器添加筛选后的任务
    for task_name in filtered_tasks.keys():
        try:
            task_manager.initialize_task(task_name)
            logger.info(f"已初始化任务: {task_name}")
        except Exception as e:
            logger.error(f"初始化任务 {task_name} 失败: {e}")
    
    # 训练配置
    train_config = config.get("training", {})
    num_epochs = train_config.get("num_epochs", 3)
    batch_size = train_config.get("batch_size", 32)
    gradient_accumulation_steps = train_config.get("gradient_accumulation_steps", 1)
    learning_rate = train_config.get("learning_rate", 5e-5)
    weight_decay = train_config.get("weight_decay", 0.01)
    warmup_steps = train_config.get("warmup_steps", 0)
    max_grad_norm = train_config.get("max_grad_norm", 1.0)
    early_stopping_patience = train_config.get("early_stopping_patience", 0)
    task_sampling_weights = train_config.get("task_sampling_weights", {})
    
    # 初始化训练器
    trainer = MultitaskTrainer(
        model=model,
        task_manager=task_manager,
        output_dir=output_dir,
        device=device,
        log_steps=train_config.get("log_steps", 10),
        save_steps=train_config.get("save_steps", 1000),
        eval_steps=train_config.get("eval_steps", 500)
    )
    logger.info("已初始化训练器")
    
    # 执行训练
    if args.do_train:
        logger.info("开始训练")
        
        # 设置优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 计算总训练步数（粗略估计）
        total_steps = 0
        for task_name in filtered_tasks.keys():
            task = task_manager.get_task(task_name)
            dataloader = task.get_dataloader("train", batch_size)
            total_steps += len(dataloader) // gradient_accumulation_steps
        total_steps *= num_epochs
        
        # 学习率调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 执行训练
        trainer.train(
            num_epochs=num_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            early_stopping_patience=early_stopping_patience,
            task_sampling_weights=task_sampling_weights
        )
        
        logger.info("训练完成")
    
    # 执行评估
    if args.do_eval:
        logger.info("开始评估")
        for task_name in filtered_tasks.keys():
            logger.info(f"评估任务: {task_name}")
            metrics = trainer.evaluate(task_name, batch_size=batch_size)
            logger.info(f"任务 {task_name} 评估结果: {metrics}")
        logger.info("评估完成")
    
    # 执行预测
    if args.do_predict:
        logger.info("开始预测")
        # 预测功能需要根据具体任务定制，此处仅为框架
        for task_name in filtered_tasks.keys():
            logger.info(f"为任务 {task_name} 生成预测结果")
            # 实际的预测代码在此处实现
        logger.info("预测完成")
    
    logger.info("脚本执行完成")


if __name__ == "__main__":
    main()