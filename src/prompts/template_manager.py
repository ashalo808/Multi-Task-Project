import os
import yaml
import json
from typing import Dict, List, Optional, Union, Any


class PromptTemplate:
    """提示模板类，用于管理单个提示模板的格式化"""
    
    def __init__(self, input_format: str, target_format: str):
        """
        初始化提示模板
        
        参数:
            input_format: 输入格式模板
            target_format: 目标格式模板
        """
        self.input_format = input_format
        self.target_format = target_format
    
    def format_input(self, **kwargs) -> str:
        """
        根据提供的关键字参数格式化输入模板
        
        参数:
            **kwargs: 用于替换模板中的占位符的关键字参数
            
        返回:
            格式化后的输入字符串
        """
        try:
            return self.input_format.format(**kwargs)
        except KeyError as e:
            missing_key = str(e).strip("'")
            raise KeyError(f"Missing key '{missing_key}' for input template formatting")
    
    def format_target(self, **kwargs) -> str:
        """
        根据提供的关键字参数格式化目标模板
        
        参数:
            **kwargs: 用于替换模板中的占位符的关键字参数
            
        返回:
            格式化后的目标字符串
        """
        try:
            return self.target_format.format(**kwargs)
        except KeyError as e:
            missing_key = str(e).strip("'")
            raise KeyError(f"Missing key '{missing_key}' for target template formatting")


class PromptManager:
    """提示管理器类，用于管理和应用各种提示模板"""
    
    def __init__(self, config_path: str):
        """
        初始化提示管理器
        
        参数:
            config_path: 提示模板配置文件的路径
        """
        self.config_path = config_path
        self.templates = {}
        self.settings = {}
        self.task_template_mapping = {}
        self._load_config()
    
    def _load_config(self):
        """从配置文件加载模板配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 加载模板
            templates_config = config.get('templates', {})
            for name, template_config in templates_config.items():
                self.templates[name] = PromptTemplate(
                    input_format=template_config.get('input_format', ''),
                    target_format=template_config.get('target_format', '')
                )
            
            # 加载设置
            self.settings = config.get('settings', {})
            
            # 加载任务到模板的映射
            self.task_template_mapping = config.get('task_template_mapping', {})
            
            print(f"Loaded {len(self.templates)} templates from {self.config_path}")
        except Exception as e:
            print(f"Error loading template config from {self.config_path}: {e}")
            # 设置默认的空模板
            self.templates['default'] = PromptTemplate(
                input_format="{input_text}",
                target_format="{output_text}"
            )
            self.settings = {
                'default_template': 'default',
                'max_input_length': 1024,
                'max_target_length': 256
            }
    
    def get_template(self, template_name: Optional[str] = None) -> PromptTemplate:
        """
        获取指定名称的模板
        
        参数:
            template_name: 模板名称，如果为None则返回默认模板
            
        返回:
            PromptTemplate实例
        """
        if template_name is None:
            template_name = self.settings.get('default_template', 'custom')
        
        if template_name not in self.templates:
            print(f"Warning: Template '{template_name}' not found, using default template")
            template_name = self.settings.get('default_template', 'custom')
        
        return self.templates[template_name]
    
    def get_template_by_task(self, task_type: str) -> PromptTemplate:
        """
        根据任务类型获取对应的模板
        
        参数:
            task_type: 任务类型
            
        返回:
            PromptTemplate实例
        """
        template_name = self.task_template_mapping.get(task_type)
        return self.get_template(template_name)
    
    def format_input(self, template_name: Optional[str] = None, **kwargs) -> str:
        """
        格式化指定模板的输入
        
        参数:
            template_name: 模板名称，如果为None则使用默认模板
            **kwargs: 用于替换模板中的占位符的关键字参数
            
        返回:
            格式化后的输入字符串
        """
        template = self.get_template(template_name)
        formatted_input = template.format_input(**kwargs)
        
        # 应用长度限制
        max_length = self.settings.get('max_input_length', 1024)
        if len(formatted_input) > max_length:
            print(f"Warning: Input prompt exceeds maximum length of {max_length} characters")
            formatted_input = formatted_input[:max_length]
        
        return formatted_input
    
    def format_target(self, template_name: Optional[str] = None, **kwargs) -> str:
        """
        格式化指定模板的目标
        
        参数:
            template_name: 模板名称，如果为None则使用默认模板
            **kwargs: 用于替换模板中的占位符的关键字参数
            
        返回:
            格式化后的目标字符串
        """
        template = self.get_template(template_name)
        formatted_target = template.format_target(**kwargs)
        
        # 应用长度限制
        max_length = self.settings.get('max_target_length', 256)
        if len(formatted_target) > max_length:
            print(f"Warning: Target text exceeds maximum length of {max_length} characters")
            formatted_target = formatted_target[:max_length]
        
        return formatted_target
    
    def process_example(self, example: Dict[str, Any], task_type: Optional[str] = None) -> Dict[str, str]:
        """
        处理单个训练/评估样例
        
        参数:
            example: 包含输入和可选目标的样例字典
            task_type: 任务类型，用于选择模板
            
        返回:
            包含格式化后的输入和目标的字典
        """
        # 确定使用哪个模板
        template_name = None
        if task_type:
            template_name = self.task_template_mapping.get(task_type)
        elif 'type' in example:
            template_name = self.task_template_mapping.get(example['type'])
        
        # 获取模板
        template = self.get_template(template_name)
        
        try:
            # 格式化输入
            input_prompt = self.format_input(template_name, **example)
            
            # 格式化目标（如果有）
            target_text = ""
            if 'target' in example:
                # 直接使用对应的target进行格式化
                target_text = template.format_target(**example)
        except KeyError as e:
            print(f"格式化错误: {e}")
            # 使用默认格式，避免抛出异常
            if 'input_text' in example:
                input_prompt = example['input_text']
            else:
                input_prompt = str(example)
            
            if 'target' in example:
                target_text = str(example['target'])
            else:
                target_text = ""
        
        return {
            'input_prompt': input_prompt,
            'target_text': target_text
        }
    
    def process_batch(self, examples: List[Dict[str, Any]], task_type: Optional[str] = None) -> List[Dict[str, str]]:
        """
        批量处理样例
        
        参数:
            examples: 样例列表
            task_type: 任务类型，用于选择模板
            
        返回:
            包含格式化后的输入和目标的字典列表
        """
        return [self.process_example(example, task_type) for example in examples]
    
    def save_templates(self, save_path: Optional[str] = None):
        """
        保存当前的模板配置
        
        参数:
            save_path: 保存路径，如果为None则使用原始配置路径
        """
        if save_path is None:
            save_path = self.config_path
        
        config = {
            'templates': {},
            'settings': self.settings,
            'task_template_mapping': self.task_template_mapping
        }
        
        for name, template in self.templates.items():
            config['templates'][name] = {
                'input_format': template.input_format,
                'target_format': template.target_format
            }
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            print(f"Templates saved to {save_path}")
        except Exception as e:
            print(f"Error saving templates to {save_path}: {e}")
    
    def add_template(self, name: str, input_format: str, target_format: str):
        """
        添加新模板
        
        参数:
            name: 模板名称
            input_format: 输入格式
            target_format: 目标格式
        """
        self.templates[name] = PromptTemplate(input_format, target_format)
        print(f"Template '{name}' added")
    
    def add_task_mapping(self, task_type: str, template_name: str):
        """
        添加任务到模板的映射
        
        参数:
            task_type: 任务类型
            template_name: 模板名称
        """
        if template_name not in self.templates:
            print(f"Warning: Template '{template_name}' does not exist")
        
        self.task_template_mapping[task_type] = template_name
        print(f"Mapped task '{task_type}' to template '{template_name}'")


def load_data(data_path: str) -> List[Dict[str, Any]]:
    """
    从JSONL文件加载数据
    
    参数:
        data_path: JSONL文件路径
        
    返回:
        数据字典列表
    """
    examples = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                examples.append(json.loads(line))
        print(f"Loaded {len(examples)} examples from {data_path}")
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
    
    return examples


# 示例用法
if __name__ == "__main__":
    # 加载模板管理器
    config_path = "prompt_templates_config.yaml"
    prompt_manager = PromptManager(config_path)
    
    # 示例数据
    example = {
        "input": "这是关于哪方面的新闻： 故事,文化,娱乐,体育,财经,房产,汽车,教育,科技,军事,旅游,国际,股票,农业,游戏?崔万军合同到期 广州龙狮主教练离职\n答案：",
        "target": "体育",
        "answer_choices": ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游", "国际", "股票", "农业", "游戏"],
        "type": "classify"
    }
    
    # 处理示例
    processed = prompt_manager.process_example(example)
    
    print("\nInput Prompt:")
    print(processed['input_prompt'])
    
    print("\nTarget Text:")
    print(processed['target_text'])