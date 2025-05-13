import os
import importlib
import inspect
import logging
from typing import Dict, List, Tuple, Any, Optional, Type, Set

from src.tasks.base_task import BaseTask


class TaskManager:
    """任务管理器，负责注册和获取任务处理器"""
    
    def __init__(self, base_data_dir: str, model_config: Dict[str, Any], tasks_config: Dict[str, Any]):
        """
        初始化任务管理器
        
        Args:
            base_data_dir: 数据根目录
            model_config: 模型配置
            tasks_config: 任务配置字典，键为任务名称
        """
        self.base_data_dir = base_data_dir
        self.model_config = model_config
        self.tasks_config = tasks_config
        self.tasks = {}  # 存储已初始化的任务实例
        self.task_classes = {}  # 存储任务类映射
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 自动发现和注册任务类
        self._discover_task_classes()
    
    def _discover_task_classes(self):
        """自动发现和注册所有任务类"""
        # 1. 搜索任务类所在的模块路径
        task_paths = [
            "src.tasks",  # 基础任务目录
            "src.tasks.examples"  # 示例任务目录
        ]
        
        for path in task_paths:
            try:
                module = importlib.import_module(path)
                
                # 获取模块文件所在目录
                module_path = os.path.dirname(inspect.getfile(module))
                
                # 遍历目录下的所有Python文件
                for file in os.listdir(module_path):
                    if file.endswith(".py") and not file.startswith("__"):
                        # 获取模块名
                        module_name = file[:-3]  # 去掉.py扩展名
                        full_module_name = f"{path}.{module_name}"
                        
                        try:
                            # 动态导入模块
                            task_module = importlib.import_module(full_module_name)
                            
                            # 遍历模块中的所有类
                            for name, obj in inspect.getmembers(task_module, inspect.isclass):
                                # 检查是否是BaseTask的子类且不是BaseTask本身
                                if (issubclass(obj, BaseTask) and 
                                    obj != BaseTask and 
                                    hasattr(obj, '_get_task_type_static')):
                                    
                                    # 获取任务类型
                                    task_type = obj._get_task_type_static()
                                    
                                    # 注册任务类
                                    self._register_task_class(task_type, obj)
                        except (ImportError, AttributeError) as e:
                            self.logger.debug(f"导入模块 {full_module_name} 时出错: {e}")
            except ImportError:
                self.logger.debug(f"找不到模块 {path}")
    
    def _register_task_class(self, task_type: str, task_class: Type[BaseTask]):
        """
        注册任务类
        
        Args:
            task_type: 任务类型标识
            task_class: 任务类
        """
        if task_type in self.task_classes:
            self.logger.warning(f"任务类型 '{task_type}' 已注册，正在覆盖")
        
        self.task_classes[task_type] = task_class
        self.logger.debug(f"已注册任务类 '{task_class.__name__}' 为类型 '{task_type}'")
    
    def initialize_task(self, task_name: str) -> BaseTask:
        """
        初始化特定任务
        
        Args:
            task_name: 任务名称
        
        Returns:
            初始化后的任务实例
        
        Raises:
            ValueError: 如果任务类型未注册或配置错误
        """
        if task_name in self.tasks:
            return self.tasks[task_name]
        
        if task_name not in self.tasks_config:
            raise ValueError(f"配置中找不到任务 '{task_name}'")
        
        task_config = self.tasks_config[task_name]
        task_type = task_config.get("type")
        
        if not task_type:
            raise ValueError(f"任务 '{task_name}' 缺少 'type' 字段")
        
        if task_type not in self.task_classes:
            raise ValueError(f"找不到类型为 '{task_type}' 的任务处理器")
        
        # 获取数据目录
        data_subdir = task_config.get("data_subdir", task_name)
        data_dir = os.path.join(self.base_data_dir, data_subdir)
        
        # 准备任务参数
        task_params = task_config.get("params", {})
        
        # 初始化任务实例
        task_class = self.task_classes[task_type]
        task = task_class(data_dir, self.model_config, task_params)
        
        # 存储任务实例
        self.tasks[task_name] = task
        
        self.logger.info(f"已初始化任务 '{task_name}' (类型: {task_type})")
        return task
    
    def initialize_all_tasks(self) -> List[BaseTask]:
        """
        初始化配置中的所有任务
        
        Returns:
            初始化的任务实例列表
        """
        tasks = []
        for task_name in self.tasks_config.keys():
            try:
                task = self.initialize_task(task_name)
                tasks.append(task)
            except Exception as e:
                self.logger.error(f"初始化任务 '{task_name}' 失败: {e}")
        
        return tasks
    
    def get_task(self, task_name: str) -> BaseTask:
        """
        获取任务实例，如果未初始化则先初始化
        
        Args:
            task_name: 任务名称
        
        Returns:
            任务实例
        """
        if task_name not in self.tasks:
            return self.initialize_task(task_name)
        return self.tasks[task_name]
    
    def get_all_task_names(self) -> Set[str]:
        """
        获取所有配置的任务名称
        
        Returns:
            任务名称集合
        """
        return set(self.tasks_config.keys())
    
    def get_registered_task_types(self) -> Dict[str, Type[BaseTask]]:
        """
        获取所有注册的任务类型
        
        Returns:
            任务类型到任务类的映射
        """
        return self.task_classes.copy()