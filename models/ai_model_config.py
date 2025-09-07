from dataclasses import dataclass
import os
import json
from typing import Optional
from pathlib import Path

@dataclass
class AIModelConfig:
    """AI模型配置类，存储调用AI所需的所有参数"""
    api_base: str
    api_key: str
    model_name: str
    timeout: int = 5
    max_tokens: int = 2048
    temperature: float = 0.2
    top_p: float = 0.95

    @classmethod
    def from_config_file(cls, file_path: str) -> 'AIModelConfig':
        """从JSON配置文件加载AI模型配置"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            return cls(
                api_base=config_data.get('api_base'),
                api_key=config_data.get('api_key'),
                model_name=config_data.get('model_name'),
                timeout=config_data.get('timeout', 30),
                max_tokens=config_data.get('max_tokens', 2048),
                temperature=config_data.get('temperature', 0.2),
                top_p=config_data.get('top_p', 0.95)
            )
        except Exception as e:
            raise ValueError(f"加载AI配置文件失败: {str(e)}")

    @classmethod
    def from_environment(cls) -> 'AIModelConfig':
        """从环境变量加载AI模型配置"""
        return cls(
            api_base=os.getenv('AI_API_BASE', ''),
            api_key=os.getenv('AI_API_KEY', ''),
            model_name=os.getenv('AI_MODEL_NAME', ''),
            timeout=int(os.getenv('AI_TIMEOUT', 30)),
            max_tokens=int(os.getenv('AI_MAX_TOKENS', 2048)),
            temperature=float(os.getenv('AI_TEMPERATURE', 0.2)),
            top_p=float(os.getenv('AI_TOP_P', 0.95))
        )

    def to_dict(self) -> dict:
        """转换为字典表示（隐藏API密钥）"""
        return {
            'api_base': self.api_base,
            'model_name': self.model_name,
            'timeout': self.timeout,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p
        }
    