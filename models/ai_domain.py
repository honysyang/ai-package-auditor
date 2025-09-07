"""AI领域相关模型和分类模块"""
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Set

from ..models.dependency import Dependency


@dataclass
class AIPackageRisk:
    """AI包风险评估结果"""
    package: str
    version: str
    risk_score: float  # 0-10分，越高风险越大
    risk_factors: List[str]  # 风险因素
    ai_specific_issues: List[str]  # AI领域特定问题
    mitigation: str  # 缓解建议


class AIDomain(Enum):
    """AI应用领域分类"""
    NATURAL_LANGUAGE = "自然语言处理"
    COMPUTER_VISION = "计算机视觉"
    REINFORCEMENT_LEARNING = "强化学习"
    GENERATIVE_MODEL = "生成式模型"
    MULTIMODAL = "多模态模型"
    AI_INFRA = "AI基础设施"
    OTHER = "其他AI领域"


class AIPackageClassifier:
    """AI包分类器，识别AI相关包并分类"""
    
    # AI领域核心包清单
    AI_PACKAGES = {
        # 深度学习框架
        "tensorflow": AIDomain.OTHER,
        "keras": AIDomain.OTHER,
        "pytorch": AIDomain.OTHER,
        "torch": AIDomain.OTHER,
        "mxnet": AIDomain.OTHER,
        
        # 自然语言处理
        "transformers": AIDomain.NATURAL_LANGUAGE,
        "nltk": AIDomain.NATURAL_LANGUAGE,
        "spacy": AIDomain.NATURAL_LANGUAGE,
        "textattack": AIDomain.NATURAL_LANGUAGE,
        
        # 计算机视觉
        "opencv-python": AIDomain.COMPUTER_VISION,
        "torchvision": AIDomain.COMPUTER_VISION,
        "tensorflow-models": AIDomain.COMPUTER_VISION,
        "detectron2": AIDomain.COMPUTER_VISION,
        
        # 生成式模型
        "diffusers": AIDomain.GENERATIVE_MODEL,
        "gpt2": AIDomain.GENERATIVE_MODEL,
        "stable-baselines3": AIDomain.REINFORCEMENT_LEARNING,
        
        # 数据处理
        "numpy": AIDomain.OTHER,
        "pandas": AIDomain.OTHER,
        "scikit-learn": AIDomain.OTHER,
        "xgboost": AIDomain.OTHER,
        "lightgbm": AIDomain.OTHER
    }
    
    @classmethod
    def is_ai_package(cls, package_name: str) -> bool:
        """判断是否为AI相关包"""
        return package_name.lower() in cls.AI_PACKAGES
    
    @classmethod
    def classify_domain(cls, package_name: str) -> AIDomain:
        """分类AI包所属领域"""
        return cls.AI_PACKAGES.get(package_name.lower(), AIDomain.OTHER)
    
    @classmethod
    def get_ai_packages(cls, dependencies: Set[Dependency]) -> Dict[AIDomain, List[Dependency]]:
        """将依赖按AI领域分类"""
        result = {domain: [] for domain in AIDomain}
        for dep in dependencies:
            if cls.is_ai_package(dep.name):
                domain = cls.classify_domain(dep.name)
                result[domain].append(dep)
        return result
    