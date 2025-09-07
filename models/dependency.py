"""依赖项模型模块"""
from typing import Dict


class Dependency:
    """依赖项模型"""
    def __init__(self, name: str, version: str):
        self.name = name.lower()
        self.version = version

    def __eq__(self, other):
        if not isinstance(other, Dependency):
            return False
        return self.name == other.name and self.version == other.version

    def __hash__(self):
        return hash((self.name, self.version))

    def __repr__(self):
        return f"{self.name}=={self.version}"
    
    def to_dict(self) -> Dict[str, str]:
        """转换为字典，用于JSON序列化"""
        return {
            "name": self.name,
            "version": self.version,
            "full_name": f"{self.name}=={self.version}"
        }
    