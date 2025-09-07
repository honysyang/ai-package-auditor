"""
AI包漏洞审计工具：支持依赖审计和AI代码审计功能
"""

# 导出核心类
from .auditor import AIPackageAuditor
from .models.ai_model_config import AIModelConfig

# 导出版本号（遵循PEP 396）
__version__ = "0.1.0"
    