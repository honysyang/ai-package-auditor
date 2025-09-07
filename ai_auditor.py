import requests
import json
from typing import List, Dict, Optional
from pathlib import Path
from .models.ai_model_config import AIModelConfig
from .exceptions import AIAuditError

class AICodeAuditor:
    """AI代码审计器，仅在有完整配置时启用"""
    
    def __init__(self, config: AIModelConfig):
        self.config = config
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }

    def audit_code(self, file_path: str, code_content: str) -> Dict:
        """使用AI模型审计单个代码文件"""
        try:
            # 构建提示词
            prompt = self._build_audit_prompt(file_path, code_content)
            
            # 调用AI API
            response = requests.post(
                self.config.api_base,
                headers=self.headers,
                json={
                    "model": self.config.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "max_tokens": self.config.max_tokens
                },
                timeout=self.config.timeout
            )
            
            if response.status_code != 200:
                raise AIAuditError(f"AI API请求失败: {response.text}")
                
            # 解析AI响应
            result = response.json()
            audit_content = result["choices"][0]["message"]["content"]
            
            # 结构化审计结果
            return self._parse_audit_result(file_path, audit_content)
            
        except Exception as e:
            raise AIAuditError(f"代码审计失败: {str(e)}") from e

    def _build_audit_prompt(self, file_path: str, code_content: str) -> str:
        """构建AI代码审计提示词"""
        return f"""你是一名专业的代码安全审计专家，尤其擅长AI/机器学习项目的安全审查。
请审计以下代码文件（{file_path}），重点关注：

1. 安全漏洞：
   - 代码注入风险
   - 数据泄露问题
   - 不安全的函数使用
   - 输入验证缺失
   - 权限控制问题

2. AI特定风险：
   - 模型加载安全（如加载不可信模型）
   - 训练数据处理漏洞
   - 推理过程中的安全问题
   - 模型序列化/反序列化风险
   - 过度拟合/数据污染风险

3. 代码质量问题：
   - 错误处理不完善
   - 日志记录不充分
   - 硬编码敏感信息
   - 依赖版本安全问题

请按照以下格式输出结果：
- 风险等级：[高/中/低/无]
- 问题摘要：[简要描述发现的问题]
- 详细分析：[详细说明问题所在及潜在影响]
- 修复建议：[具体可实施的修复方案]
- AI特定风险：[如果是AI相关代码，说明特有的风险]

代码内容：{code_content[:8000]}  # 限制代码长度，避免超出token限制"""

    def _parse_audit_result(self, file_path: str, audit_content: str) -> Dict:
        """解析AI审计结果为结构化数据"""
        # 简单解析提示词规定的格式
        sections = {
            "风险等级": "未指定",
            "问题摘要": "无",
            "详细分析": "无",
            "修复建议": "无",
            "AI特定风险": "无"
        }
        
        for line in audit_content.split('\n'):
            for key in sections:
                if line.startswith(f"- {key}："):
                    sections[key] = line[len(f"- {key}："):].strip()
        
        return {
            "file_path": file_path,
            "risk_level": sections["风险等级"],
            "summary": sections["问题摘要"],
            "detailed_analysis": sections["详细分析"],
            "recommendations": sections["修复建议"],
            "ai_specific_risk": sections["AI特定风险"],
            "raw_audit": audit_content  # 保留原始审计内容
        }

    def audit_files(self, file_paths: List[str]) -> List[Dict]:
        """批量审计多个文件"""
        results = []
        for path in file_paths:
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                results.append(self.audit_code(path, content))
            except Exception as e:
                results.append({
                    "file_path": path,
                    "error": f"无法审计文件: {str(e)}",
                    "risk_level": "未知"
                })
        return results

    def save_audit_report(self, results: List[Dict], output_path: str):
        """保存AI审计报告为JSON文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "audit_results": results,
                    "model_used": self.config.model_name,
                    "audit_time": self._get_current_time()
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise AIAuditError(f"保存AI审计报告失败: {str(e)}")

    def _get_current_time(self) -> str:
        """获取当前时间字符串"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    