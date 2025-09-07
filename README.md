# AI Package Auditor · 智能包漏洞审计工具

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)

> 专为 AI/机器学习项目设计的依赖包漏洞审计工具，支持 **命令行独立运行** 和 **代码模块调用**。结合静态分析与 AI 增强审计，帮助开发者高效识别依赖包中的安全风险，尤其针对 AI 框架（如 TensorFlow、PyTorch）特定漏洞。

---

## 目录

- [项目亮点](#项目亮点)
- [核心功能](#核心功能)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [使用说明](#使用说明)
  - [命令行模式](#命令行模式)
  - [代码集成模式](#代码集成模式)
- [AI 模型配置](#ai-模型配置)
- [报告解读](#报告解读)
- [项目架构](#项目架构)
- [开发&贡献指南](#开发贡献指南)
- [许可证](#许可证)

---

## 项目亮点

- 🚀 **AI增强审计**：支持代码级安全分析，智能识别AI特有风险
- 📦 **依赖自动收集**：自动识别项目依赖及AI相关包
- 🛡️ **多源漏洞查询**：集成主流安全数据库
- 📊 **报告精细**：支持控制台和JSON格式报告、风险评分
- 🔌 **灵活集成**：命令行/模块调用两种模式

---

## 核心功能

### 基础审计
- 自动收集依赖（`requirements.txt`、`pyproject.toml`等）
- AI包智能分类（NLP、CV等领域包识别）
- 多源漏洞检测（集成 OSV 等数据库）
- 风险等级评估（0-10分）
- 多格式报告输出（控制台、JSON）

### AI增强功能（可选）
- 代码级安全审计
- 上下文感知检测（模型加载、数据处理等安全风险）
- 智能修复建议

---

## 安装指南

### 前置要求
- Python 3.8 及以上
- 网络连接（用于漏洞数据库及AI API）

### 推荐安装方式

```bash
# 安装主程序
pip install ai_audit_tool-0.1.0-py3-none-any.whl

# 启用AI功能（可选）
pip install ai_audit_tool-0.1.0-py3-none-any.whl[ai]
```

### 源码安装

```bash
git clone https://github.com/honysyang/ai-package-auditor.git
cd ai-package-auditor
pip install .
# 开发依赖
pip install .[dev]
```

---

## 快速开始

### 命令行审计

```bash
# 基础审计
ai-audit

# 指定项目路径并输出JSON报告
ai-audit --project-path ./my_ai_project --json --json-path audit_report.json

# 启用AI审计
ai-audit \
  --project-path ./my_ai_project \
  --ai-api-base "https://api.chatanywhere.tech/v1/chat/completions" \
  --ai-api-key "your-api-key" \
  --ai-model-name "gpt-4o-mini-ca" \
  --max-workers 8 \
  --json
```

### 命令行参数说明

| 参数             | 描述                           | 默认值                   |
| ---------------- | ------------------------------ | ------------------------ |
| `--project-path` | 目标项目路径                   | `.`（当前目录）          |
| `--max-workers`  | 并行查询线程数                 | `5`                      |
| `--print`/`--no-print` | 控制台打印报告         | 启用                     |
| `--json`         | 生成JSON报告                   | 禁用                     |
| `--json-path`    | JSON报告路径                   | `ai_audit_report.json`   |
| `--ai-api-base`  | AI模型API地址（启用AI需提供）  | -                        |
| `--ai-api-key`   | AI模型API密钥（启用AI需提供）  | -                        |
| `--ai-model-name`| AI模型名称（启用AI需提供）     | -                        |

---

## 使用说明

### 命令行模式

安装后使用 `ai-audit` 命令，具体参数见上表。

### 代码集成模式

```python
from ai_audit import AIPackageAuditor, AIModelConfig

# 基础审计（不含AI）
auditor = AIPackageAuditor(max_workers=5)
results = auditor.audit(
    project_path="./my_ai_project",
    report_options={
        "print": True,
        "json": True,
        "json_path": "custom_report.json"
    }
)

# 启用AI审计
ai_config = AIModelConfig(
    api_base="https://api.chatanywhere.tech/v1/chat/completions",
    api_key="your-api-key",
    model_name="gpt-4o-mini-ca",
    timeout=30
)
auditor = AIPackageAuditor(max_workers=8, ai_config=ai_config)
results = auditor.audit(
    project_path="./my_ai_project",
    report_options={"print": True, "json": True}
)
print(f"发现 {len(results['base_audit']['summary']['high_risk_packages'])} 个高风险包")
```

---

## AI 模型配置

支持三种方式（优先级从高到低）：

1. **配置文件（推荐）**
    ```json
    {
      "api_base": "https://api.chatanywhere.tech/v1/chat/completions",
      "api_key": "your-api-key",
      "model_name": "gpt-4o-mini-ca",
      "timeout": 30,
      "max_tokens": 2048
    }
    ```
    使用：`ai-audit --ai-config ai_config.json`

2. **环境变量**
    ```bash
    export AI_API_BASE="https://api.chatanywhere.tech/v1/chat/completions"
    export AI_API_KEY="your-api-key"
    export AI_MODEL_NAME="gpt-4o-mini-ca"
    ai-audit
    ```

3. **命令行参数**
    （见“快速开始”命令行示例）

---

## 报告解读

- **基础审计报告**：依赖分析与漏洞评估  
  示例：
    ```json
    {
      "meta": { ... },
      "summary": { ... },
      "vulnerable_packages": [
        {
          "package": "tensorflow",
          "version": "2.5.0",
          "risk_score": 8.5,
          "vulnerabilities": [ ... ],
          "mitigation": "建议升级到2.10.0版本..."
        }
      ]
    }
    ```

- **AI代码审计报告**（启用AI时）：  
  示例：
    ```json
    {
      "ai_audit_results": [
        {
          "file_path": "models/trainer.py",
          "risk_score": 7.2,
          "issues": [
            {
              "description": "模型加载未验证文件完整性，可能导致恶意模型执行",
              "suggestion": "添加模型哈希校验，使用tf.saved_model.load时验证签名"
            }
          ]
        }
      ]
    }
    ```

---

## 项目架构

```
ai_audit/
├── __init__.py           # 包导出声明
├── auditor.py            # 审计主逻辑
├── cli.py                # 命令行入口
├── models/               # 数据模型
├── collectors/           # 依赖收集
├── assessors/            # 风险评估
├── services/             # 漏洞查询服务
└── ai_auditor.py         # AI审计模块
```

**技术特点**  
- 条件性功能加载：AI模块仅在配置完整时加载
- 多源配置兼容：支持文件、环境变量、命令行参数
- 并行化漏洞查询
- 跨平台兼容（Windows/macOS/Linux）
- 高可扩展性架构

---

## 开发贡献指南

### 本地开发

```bash
pip install .[dev]    # 安装开发依赖
pytest tests/         # 运行测试
black ai_audit/       # 代码格式化
python -m build       # 构建Wheel包
```

### 贡献流程

1. Fork 仓库并创建分支（`feature/xxx` 或 `fix/xxx`）
2. 确保测试全部通过
3. 提交 PR 时请详细说明变更内容

---

## 许可证

本项目采用 [MIT License](LICENSE) 开源，欢迎自由使用、修改与分发。

---
