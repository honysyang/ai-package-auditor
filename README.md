# AI Package Auditor: 智能包漏洞审计工具

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)

AI Package Auditor 是一款专为AI/机器学习项目设计的包漏洞审计工具，支持**命令行独立运行**和**代码模块调用**两种模式。通过静态分析与AI增强审计结合，帮助开发者识别依赖包中的安全风险，特别是针对AI框架（如TensorFlow、PyTorch）的特定漏洞。


## 核心功能

### 基础审计能力
- **依赖自动收集**：扫描项目依赖文件（`requirements.txt`、`pyproject.toml`等）及已安装包
- **AI包智能分类**：按领域（自然语言处理、计算机视觉等）识别AI相关包
- **多源漏洞查询**：整合OSV等数据库，检测已知安全漏洞
- **风险等级评估**：基于漏洞 severity、影响范围等因素评分（0-10分）
- **多格式报告**：支持控制台输出和JSON格式报告，包含修复建议

### AI增强功能（可选）
当配置AI模型后，工具将额外提供：
- **代码级安全审计**：对核心AI代码进行深度分析
- **上下文感知检测**：识别AI特有的风险（如模型加载漏洞、数据处理安全）
- **智能修复建议**：针对AI框架特性提供定制化修复方案


## 安装指南

### 前置要求
- Python 3.8+
- 网络连接（用于漏洞数据库查询和AI API调用）

### 安装方式

#### 1. 从Wheel包安装（推荐）# 下载或构建whl包后安装
pip install ai_audit_tool-0.1.0-py3-none-any.whl

# 如需启用AI功能，安装可选依赖
pip install ai_audit_tool-0.1.0-py3-none-any.whl[ai]
#### 2. 源码安装# 克隆仓库
git clone https://github.com/honysyang/ai-package-auditor.git
cd ai-package-auditor

# 安装
pip install .

# 安装开发依赖（用于二次开发）
pip install .[dev]

## 使用指南

### 模式1：命令行工具
安装后可直接通过`ai-audit`命令运行：
# 基础审计（当前目录，控制台输出报告）
ai-audit

# 自定义项目路径并生成JSON报告
ai-audit --project-path ./my_ai_project --json --json-path audit_report.json

# 启用AI审计（通过命令行参数配置AI模型）
ai-audit \
  --project-path ./my_ai_project \
  --ai-api-base "https://api.chatanywhere.tech/v1/chat/completions" \
  --ai-api-key "your-api-key" \
  --ai-model-name "gpt-4o-mini-ca" \
  --max-workers 8 \
  --json
#### 命令行参数说明
| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--project-path` | 目标项目路径 | `.`（当前目录） |
| `--max-workers` | 并行查询线程数 | `5` |
| `--print`/`--no-print` | 是否在控制台打印报告 | 启用（`--print`） |
| `--json` | 是否生成JSON报告 | 禁用 |
| `--json-path` | JSON报告路径 | `ai_audit_report.json` |
| `--ai-api-base` | AI模型API地址（启用AI功能需提供） | - |
| `--ai-api-key` | AI模型API密钥（启用AI功能需提供） | - |
| `--ai-model-name` | AI模型名称（启用AI功能需提供） | - |


### 模式2：代码模块调用
作为Python模块集成到现有工作流中：
from ai_audit import AIPackageAuditor, AIModelConfig

# 1. 基础审计（无AI功能）
auditor = AIPackageAuditor(max_workers=5)
results = auditor.audit(
    project_path="./my_ai_project",
    report_options={
        "print": True,
        "json": True,
        "json_path": "custom_report.json"
    }
)

# 2. 启用AI审计
ai_config = AIModelConfig(
    api_base="https://api.chatanywhere.tech/v1/chat/completions",
    api_key="your-api-key",
    model_name="gpt-4o-mini-ca",
    timeout=30  # 超时时间（秒）
)

auditor = AIPackageAuditor(
    max_workers=8,
    ai_config=ai_config
)

results = auditor.audit(
    project_path="./my_ai_project",
    report_options={"print": True, "json": True}
)

# 解析审计结果
high_risk_pkgs = results["base_audit"]["summary"]["high_risk_packages"]
print(f"发现 {len(high_risk_pkgs)} 个高风险包")

## AI模型配置

AI功能支持三种配置方式（优先级从高到低）：

### 1. 配置文件（推荐）
创建`ai_config.json`：{
  "api_base": "https://api.chatanywhere.tech/v1/chat/completions",
  "api_key": "your-api-key",
  "model_name": "gpt-4o-mini-ca",
  "timeout": 30,
  "max_tokens": 2048
}使用时指定配置文件：ai-audit --ai-config ai_config.json
### 2. 环境变量export AI_API_BASE="https://api.chatanywhere.tech/v1/chat/completions"
export AI_API_KEY="your-api-key"
export AI_MODEL_NAME="gpt-4o-mini-ca"
ai-audit
### 3. 命令行参数
如「使用指南」中命令行示例所示。


## 报告解读

工具生成两种报告（取决于是否启用AI功能）：

### 1. 基础审计报告
包含依赖分析和漏洞评估：{
  "meta": {
    "audit_time": "2023-10-01 15:30:00",
    "project_path": "/path/to/project",
    "total_dependencies": 28,
    "total_ai_packages": 5
  },
  "summary": {
    "high_risk_count": 2,
    "medium_risk_count": 3,
    "low_risk_count": 1
  },
  "vulnerable_packages": [
    {
      "package": "tensorflow",
      "version": "2.5.0",
      "risk_score": 8.5,
      "vulnerabilities": [
        {
          "id": "CVE-2023-1234",
          "description": "TensorFlow存在代码执行漏洞...",
          "fix_versions": ["2.10.0"]
        }
      ],
      "mitigation": "建议升级到2.10.0版本..."
    }
  ]
}
### 2. AI代码审计报告（启用AI时）
额外包含代码级分析：{
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

## 工程化设计

### 项目结构ai_audit/
├── __init__.py           # 包导出声明
├── auditor.py            # 审计主逻辑
├── cli.py                # 命令行入口
├── models/               # 数据模型（依赖、漏洞、AI配置等）
├── collectors/           # 依赖收集模块
├── assessors/            # 风险评估模块
├── services/             # 漏洞查询服务
└── ai_auditor.py         # AI审计模块（条件加载）
### 核心技术特点
- **条件性功能加载**：AI模块仅在配置完整时加载，不影响基础功能
- **多源配置兼容**：支持文件、环境变量、命令行参数混合配置
- **并行化查询**：通过线程池加速漏洞数据库查询
- **跨平台兼容**：路径处理适配Windows/macOS/Linux
- **可扩展架构**：模块化设计便于添加新的漏洞数据源或AI模型


## 开发指南

### 本地开发# 安装开发依赖
pip install .[dev]

# 运行测试
pytest tests/

# 代码格式化
black ai_audit/

# 构建Wheel包
python -m build
### 贡献指南
1. Fork 仓库并创建分支（`feature/xxx` 或 `fix/xxx`）
2. 提交代码前确保测试通过
3. 提交PR时请说明功能变更或修复内容


## 许可证
本项目基于 [MIT许可证](LICENSE) 开源，允许自由使用、修改和分发。
    