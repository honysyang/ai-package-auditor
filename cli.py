"""命令行接口模块"""
import argparse
from typing import Optional, Dict
from .auditor import AIPackageAuditor
from .models.ai_model_config import AIModelConfig
# from .auditor import AIPackageAuditor
import logging


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='AI依赖包漏洞审计工具')
    
    # 项目路径参数
    parser.add_argument('--project-path', type=str, default='.', 
                      help='项目路径，默认为当前目录')
    
    # 线程数参数
    parser.add_argument('--max-workers', type=int, default=5, 
                      help='并行查询的线程数，默认为5')
    
    # 报告打印参数
    parser.add_argument('--print', action='store_true', default=True,
                      help='是否在控制台打印报告，默认启用')
    parser.add_argument('--no-print', action='store_false', dest='print',
                      help='不在控制台打印报告')
    
    # JSON报告参数
    parser.add_argument('--json', action='store_true', default=False,
                      help='是否生成JSON报告，默认不启用')
    parser.add_argument('--json-path', type=str, default='ai_package_audit_report.json',
                      help='JSON报告路径，默认为ai_package_audit_report.json')
    parser.add_argument('--ai-json-path', type=str, default='ai_code_audit_report.json',
                      help='AI代码审计报告路径，默认为ai_code_audit_report.json')
    
    # AI模型配置参数
    parser.add_argument('--ai-config', type=str, 
                      help='AI模型配置文件路径')
    parser.add_argument('--ai-api-base', type=str, 
                      help='AI模型API地址')
    parser.add_argument('--ai-api-key', type=str, 
                      help='AI模型API密钥')
    parser.add_argument('--ai-model-name', type=str, 
                      help='AI模型名称')
    parser.add_argument('--ai-timeout', type=int, 
                      help='AI模型请求超时时间(秒)')
    parser.add_argument('--ai-max-tokens', type=int, 
                      help='AI模型最大token数')
    
    
    return parser.parse_args()


def get_ai_model_config(args) -> Optional[AIModelConfig]:
    """根据命令行参数获取AI模型配置，不完整则返回None"""
    # 优先级: 配置文件 > 命令行参数 > 环境变量
    if args.ai_config:
        config = AIModelConfig.from_config_file(args.ai_config)
    else:
        # 从环境变量加载基础配置
        config = AIModelConfig.from_environment()
        
        # 命令行参数覆盖环境变量
        if args.ai_api_base:
            config.api_base = args.ai_api_base
        if args.ai_api_key:
            config.api_key = args.ai_api_key
        if args.ai_model_name:
            config.model_name = args.ai_model_name
        if args.ai_timeout:
            config.timeout = args.ai_timeout
        if args.ai_max_tokens:
            config.max_tokens = args.ai_max_tokens
    
    # 检查配置完整性
    if not all([config.api_base, config.api_key, config.model_name]):
        return None  # 配置不完整，不启用AI功能
    
    return config


def main():
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 获取AI模型配置（可能为None）
        ai_config = get_ai_model_config(args)
        if ai_config:
            logging.info(f"检测到AI模型配置，将启用AI代码审计功能: {ai_config.model_name}")
            # 转换为字典传递给审计器
            ai_config_dict = {
                "api_base": ai_config.api_base,
                "api_key": ai_config.api_key,
                "model_name": ai_config.model_name,
                "timeout": ai_config.timeout,
                "max_tokens": ai_config.max_tokens
            }
        else:
            logging.info("未检测到完整的AI模型配置，将使用基础审计功能")
            ai_config_dict = None
        
        # 准备报告选项
        report_options = {
            "print": args.print,
            "json": args.json,
            "json_path": args.json_path,
            "ai_json_path": args.ai_json_path
        }
        
        # 初始化审计器（根据AI配置决定是否启用AI功能）
        auditor = AIPackageAuditor(
            max_workers=args.max_workers,
            ai_config=ai_config_dict
        )
        
        # 执行审计
        results = auditor.audit(
            project_path=args.project_path,
            report_options=report_options
        )
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"审计过程中发生错误: {str(e)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()