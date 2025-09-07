"""AI包审计主类模块"""
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Union, Set, List

from .collectors.dependency_collector import AIDependencyCollector
from .models.ai_domain import AIPackageClassifier, AIDomain
from .models.dependency import Dependency
from .models.vulnerability import Vulnerability
from .services.vulnerability_service import VulnerabilityClient
from .assessors.vulnerability_assessor import AIVulnerabilityAssessor
from .ai_auditor import AICodeAuditor  # 延迟导入AI审计器

class AIPackageAuditor:
    """AI模型及应用专用包审计工具"""
    
    def __init__(self, max_workers: int = 5, ai_config: Optional[Dict] = None):
        """
        初始化AI包审计器
        
        Args:
            max_workers: 并行查询的线程数
        """
        self.collector = AIDependencyCollector()
        self.classifier = AIPackageClassifier()
        self.vuln_client = VulnerabilityClient()
        self.risk_assessor = AIVulnerabilityAssessor()
        self.max_workers = max_workers

         # 条件性初始化AI审计器
        self.ai_auditor = None
        if ai_config:
            try:
                from .models.ai_model_config import AIModelConfig
                # 转换为AIModelConfig实例
                config_obj = AIModelConfig(
                    api_base=ai_config["api_base"],
                    api_key=ai_config["api_key"],
                    model_name=ai_config["model_name"],
                    timeout=ai_config.get("timeout", 30),
                    max_tokens=ai_config.get("max_tokens", 2048)
                )
                self.ai_auditor = AICodeAuditor(config_obj)
            except Exception as e:
                print(f"初始化AI审计器失败，将禁用AI功能: {str(e)}")

    def audit(
        self, 
        project_path: str = ".",
        report_options: Optional[Dict[str, Union[bool, str]]] = None
    ) -> Dict:
        """
        执行AI项目包审计
        self:    根据AI配置决定是否启用AI代码审计
        Args:
            project_path: 项目路径，默认当前目录
            report_options: 报告生成配置，默认 {"print": True, "json": False, "json_path": "ai_package_audit_report.json"}
                - print: 是否在控制台打印报告（bool）
                - json: 是否生成JSON报告（bool）
                - json_path: JSON报告路径（str，相对项目根目录）
        
        Returns:
            审计结果字典
        """
        # 初始化默认报告配置
        report_opts = {
            "print": True,
            "json": False,
            "json_path": "ai_package_audit_report.json",
            "ai_json_path": "ai_code_audit_report.json"
        }
        # 合并用户配置（用户配置覆盖默认）
        if report_options:
            report_opts.update(report_options)
        
        start_time = time.time()
        print("开始AI项目包审计...")
        
        # 1. 收集依赖
        dependencies = self.collector.collect_all(project_path)
        print(f"共收集到 {len(dependencies)} 个依赖项")
        
        # 2. 分类AI包
        ai_packages_by_domain = self.classifier.get_ai_packages(dependencies)
        ai_package_count = sum(len(pkgs) for pkgs in ai_packages_by_domain.values())
        print(f"识别到 {ai_package_count} 个AI相关包")
        
        # 3. 查询漏洞
        print(f"开始查询漏洞 (并行线程: {self.max_workers})...")
        vulnerability_results = self.vuln_client.query_batch(dependencies, self.max_workers)
        
        # 4. AI风险评估
        print("正在进行AI风险评估...")
        audit_results = self._assess_ai_risks(vulnerability_results, ai_packages_by_domain)
        
        ai_audit_results = None
        if self.ai_auditor:
            print("开始AI代码审计...")
            # 获取项目中的关键代码文件（优先AI相关）
            code_files = self._collect_code_files(project_path)
            print(f"发现 {len(code_files)} 个代码文件，准备进行AI审计...")
            
            # 执行AI审计
            ai_audit_results = self.ai_auditor.audit_files(code_files)
            
            # 保存AI审计报告
            if report_opts["json"]:
                self.ai_auditor.save_audit_report(
                    ai_audit_results, 
                    report_opts["ai_json_path"]
                )
        
        
        # 5. 生成报告（根据配置执行）
        duration = time.time() - start_time
        print(f"审计完成! 总耗时: {duration:.2f}s")
        
        # 补充审计元信息（用于JSON报告）
        audit_results["meta"] = {
            "audit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "project_path": str(Path(project_path).absolute()),
            "total_dependencies": len(dependencies),
            "total_ai_packages": ai_package_count,
            "audit_duration": f"{duration:.2f}s",
            "max_workers": self.max_workers,
            "ai_audit_enabled": self.ai_auditor is not None
        }
        
        # 添加AI审计结果（如果启用）
        if ai_audit_results:
            audit_results["ai_audit"] = ai_audit_results
        
        # 执行报告生成
        if report_opts["print"]:
            self.print_ai_audit_report(audit_results)
        if report_opts["json"]:
            self._generate_json_report(audit_results, report_opts["json_path"])
        
        return audit_results


    def _collect_code_files(self, project_path: str, max_files: int = 20) -> List[str]:
        """收集项目中的代码文件（优先AI相关）"""
        code_extensions = ('.py', '.ipynb', '.json', '.yaml', '.yml')
        ai_related_dirs = ['models', 'src', 'ai', 'ml', 'notebooks']
        
        project_path = Path(project_path)
        files = []
        
        # 优先收集AI相关目录
        for dir_name in ai_related_dirs:
            dir_path = project_path / dir_name
            if dir_path.is_dir():
                for ext in code_extensions:
                    files.extend(str(p) for p in dir_path.rglob(f'*{ext}') if p.is_file())
        
        # 补充项目根目录文件
        if len(files) < max_files:
            for ext in code_extensions:
                files.extend(str(p) for p in project_path.rglob(f'*{ext}') if p.is_file())
        
        # 去重并限制数量
        unique_files = list(dict.fromkeys(files))  # 保持顺序去重
        return unique_files[:max_files]

    def _assess_ai_risks(self, vulnerability_results: Dict[Dependency, List[Vulnerability]], 
                        ai_packages_by_domain: Dict[AIDomain, List[Dependency]]) -> Dict:
        """评估AI包风险"""
        results = {}
        
        # 构建领域查询映射
        domain_map = {}
        for domain, deps in ai_packages_by_domain.items():
            for dep in deps:
                domain_map[dep.name] = domain
        
        # 评估每个依赖的风险
        for dep, vulns in vulnerability_results.items():
            if not vulns:
                continue
                
            # 如果是AI包，进行专门评估
            if self.classifier.is_ai_package(dep.name):
                domain = domain_map.get(dep.name, AIDomain.OTHER)
                ai_vulns = []
                
                for vuln in vulns:
                    # 分析对AI的影响
                    vuln.ai_impact = self.risk_assessor.analyze_ai_impact(
                        vuln, dep.name, domain
                    )
                    ai_vulns.append(vuln)
                
                # 计算整体风险分数
                risk_score = sum(
                    self.risk_assessor.assess_vulnerability_risk(vuln, dep.name)
                    for vuln in vulns
                ) / max(len(vulns), 1)
                
                # 生成缓解建议
                mitigation = self.risk_assessor.generate_mitigation(vulns[0], dep.name)
                
                results[dep] = {
                    "vulnerabilities": ai_vulns,
                    "risk_score": min(risk_score, 10),
                    "domain": domain,
                    "mitigation": mitigation
                }
            else:
                # 非AI包常规处理
                results[dep] = {
                    "vulnerabilities": vulns,
                    "risk_score": sum(
                        self.risk_assessor.assess_vulnerability_risk(vuln, dep.name)
                        for vuln in vulns
                    ) / max(len(vulns), 1),
                    "domain": None,
                    "mitigation": "建议按照常规安全实践进行更新修复"
                }
        
        # 补充摘要和AI包分类信息
        results["summary"] = self._generate_summary(results)
        results["ai_packages_by_domain"] = ai_packages_by_domain
        return results

    def _generate_summary(self, audit_results: Dict) -> Dict:
        """生成审计摘要"""
        # 排除meta、summary、ai_packages_by_domain等非依赖数据
        dep_results = {k: v for k, v in audit_results.items() if isinstance(k, Dependency)}
        
        high_risk = []
        medium_risk = []
        low_risk = []
        
        for dep, info in dep_results.items():
            if info["risk_score"] >= 7:
                high_risk.append(dep)
            elif info["risk_score"] >= 4:
                medium_risk.append(dep)
            else:
                low_risk.append(dep)
        
        return {
            "total_vulnerable_packages": len(dep_results),
            "high_risk_count": len(high_risk),
            "medium_risk_count": len(medium_risk),
            "low_risk_count": len(low_risk),
            "high_risk_packages": high_risk,
            "medium_risk_packages": medium_risk,
            "low_risk_packages": low_risk
        }

    def print_ai_audit_report(self, audit_results: Dict):
        """打印AI审计报告到控制台"""
        summary = audit_results["summary"]
        vulnerabilities = {k: v for k, v in audit_results.items() if isinstance(k, Dependency)}
        
        print("\n" + "="*70)
        print("AI模型及应用包审计报告")
        print("="*70)
        print(f"审计时间: {audit_results['meta']['audit_time']}")
        print(f"项目路径: {audit_results['meta']['project_path']}")
        print(f"总依赖数: {audit_results['meta']['total_dependencies']} | AI包数: {audit_results['meta']['total_ai_packages']}")
        print(f"总览: 共发现 {summary['total_vulnerable_packages']} 个有风险的包")
        print(f"高风险: {summary['high_risk_count']} 个 | 中风险: {summary['medium_risk_count']} 个 | 低风险: {summary['low_risk_count']} 个")
        
        if not vulnerabilities:
            print("\n未发现任何漏洞")
            return
        
        # 按风险等级排序显示（前10个高风险包）
        sorted_results = sorted(
            vulnerabilities.items(),
            key=lambda x: x[1]["risk_score"],
            reverse=True
        )[:10]
        
        for i, (dep, info) in enumerate(sorted_results, 1):
            risk_level = "高" if info["risk_score"] >=7 else "中" if info["risk_score"] >=4 else "低"
            domain = info["domain"].value if info["domain"] else "非AI包"
            
            print(f"\n{i}. {dep}")
            print(f"   风险等级: {risk_level} ({info['risk_score']:.1f}/10)")
            print(f"   所属领域: {domain}")
            print(f"   漏洞数量: {len(info['vulnerabilities'])}")
            
            # 显示主要漏洞
            main_vuln = info["vulnerabilities"][0]
            print(f"   主要漏洞: {main_vuln.id}")
            print(f"   漏洞描述: {main_vuln.description[:100]}..." if len(main_vuln.description) > 100 else f"   漏洞描述: {main_vuln.description}")
            print(f"   对AI的影响: {main_vuln.ai_impact or '无特定影响'}")
            print(f"   建议措施: {info['mitigation']}")
        
        if len(vulnerabilities) > 10:
            print(f"\n... 还有 {len(vulnerabilities)-10} 个有风险的包未显示（完整信息请查看JSON报告）")

    def _generate_json_report(self, audit_results: Dict, project_path: str):
        """在项目根目录生成JSON格式审计报告"""
        # 构建JSON文件路径（项目根目录 + 配置路径）
        json_path = Path(project_path)
        summary = audit_results["summary"]
        vulnerabilities = {k: v for k, v in audit_results.items() if isinstance(k, Dependency)}
        ai_packages_by_domain = audit_results["ai_packages_by_domain"]
        meta = audit_results["meta"]
        
        # 准备JSON数据
        json_data = {
            "meta": meta,
            "summary": {
                "total_vulnerable_packages": summary["total_vulnerable_packages"],
                "high_risk_count": summary["high_risk_count"],
                "medium_risk_count": summary["medium_risk_count"],
                "low_risk_count": summary["low_risk_count"],
                "high_risk_packages": [dep.to_dict() for dep in summary["high_risk_packages"]],
                "medium_risk_packages": [dep.to_dict() for dep in summary["medium_risk_packages"]],
                "low_risk_packages": [dep.to_dict() for dep in summary["low_risk_packages"]]
            },
            "ai_packages_by_domain": {
                domain.value: [dep.to_dict() for dep in deps]
                for domain, deps in ai_packages_by_domain.items()
            },
            "vulnerable_packages": []
        }
        
        # 添加风险包详细信息（按风险等级排序）
        sorted_results = sorted(
            vulnerabilities.items(),
            key=lambda x: x[1]["risk_score"],
            reverse=True
        )
        
        for dep, info in sorted_results:
            domain_value = info["domain"].value if info["domain"] else None
            risk_level = "高" if info["risk_score"] >=7 else "中" if info["risk_score"] >=4 else "低"
            
            json_data["vulnerable_packages"].append({
                "package": dep.to_dict(),
                "risk_score": round(info["risk_score"], 1),
                "risk_level": risk_level,
                "domain": domain_value,
                "vulnerabilities": [vuln.to_dict() for vuln in info["vulnerabilities"]],
                "mitigation": info["mitigation"]
            })
        
        # 写入JSON文件
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"JSON审计报告已生成: {json_path.absolute()}")
        except Exception as e:
            print(f"生成JSON报告失败: {str(e)}")
    