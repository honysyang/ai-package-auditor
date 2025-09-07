"""依赖项收集器模块"""
import pkg_resources
import re
from pathlib import Path
from typing import Set

from ..exceptions import AIAuditError
from ..models.dependency import Dependency


class AIDependencyCollector:
    """AI项目依赖收集器"""
    
    def collect_installed_packages(self) -> Set[Dependency]:
        """收集已安装包，优先识别AI相关包"""
        try:
            # 使用pkg_resources收集已安装的包
            return {
                Dependency(dist.project_name, dist.version)
                for dist in pkg_resources.working_set
            }
        except Exception as e:
            raise AIAuditError(f"收集已安装包失败: {str(e)}") from e

    def collect_from_project(self, project_path: str = ".") -> Set[Dependency]:
        """从AI项目中收集依赖，包括模型配置文件"""
        dependencies = set()
        project_path_obj = Path(project_path)
        
        # 收集常规依赖文件
        dep_files = [
            "requirements.txt",
            "requirements-ai.txt",
            "pyproject.toml",
            "Pipfile"
        ]
        
        for file in dep_files:
            file_path = project_path_obj / file
            if file_path.exists():
                if file.endswith(".txt"):
                    dependencies.update(self._parse_requirements(file_path))
                elif file.endswith(".toml"):
                    dependencies.update(self._parse_pyproject(file_path))
        
        # 从AI模型配置文件中提取依赖
        ai_config_files = [
            "model_config.json",
            "config.yaml",
            "requirements-ml.txt"
        ]
        
        for file in ai_config_files:
            file_path = project_path_obj / file
            if file_path.exists():
                dependencies.update(self._parse_ai_config(file_path))
        
        return dependencies

    def _parse_requirements(self, file_path: Path) -> Set[Dependency]:
        """解析requirements文件"""
        dependencies = set()
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '==' in line:
                        name, version = line.split('==', 1)
                        dependencies.add(Dependency(name.strip(), version.strip()))
        except Exception as e:
            print(f"警告: 解析{file_path}失败 - {e}")
        return dependencies

    def _parse_pyproject(self, file_path: Path) -> Set[Dependency]:
        """解析pyproject.toml"""
        import toml  # 延迟导入，只在需要时加载
        dependencies = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                pyproject_data = toml.load(f)
            
            if 'tool' in pyproject_data and 'poetry' in pyproject_data['tool']:
                deps = pyproject_data['tool']['poetry'].get('dependencies', {})
                for name, spec in deps.items():
                    if name == 'python':
                        continue
                    if isinstance(spec, str):
                        version = spec.strip('^~>=<')
                        dependencies.add(Dependency(name, version))
        except Exception as e:
            print(f"警告: 解析{file_path}失败 - {e}")
        return dependencies

    def _parse_ai_config(self, file_path: Path) -> Set[Dependency]:
        """解析AI模型配置文件中的依赖信息"""
        dependencies = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 尝试从配置文件中提取包名和版本
                pattern = r'(tensorflow|pytorch|torch|transformers|opencv)\s*[=<>~^]+\s*[\d\.]+'
                matches = re.findall(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    parts = match.split('=')
                    if len(parts) == 2:
                        dependencies.add(Dependency(parts[0].strip().lower(), parts[1].strip()))
        except Exception as e:
            print(f"警告: 解析AI配置文件{file_path}失败 - {e}")
        return dependencies

    def collect_all(self, project_path: str = ".") -> Set[Dependency]:
        """收集所有依赖，优先处理AI相关包"""
        all_deps = self.collect_installed_packages()
        project_deps = self.collect_from_project(project_path)
        all_deps.update(project_deps)
        return all_deps
    