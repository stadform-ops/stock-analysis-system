#!/usr/bin/env python3
"""
环境体检脚本（标准库实现）
- 不依赖 numpy/pandas/torch，可在最小 Python 环境运行
- 检查 Python 版本、关键依赖可用性、并给出安装建议
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class CheckResult:
    name: str
    installed: bool
    detail: str


REQUIRED_MODULES: Dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "tqdm": "tqdm",
    "torch": "torch",
    "torch_geometric": "torch-geometric",
    "baostock": "baostock",
}


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def check_modules() -> List[CheckResult]:
    results: List[CheckResult] = []
    for module_name, package_name in REQUIRED_MODULES.items():
        installed = has_module(module_name)
        detail = f"pip/conda 包名: {package_name}"
        results.append(CheckResult(module_name, installed, detail))
    return results


def python_advice() -> Tuple[bool, str]:
    major, minor = sys.version_info[:2]
    if (major, minor) == (3, 7):
        return True, "Python 3.7 ✅（与当前 requirements 的 torch<1.9 更兼容）"
    if (major, minor) < (3, 7):
        return False, "Python 版本过低，建议升级到 3.7"
    return False, (
        f"当前 Python {major}.{minor}，项目 requirements 固定了 torch<1.9，"
        "建议在 py3.7 虚拟环境运行（如你的 pytorch_gpu）。"
    )


def print_header() -> None:
    print("=" * 80)
    print("🩺 Stock Project Environment Doctor")
    print("=" * 80)
    print(f"Python: {platform.python_version()} ({platform.python_implementation()})")
    print(f"Platform: {platform.platform()}")
    print()


def print_module_report(results: List[CheckResult]) -> None:
    print("[1] 关键依赖检测")
    for r in results:
        icon = "✅" if r.installed else "❌"
        print(f"  {icon} {r.name:<16} {r.detail}")
    print()


def print_install_guide(missing_modules: List[str]) -> None:
    print("[2] 安装建议（推荐在 conda py3.7 环境执行）")
    print("  conda create -n pytorch_gpu python=3.7 -y")
    print("  conda activate pytorch_gpu")
    print("  conda install numpy pandas scikit-learn pyyaml matplotlib seaborn tqdm -y")
    print("  pip install baostock")
    print("  # 按你的 CUDA 版本选择 PyTorch（示例为 CUDA 11.3）")
    print("  conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -y")
    print("  # PyG 与 torch 1.8.1 对齐示例")
    print("  pip install torch-geometric==2.0.3 torch-scatter==2.0.8 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1")
    if missing_modules:
        print()
        print("  当前环境缺失模块:")
        for m in missing_modules:
            print(f"    - {m}")
    print()


def main() -> int:
    print_header()

    py_ok, py_msg = python_advice()
    print("[0] Python 版本检查")
    print(("✅ " if py_ok else "⚠️  ") + py_msg)
    print()

    results = check_modules()
    print_module_report(results)

    missing = [r.name for r in results if not r.installed]
    if missing:
        print("⚠️  当前环境依赖不完整，无法直接运行 main.py。")
        print_install_guide(missing)
        return 1

    print("✅ 依赖齐全，可以尝试运行: python main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
