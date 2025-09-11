#!/usr/bin/env python3
"""
测试代码跳转功能
在这个文件中测试能否跳转到标准库源码
"""

import os  # 应该能跳转到 /Users/garygao/miniconda/lib/python3.13/os.py
import sys  # 内置模块，没有 __file__，但应该有智能提示
import subprocess  # 应该能跳转到标准库
from pathlib import Path  # 应该能跳转到标准库

def test_navigation():
    """测试各种导航功能"""
    # 测试 os 模块
    current_dir = os.getcwd()  # os.getcwd 应该能跳转
    env_path = os.environ.get('PATH')  # os.environ 应该能跳转
    
    # 测试 sys 模块
    python_path = sys.executable  # sys.executable 应该有提示
    version = sys.version  # sys.version 应该有提示
    
    # 测试 subprocess 模块
    result = subprocess.run(['echo', 'hello'], capture_output=True)  # subprocess.run 应该能跳转
    
    # 测试 pathlib 模块
    path = Path(__file__)  # Path 应该能跳转到类定义
    parent = path.parent  # .parent 应该有属性提示
    
    print("所有测试完成")

if __name__ == "__main__":
    test_navigation()
