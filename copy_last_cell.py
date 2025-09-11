#!/usr/bin/env python3
"""
将SvgDiffusion.ipynb的最后一个cell复制到SvgDiffusion_backup.ipynb
"""

import json
import os
import shutil

def copy_last_cell():
    # 文件路径
    source_file = "notebooks/SvgDiffusion.ipynb"
    backup_file = "notebooks/SvgDiffusion_backup.ipynb"
    
    # 检查源文件是否存在
    if not os.path.exists(source_file):
        print(f"❌ Source file not found: {source_file}")
        return
    
    # 读取源notebook
    with open(source_file, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    # 检查是否有cells
    if not source_data.get('cells'):
        print("❌ No cells found in source notebook")
        return
    
    # 获取最后一个非空cell
    last_cell = None
    for i in range(len(source_data['cells']) - 1, -1, -1):
        cell = source_data['cells'][i]
        source_lines = cell.get('source', [])
        if source_lines and any(line.strip() for line in source_lines):
            last_cell = cell
            print(f"📋 Found last non-empty cell at index {i}: {cell.get('cell_type', 'unknown')} - {len(source_lines)} lines")
            break
    
    if not last_cell:
        print("❌ No non-empty cells found")
        return
    
    # 读取或创建备份文件
    if os.path.exists(backup_file):
        with open(backup_file, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
    else:
        # 创建新的notebook结构
        backup_data = {
            "cells": [],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
    
    # 添加最后一个cell到备份文件
    backup_data['cells'].append(last_cell)
    
    # 保存备份文件
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(backup_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Last cell copied to {backup_file}")
    print(f"📊 Backup notebook now has {len(backup_data['cells'])} cells")

if __name__ == "__main__":
    copy_last_cell()
