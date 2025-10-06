#!/bin/bash

# 获取当前时间戳
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "📓 Starting notebook backup process..."
echo "⏰ Timestamp: $TIMESTAMP"

# 定义文件路径
SOURCE_NOTEBOOK="notebooks/SvgDiffusion.ipynb"
BACKUP_NOTEBOOK="notebooks/SvgDiffusion_bak.ipynb"

# 检查源文件是否存在
if [ ! -f "$SOURCE_NOTEBOOK" ]; then
    echo "❌ Source notebook not found: $SOURCE_NOTEBOOK"
    exit 1
fi

# 备份操作
echo "📋 Copying $SOURCE_NOTEBOOK to $BACKUP_NOTEBOOK..."
cp "$SOURCE_NOTEBOOK" "$BACKUP_NOTEBOOK"

# 检查备份是否成功
if [ $? -eq 0 ]; then
    echo "✅ Backup completed successfully!"
    echo "📊 Summary:"
    echo "   - Source: $SOURCE_NOTEBOOK"
    echo "   - Backup: $BACKUP_NOTEBOOK"
    echo "   - Time: $TIMESTAMP"
    
    # 显示文件大小信息
    SOURCE_SIZE=$(ls -lh "$SOURCE_NOTEBOOK" | awk '{print $5}')
    BACKUP_SIZE=$(ls -lh "$BACKUP_NOTEBOOK" | awk '{print $5}')
    echo "   - Source size: $SOURCE_SIZE"
    echo "   - Backup size: $BACKUP_SIZE"
else
    echo "❌ Backup failed!"
    exit 1
fi

echo "🎯 Ready for git commit!"
