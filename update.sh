#!/bin/bash

# 获取当前时间戳
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# 提交信息
COMMIT_MSG="update@$TIMESTAMP"

echo "🚀 Starting update process..."
echo "📝 Commit message: $COMMIT_MSG"

# 添加所有文件到暂存区
echo "📁 Adding all files to staging area..."
git add .

# 检查是否有变更
if git diff --staged --quiet; then
    echo "⚠️  No changes to commit"
    exit 0
fi

# 提交变更
echo "💾 Committing changes..."
git commit -m "$COMMIT_MSG"

# 推送到远程仓库
echo "🚀 Pushing to remote repository..."
git push

echo "✅ Update completed successfully!"
echo "📊 Summary:"
echo "   - Commit: $COMMIT_MSG"
echo "   - Status: Pushed to remote"
