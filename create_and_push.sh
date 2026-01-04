#!/bin/bash
# GitHub仓库创建和推送脚本

set -e

echo "=========================================="
echo "GitHub仓库创建和推送"
echo "=========================================="
echo ""

# 检查是否在正确的目录
if [ ! -f "README.md" ]; then
    echo "错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 检查Git状态
echo "[1] 检查Git状态..."
if ! git status > /dev/null 2>&1; then
    echo "错误: 不是Git仓库"
    exit 1
fi

echo "✅ Git仓库正常"
echo ""

# 显示当前提交
echo "[2] 当前提交:"
git log --oneline -3
echo ""

# 检查远程仓库
echo "[3] 检查远程仓库..."
if git ls-remote origin > /dev/null 2>&1; then
    echo "✅ 远程仓库存在"
    echo ""
    echo "[4] 推送到GitHub..."
    git push -u origin main
    echo ""
    echo "✅ 推送成功！"
    echo "访问: https://github.com/zhengbrody/RL"
else
    echo "❌ 远程仓库不存在"
    echo ""
    echo "请先创建GitHub仓库:"
    echo "1. 访问: https://github.com/new"
    echo "2. Repository name: RL"
    echo "3. 不要初始化README"
    echo "4. 点击 Create repository"
    echo ""
    echo "然后运行: git push -u origin main"
fi

