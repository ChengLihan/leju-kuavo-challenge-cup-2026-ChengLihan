#!/usr/bin/env bash
# ============================================================
# scene2 数据集采集脚本
# 循环 seed 0-199，每个 seed 截取 3 张头部相机中下方 640×640
# 保存到 images/scene2_seed{seed}_c{1,2,3}.jpg
#
# 用法（容器内）：
#   cd /root/kuavo_ws
#   source devel/setup.zsh          # 或 setup.bash
#   bash src/challenge_cup_task_template/scripts/collect_dataset_scene2.sh
# ============================================================

set -euo pipefail

TOTAL=200          # seed 0 ~ 199
TIMEOUT=120        # 每个 seed 仿真启动超时（秒）
START_SEED=0
END_SEED=199
FAILED_SEEDS=()
SUCCESS_COUNT=0
FAIL_COUNT=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SCRIPT_DIR = .../src/challenge_cup_task_template/scripts
# WS_ROOT    = .../ (仓库根目录)
WS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
IMAGES_DIR="$WS_ROOT/images"

mkdir -p "$IMAGES_DIR"

echo "============================================"
echo " scene2 数据集采集"
echo " seed 范围: $START_SEED ~ $END_SEED (共 $TOTAL 个)"
echo " 每个 seed 截取 3 张 640×640"
echo " 保存目录: $IMAGES_DIR"
echo "============================================"
echo ""

for ((seed = START_SEED; seed <= END_SEED; seed++)); do
    echo "----------------------------------------"
    echo "[$(date '+%H:%M:%S')] 正在采集 seed=$seed ..."
    echo "----------------------------------------"

    set +e
    rosrun challenge_cup_task_template challenge_task.py \
        --scene scene2 \
        --seed "$seed" \
        --no-timer-gui \
        --timeout "$TIMEOUT" \
        --node-name "dataset_scene2_s${seed}"

    exit_code=$?
    set -e

    if [ $exit_code -eq 0 ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "[$(date '+%H:%M:%S')] seed=$seed 完成 ✓"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_SEEDS+=("$seed")
        echo "[$(date '+%H:%M:%S')] seed=$seed 失败 (exit=$exit_code) ✗"
    fi

    # 等待仿真进程完全退出再启动下一个
    sleep 2

    # 进度报告
    echo "  进度: $((seed - START_SEED + 1))/$TOTAL  成功=$SUCCESS_COUNT  失败=$FAIL_COUNT"
    echo ""
done

echo "============================================"
echo " 采集完成！"
echo " 成功: $SUCCESS_COUNT / $TOTAL"
if [ ${#FAILED_SEEDS[@]} -gt 0 ]; then
    echo " 失败 seed: ${FAILED_SEEDS[*]}"
else
    echo " 全部成功 ✓"
fi
echo " 图片保存位置: $IMAGES_DIR"
echo "============================================"
