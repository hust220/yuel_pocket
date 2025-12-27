#!/usr/bin/env bash
# 下载或续传 PLINDER 数据集

set -e

# 配置版本，可按需修改
PLINDER_RELEASE=${PLINDER_RELEASE:-2024-06}
PLINDER_ITERATION=${PLINDER_ITERATION:-v2}

# 本地目标目录；可通过第一个参数或环境变量覆盖
if [ -n "${1:-}" ]; then
  TARGET_DIR="$1"
else
  TARGET_DIR=${TARGET_DIR:-./data/${PLINDER_RELEASE}/${PLINDER_ITERATION}}
fi

# 要同步的子目录，留空表示同步整个迭代目录
# 例如只拉 splits: SYNC_DIRS=("splits")
SYNC_DIRS=()

mkdir -p "$TARGET_DIR"

if [ ${#SYNC_DIRS[@]} -eq 0 ]; then
  echo "Sync full dataset to $TARGET_DIR"
  gsutil -m rsync -r "gs://plinder/${PLINDER_RELEASE}/${PLINDER_ITERATION}" "$TARGET_DIR"
else
  for d in "${SYNC_DIRS[@]}"; do
    echo "Sync $d to $TARGET_DIR/$d"
    mkdir -p "$TARGET_DIR/$d"
    gsutil -m rsync -r "gs://plinder/${PLINDER_RELEASE}/${PLINDER_ITERATION}/$d" "$TARGET_DIR/$d"
  done
fi

echo "Done"