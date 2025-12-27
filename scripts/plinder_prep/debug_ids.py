
import pyarrow.parquet as pq
import sys
import os

split_path = "/home/tyq4zn/scratch/codes/yuel_pocket/data/plinder/data/2024-06/v2/splits/split.parquet"
if not os.path.exists(split_path):
    print("Split file not found")
    sys.exit(1)

table = pq.read_table(split_path, columns=['system_id'])
ids = table.to_pandas()['system_id'].head(5).tolist()
print("Sample IDs:", ids)

for sid in ids:
    print(f"ID: {sid}, Bucket: {sid[1:3]}")
