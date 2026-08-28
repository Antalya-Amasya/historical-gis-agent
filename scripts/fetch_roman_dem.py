import os
import requests
import zipfile
import io
import time

# 目标本地存放目录
OUTPUT_DIR = r"C:\data\srtm-hgt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 扩大后的绝对安全冗余范围 (覆盖大罗马/帕提亚/日耳曼/北非)
ROWS = ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
COLS = range(28, 41)  # 包含 28 到 40 列


def fetch_dem_tiles():
    total_grids = len(ROWS) * len(COLS)
    current = 0

    for row in ROWS:
        for col in COLS:
            current += 1
            grid_id = f"{row}{col}"
            url = f"http://viewfinderpanoramas.org/dem3/{grid_id}.zip"

            print(f"[{current}/{total_grids}] Fetching {grid_id}...")

            try:
                # 设置 20 秒超时，防止请求挂起
                response = requests.get(url, timeout=20)

                # 纯海洋或服务器不存在的区块直接跳过
                if response.status_code == 404:
                    print(f"  -> {grid_id} not found (likely ocean). Skipping.")
                    continue

                response.raise_for_status()

                # 在内存中解压，过滤并提取 .hgt 文件
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    for file_info in z.infolist():
                        if file_info.filename.endswith('.hgt'):
                            # 忽略原压缩包内的文件夹层级，直接平铺到 OUTPUT_DIR
                            file_info.filename = os.path.basename(file_info.filename)
                            z.extract(file_info, OUTPUT_DIR)

                print(f"  -> {grid_id} extracted successfully.")
                time.sleep(1)  # 礼貌性防封禁延迟

            except requests.exceptions.RequestException as e:
                print(f"  -> Network error fetching {grid_id}: {e}")
            except zipfile.BadZipFile:
                print(f"  -> Error: Downloaded file for {grid_id} is not a valid ZIP.")
            except Exception as e:
                print(f"  -> Unexpected error with {grid_id}: {e}")


if __name__ == "__main__":
    fetch_dem_tiles()
    print(f"\nFinished! All .hgt files are now saved in {OUTPUT_DIR}")