import os
from PIL import Image
import json

# ========= 路径（main_task2 在 C 盘根目录） =========
base = r"C:\main_task2"
img_dir = os.path.join(base, "images")
comp_dir = os.path.join(base, "compressed")
res_dir = os.path.join(base, "restored")

# 创建输出文件夹（不存在自动新建）
os.makedirs(comp_dir, exist_ok=True)
os.makedirs(res_dir, exist_ok=True)

# ========= RLE 编码 =========
def rle_encode(pixel_list):
    if not pixel_list:
        return []
    result = []
    current_val = pixel_list[0]
    count = 1
    for val in pixel_list[1:]:
        if val == current_val:
            count += 1
        else:
            result.append([current_val, count])
            current_val = val
            count = 1
    result.append([current_val, count])
    return result

# ========= RLE 解码（RGB list -> tuple） =========
def rle_decode(rle_data):
    pixels = []
    for val, cnt in rle_data:
        if isinstance(val, list):
            val = tuple(val)
        pixels.extend([val] * cnt)
    return pixels

# ========= 压缩：遍历 images 所有图片，增加统计 =========
def compress_all():
    file_list = [f for f in os.listdir(img_dir) if f.lower().endswith(".ppm")]
    print(f"===== 开始压缩，找到 PPM 图像数量：{len(file_list)} =====\n")

    for fname in file_list:
        img_path = os.path.join(img_dir, fname)
        try:
            img = Image.open(img_path)
            width, height = img.size
            pixels = list(img.getdata())

            # RLE
            rle_data = rle_encode(pixels)
            save_data = {
                "width": width,
                "height": height,
                "rle": rle_data
            }

            json_name = os.path.splitext(fname)[0] + ".json"
            json_path = os.path.join(comp_dir, json_name)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f)

            # 文件大小统计（单位：字节）
            origin_size = os.path.getsize(img_path)
            compressed_size = os.path.getsize(json_path)
            compress_ratio = compressed_size / origin_size
            reduce_ratio = 1 - compress_ratio

            print(f"【压缩成功】图片：{fname}")
            print(f"    原始文件大小：{origin_size} 字节")
            print(f"    压缩后 JSON：{compressed_size} 字节")
            print(f"    文件缩减率：{reduce_ratio * 100:.2f} %\n")

        except Exception as e:
            print(f"【压缩跳过】文件 {fname}，读取异常：{str(e)}\n")

# ========= 解压 =========
def decompress_all():
    json_list = [f for f in os.listdir(comp_dir) if f.lower().endswith(".json")]
    print(f"===== 开始解压，待解压文件数量：{len(json_list)} =====\n")
    for fname in json_list:
        json_path = os.path.join(comp_dir, fname)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            w = data["width"]
            h = data["height"]
            rle_data = data["rle"]

            pixel_list = rle_decode(rle_data)
            new_img = Image.new("RGB", (w, h))
            new_img.putdata(pixel_list)

            ppm_name = os.path.splitext(fname)[0] + ".ppm"
            out_path = os.path.join(res_dir, ppm_name)
            new_img.save(out_path)
            print(f"【解压成功】{fname} -> {ppm_name}")
        except Exception as e:
            print(f"【解压异常】{json_path} 解压失败：{str(e)}")


if __name__ == "__main__":
    compress_all()
    decompress_all()
