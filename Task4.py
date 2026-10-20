from PIL import Image
import os

def batch_to_gray(input_dir, output_dir):
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历文件夹里的图片
    support_suffix = (".jpg", ".jpeg", ".png", ".bmp")
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(support_suffix):
            img_path = os.path.join(input_dir, filename)
            try:
                img = Image.open(img_path)
                gray_img = img.convert("L")   # 转为灰度图
                save_path = os.path.join(output_dir, filename)
                gray_img.save(save_path)
                print(f"已处理：{filename}")
            except Exception as e:
                print(f"处理失败 {filename}，错误：{e}")

if __name__ == "__main__":
    # 输入文件夹、输出文件夹，可自行修改
    input_folder = "./input_img"
    output_folder = "./gray_output"
    batch_to_gray(input_folder, output_folder)
    print("\n==== 全部任务结束 ====")


