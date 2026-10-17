# env_check.py
import torch

def check_pytorch_env():
    print("==== PyTorch 环境检测 ====")
    print(f"PyTorch 版本：{torch.__version__}")
    print(f"CUDA 是否可用：{torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU 数量：{torch.cuda.device_count()}")
        print(f"当前显卡名称：{torch.cuda.get_device_name(0)}")
    else:
        print("当前使用 CPU 版本")

    test_tensor = torch.tensor([1, 2, 3])
    print(f"\n张量测试成功，张量：{test_tensor}")

if __name__ == "__main__":
    check_pytorch_env()