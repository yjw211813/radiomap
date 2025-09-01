import torch
# from model.sub_block.top_conv import SK_Channel_atten2D
import time
class gpu_statistic():
    def __init__(self,device):
        self.device = device

    def print_gpu_memory(self,description,x,model,temb = None):
        print(description)
        # 清空GPU缓存并记录初始显存
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(self.device) / 1024 ** 2  # MB
        print(f"初始显存占用: {initial_memory:.2f} MB")
        if temb is None:
            x = x.to(self.device)
        else:
            x = x.to(self.device)
            temb = temb.to(self.device)
        input_memory = torch.cuda.memory_allocated(self.device) / 1024 ** 2 - initial_memory
        print(f"输入张量显存占用: {input_memory:.2f} MB")
        model.to(self.device)
        model_memory = torch.cuda.memory_allocated(self.device) / 1024 ** 2 - initial_memory - input_memory
        print(f"模型参数显存占用: {model_memory:.2f} MB")
        # 前向传播
        start_time = time.time()
        if temb is None:
            output = model(x)
        else:
            output = model(x, temb)
        forward_time = time.time() - start_time
        print(f"前向传播时间: {forward_time:.4f} 秒")

        forward_memory = torch.cuda.memory_allocated(self.device) / 1024 ** 2 - initial_memory - input_memory - model_memory
        print(f"前向传播中间变量显存占用: {forward_memory:.2f} MB")
        # 统计信息
        total_memory = torch.cuda.memory_allocated(self.device) / 1024 ** 2
        print(f"总显存占用: {total_memory:.2f} MB")

        # 峰值显存使用
        peak_memory = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
        print(f"峰值显存使用: {peak_memory:.2f} MB")

        print("Input shape:", x.shape)
        print("Output shape:", output.shape)

# if __name__ == '__main__':
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     get_gpu_info = gpu_statistic(device)
#     # 测试参数
#     img_size = 256
#     C_in = 4
#     C_out = 64
#     kernel_sizes = [3, 5, 7, 9]
#     dilated_list = [1, 1, 1, 1]
#
#     x = torch.randn(16, C_in, img_size, img_size)
#     model = SK_Channel_atten2D(img_size, C_in, C_out, kernel_sizes, dilated_list)
#     get_gpu_info.print_gpu_memory("SK_Channel_atten2D GPU info",x,model)






