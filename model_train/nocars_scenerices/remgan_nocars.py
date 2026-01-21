import torch
from model.rem_gan.EncoderModels import Discriminator
import os
from model.rem_gan import modules
from model_app.RemGANAPP import REM_GAN_app
from model_train.data_config import get_nocars_load, train_batch_size, val_batch_size, test_batch_size


if __name__ == '__main__':
    ########################
    # Load dataset         #
    ########################
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    base_dir = r"/home/code/radioMap/runs/"

    train_loader, val_loader, test_loader =  get_nocars_load()

    log_dir = base_dir + r'model_log/REM_GAN_no_cars/'
    model_load_dir = base_dir + r"model_pth/REM_GAN_no_cars/"
    model_save_dir = base_dir + r"model_pth/REM_GAN_no_cars/"
    test_dir = base_dir + r"test_results/REM_GAN_no_cars/"

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    print("REM_GAN")
    netG = modules.RadioWNet(inputs=5,phase="firstU")
    netD = Discriminator()
    # 配置模型应用字典
    model_app_dict = {
        'start_epoch': 33,
        'device': device,
        'save_dir': model_save_dir,
        'log_dir': log_dir,
        'load_dir': model_load_dir,
        'test_dir': test_dir,
        'phase': 'first',
        'total_epoch': 400,
        'netG': netG,
        'netD': netD,
        'learning_rate': 0.001,
        'Gscheduler_step_size': 25,
        'Dscheduler_step_size': 25,
        'slic_block_num': 100,
        'train_batch_size': train_batch_size,
        'val_batch_size': val_batch_size,
        'test_batch_size': test_batch_size
    }

    # 配置数据集字典
    dataSetDict = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
    }
    # 初始化REM_GAN应用
    rem_gan_app = REM_GAN_app(model_app_dict, dataSetDict)

    # 训练模型
    print("开始训练REM-GAN模型...")
    rem_gan_app.train()
    # 测试模型
    print("开始测试REM-GAN模型...")
    rem_gan_app.test()
    print("REM-GAN 训练和测试完成")

