import torch
from model.UVM.UVM_model import UVMNet
import os
from model.radioUnet.RadioUnetModel import RadioWNet
from model.rem_gan import modules
from model.rem_gan.EncoderModels import ResnetGenerator, Discriminator
from model.sigle_Unet.SAUnet import SAUnet
from model_train.data_config import get_cars_load, get_nocars_load
from compare.compare_utils import model_compare

def get_radioUnet_model(base_dir,load_epoch):
    input_channels = 6
    WNetPhase = "secondU"
    radioUnet_model = RadioWNet(inputs=input_channels, phase=WNetPhase)
    radioUnet_model.to(device)
    radioUnet_model.eval()
    radioUnet_load_epoch = load_epoch
    radioUnet_save_dir = base_dir + r"/model_pth/RadioUnet/"  # 模型存储位置
    radioUnet_checkpoint_path = os.path.join(radioUnet_save_dir,
                                             f"checkpoint_{WNetPhase}_epoch_{radioUnet_load_epoch}.pth")
    radioUnet_checkpoint = torch.load(radioUnet_checkpoint_path, weights_only=True, map_location=device)

    radioUnet_model.load_state_dict(radioUnet_checkpoint['model_state_dict'])
    print(f"radioUnet加载历史数据load_epoch:{radioUnet_load_epoch}成功")
    return radioUnet_model

def get_UVM_model(base_dir,load_epoch):
    input_channels = 6
    UVM_model = UVMNet(n_channels=input_channels)
    UVM_model.to(device)
    UVM_model.eval()
    UVM_load_epoch = load_epoch
    UVM_save_dir = base_dir + r"/model_pth/UVM_cars/"
    UVM_checkpoint_path = os.path.join(UVM_save_dir, f"checkpoint_epoch_{UVM_load_epoch}.pth")
    UVM_checkpoint = torch.load(UVM_checkpoint_path, weights_only=True, map_location=device)

    UVM_model.load_state_dict(UVM_checkpoint['model_state_dict'])
    print(f"UVM 加载历史数据load_epoch:{UVM_load_epoch}成功")
    return UVM_model

def get_REM_model(base_dir,load_epoch):
    input_channels = 6
    REMGAN_netG = modules.RadioWNet(inputs=input_channels, phase="firstU")
    REMGAN_netD = Discriminator()
    REMGAN_load_epoch = load_epoch
    REMGAN_netG.to(device)
    REMGAN_netD.to(device)
    REMGAN_save_dir = base_dir + r"/model_pth/REM_GAN/"
    # 加载最佳检查点
    best_checkpoint_path = os.path.join(REMGAN_save_dir, f"checkpoint_REMGAN_epoch_{REMGAN_load_epoch}.pth")
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        REMGAN_netG.load_state_dict(checkpoint['netG_state_dict'])
        REMGAN_netD.load_state_dict(checkpoint['netD_state_dict'])
        print(
            f"Loaded best model from epoch {checkpoint['epoch']} with validation loss: {checkpoint['best_loss']:.6f}")
    else:
        print("Warning: Best checkpoint not found. Using current model weights.")

    # 设置模型为评估模式
    REMGAN_netG.eval()
    REMGAN_netD.eval()
    return REMGAN_netG

def get_SAUNet_model(base_dir,load_epoch):
    input_shape = [6, 256, 256]
    output_shape = [1, 256, 256]
    C_down_list = [64, 128, 256, 512]
    SAUNet_model = SAUnet(input_shape = input_shape,output_shape= output_shape,C_down_list=C_down_list)
    load_epoch = load_epoch
    SAUNet_save_dir = base_dir + r"/model_pth/SAUnet_cars/"

    SAUNet_checkpoint_path = os.path.join(SAUNet_save_dir, f"checkpoint_epoch_{load_epoch}.pth")
    SAUNet_checkpoint = torch.load(SAUNet_checkpoint_path, weights_only=True, map_location=device)
    print(f"加载历史数据load_epoch:{load_epoch}成功")
    SAUNet_model.load_state_dict(SAUNet_checkpoint['model_state_dict'])

    SAUNet_model.to(device)
    SAUNet_model.eval()  # Set model to evaluation mode
    return SAUNet_model

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = get_cars_load(noise_sigma = 4)
    base_dir = r"/home/code/radioMap/runs"
    compare_dir = base_dir + r"/model_val_log/compare_cars/"

    radioUnet_model = get_radioUnet_model(base_dir, 95)
    UVM_model = get_UVM_model(base_dir, 19)
    rem_base_dir = r"/home/code/radio_map_construction/runs"
    REMGAN_model = get_REM_model(rem_base_dir, 160)
    SAUNet_model = get_SAUNet_model(base_dir, 58)
    models_dict = {
        "PAUNet": SAUNet_model,
        "radioUnet": radioUnet_model,
        "UVM": UVM_model,
        "REM_GAN": REMGAN_model,


    }

    avg_metrics = model_compare(models_dict, compare_dir, val_loader, device,cars_flag= True)