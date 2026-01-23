import torch
from model.UVM.UVM_model import UVMNet
import os
from model.radioUnet.RadioUnetModel import RadioWNet
from model.rem_gan import modules
from model.rem_gan.EncoderModels import ResnetGenerator, Discriminator
from model.sigle_Unet.SAUnet import SAUnet
from model.sigle_Unet.SAUnet_NoSA import SAUnetNoSA
from model.sigle_Unet.SAUnet_nofrac import SAUnet_nofrac
from model.sigle_Unet.SAUnetNoMultiScale import SAUnetNoMultiScale
from model_train.data_config import get_cars_load, get_nocars_load
from compare.compare_utils import model_compare,model_ablation

def get_SAUNet_nosa_model(base_dir,load_epoch):
    input_shape = [6, 256, 256]
    output_shape = [1, 256, 256]
    C_down_list = [64, 128, 256, 512]
    model = SAUnetNoSA(input_shape = input_shape,output_shape= output_shape,C_down_list=C_down_list)
    load_epoch = load_epoch
    SAUNet_save_dir = base_dir + r"/model_pth/SAUnet_nosa_cars/"

    SAUNet_checkpoint_path = os.path.join(SAUNet_save_dir, f"checkpoint_epoch_{load_epoch}.pth")
    SAUNet_checkpoint = torch.load(SAUNet_checkpoint_path, weights_only=True, map_location=device)
    print(f"加载历史数据load_epoch:{load_epoch}成功")
    model.load_state_dict(SAUNet_checkpoint['model_state_dict'])

    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model


def get_SAUNet_nofrac_model(base_dir,load_epoch):
    input_shape = [6, 256, 256]
    output_shape = [1, 256, 256]
    C_down_list = [64, 128, 256, 512]
    model = SAUnet_nofrac(input_shape = input_shape,output_shape= output_shape,C_down_list=C_down_list)
    load_epoch = load_epoch
    SAUNet_save_dir = base_dir + r"/model_pth/SAUnet_nofrac_cars/"
    SAUNet_checkpoint_path = os.path.join(SAUNet_save_dir, f"checkpoint_epoch_{load_epoch}.pth")
    SAUNet_checkpoint = torch.load(SAUNet_checkpoint_path, weights_only=True, map_location=device)
    print(f"加载历史数据load_epoch:{load_epoch}成功")
    model.load_state_dict(SAUNet_checkpoint['model_state_dict'])

    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model


def get_SAUNet_noMultiScale_model(base_dir,load_epoch):
    input_shape = [6, 256, 256]
    output_shape = [1, 256, 256]
    C_down_list = [64, 128, 256, 512]
    model = SAUnetNoMultiScale(input_shape = input_shape,output_shape= output_shape,C_down_list=C_down_list)
    load_epoch = load_epoch
    SAUNet_save_dir = base_dir + r"/model_pth/SAUnetNoMultiScale_cars/"
    SAUNet_checkpoint_path = os.path.join(SAUNet_save_dir, f"checkpoint_epoch_{load_epoch}.pth")
    SAUNet_checkpoint = torch.load(SAUNet_checkpoint_path, weights_only=True, map_location=device)
    print(f"加载历史数据load_epoch:{load_epoch}成功")
    model.load_state_dict(SAUNet_checkpoint['model_state_dict'])

    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model


def get_SAUNet_noStatistic_model(base_dir,load_epoch):
    input_shape = [4, 256, 256]
    output_shape = [1, 256, 256]
    C_down_list = [64, 128, 256, 512]
    model = SAUnet(input_shape = input_shape,output_shape= output_shape,C_down_list=C_down_list)
    load_epoch = load_epoch
    SAUNet_save_dir = base_dir + r"/model_pth/SAUnetNoSatistic_cars/"

    SAUNet_checkpoint_path = os.path.join(SAUNet_save_dir, f"checkpoint_epoch_{load_epoch}.pth")
    SAUNet_checkpoint = torch.load(SAUNet_checkpoint_path, weights_only=True, map_location=device)
    print(f"加载历史数据load_epoch:{load_epoch}成功")
    model.load_state_dict(SAUNet_checkpoint['model_state_dict'])

    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model


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

    train_loader, val_loader, test_loader = get_cars_load()
    base_dir = r"/home/code/radioMap/runs"
    compare_dir = base_dir + r"/model_val_log/compare_ablation_cars/"
    SAUNet_model = get_SAUNet_model(base_dir, 46)
    SAUNet_nosa = get_SAUNet_nosa_model(base_dir, 20)
    SAUNet_nofrac = get_SAUNet_nofrac_model(base_dir, 20)
    SAUNet_noMultiScale = get_SAUNet_noMultiScale_model(base_dir, 20)
    SAUNet_noStatistic = get_SAUNet_noStatistic_model(base_dir, 20)

    models_dict = {
        "SAUNet": SAUNet_model,
        "SAUNet_nopa": SAUNet_nosa,
        "SAUNet_nofrac": SAUNet_nofrac,
        "SAUNet_noMultiScale": SAUNet_noMultiScale,
        "SAUNet_noStatistic": SAUNet_noStatistic,

    }

    # models_dict = {
    #     "REM_GAN": REMGAN_model
    # }

    avg_metrics = model_ablation(models_dict, compare_dir, val_loader, device)