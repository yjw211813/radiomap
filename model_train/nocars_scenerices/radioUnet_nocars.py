from model_app.radioUnetAPP import RadioWNet_app
import torch
import os
from model.radioUnet.RadioUnetModel import RadioWNet
from model_train.data_config import get_cars_load

if __name__ == '__main__':
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    base_dir = r"/home/code/radioMap/runs/"

    train_loader, val_loader, test_loader = get_cars_load()


    log_dir = base_dir + r'model_log/RadioUnet/'  # log 存储位置
    model_load_dir = base_dir + r"model_pth/RadioUnet/"  # 模型加载目录
    model_save_dir = base_dir + r"model_pth/RadioUnet/"  # 模型存储位置

    os.makedirs(model_save_dir, exist_ok=True)

    # 定义训练对象
    step_size = 30
    total_epoch = 90
    start_epoch = 0
    app = RadioWNet_app(start_epoch, log_dir, step_size, model_save_dir, device)

    print("RadioUnet")
    # 定义模型
    WNetPhase = "firstU"
    model = RadioWNet(inputs=6,phase=WNetPhase)
    app.train(model, train_loader, val_loader, total_epoch,WNetPhase = "firstU")

    WNetPhase = "secondU"
    model = RadioWNet(inputs=6,phase=WNetPhase)
    # 加载第一层最佳权重
    model, best_loss, epoch = app.load_best_checkpoint(model = model,WNetPhase = "firstU")
    print("best_loss = ",best_loss)
    print("epoch = ",epoch)
    app.train(model, train_loader, val_loader, total_epoch, WNetPhase=WNetPhase)
    # WNetPhase = "secondU"
    # model = RadioWNet(inputs=2, phase=WNetPhase)
    # load_epoch = 0
    # val_dir = r"/home/code/radio_map_construction/runs/model_val_log/RadioUnet/"
    # app.predict(model, load_epoch, test_loader, val_dir, WNetPhase="secondU", targetType="dense")
