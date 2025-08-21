from model.flow_matching_dir.Train_fun import train, eval


def main(model_config=None):
    modelConfig = {
        "state": "train", # or eval
        "device": "cuda:0",
        # 训练到70轮之后开始调用余弦学习率调度器
        "epoch": 1000,
        "batch_size": 32,
        "T": 100,

        "UNet_input_shape":[5, 256, 256],
        "UNet_output_shape":[1, 256, 256],
        "C_down_list": [64, 128, 256, 512],
        "C_list_attn": [64, 64, 128, 128, 128],

        "w": 2,

        "dropout": 0.15,
        "lr": 1e-3,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.03,

        "img_H":256,
        "img_W":256,

        "grad_clip": 2.,

        "log_dir":"../runs/model_log/FLOW_v1_log/",


        "save_dir": "../runs/model_pth/FLOW_v1_pth/",
        "training_load_weight": "ckpt_370_.pt",
        "test_load_weight": "ckpt_998_.pt",
        "sampled_dir": "../runs/flow_model/SampledImgs/",

        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "originalImgName": "OriginalImgs.png",
        "nrow": 8
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
