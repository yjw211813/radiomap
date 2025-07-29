
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.diffusion_model.noisePreCondition import noise_UNet
import numpy as np

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.reshape([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, condition_info):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t =   extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        loss = F.mse_loss(self.model(x_t, t, condition_info), noise, reduction='none')
        return loss

def GaussianDiffusionTrainer_test():

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = 8
    T = 1000
    img_H = 256
    img_W = 256
    beta_1 = 1e-4
    beta_T = 0.028

    condition_info = torch.randn(batch_size, 4, img_H, img_W).to(device)
    x_0 = torch.randn(batch_size, 1, img_H, img_W).to(device)
    t = torch.randint(1000, size=[batch_size]).to(device)
    BTM_ghost_UNet_input_shape = [condition_info.shape[1] + 1, condition_info.shape[2], condition_info.shape[3]]
    BTM_ghost_UNet_output_shape = [1, condition_info.shape[2], condition_info.shape[3]]
    C_down_list = [64, 128, 256, 512]
    C_list_attn = torch.tensor([64, 64, 128, 128, 128])
    attn_params = [C_list_attn, C_list_attn // 2, C_list_attn // 2, C_list_attn // 2]

    net_model = noise_UNet(T, BTM_ghost_UNet_input_shape, BTM_ghost_UNet_output_shape, C_down_list, attn_params).to(device)
    trainer = GaussianDiffusionTrainer(net_model, beta_1, beta_T, T).to(device)
    output = trainer(x_0, condition_info).sum() / batch_size ** 2.
    print(output)






class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w = 0.):
        super().__init__()

        self.model = model
        self.T = T
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def batch_minmax_normalize(self,x):
        """输入形状: (B, C, H, W)"""
        # 计算每张图片的最小值和最大值（保持维度以便广播）
        min_vals = x.view(x.size(0), -1).min(dim=1)[0]  # shape: (B,)
        max_vals = x.view(x.size(0), -1).max(dim=1)[0]  # shape: (B,)

        # 扩展维度以便广播 [B,] -> [B,1,1,1]
        min_vals = min_vals[:, None, None, None]
        max_vals = max_vals[:, None, None, None]

        # 归一化到 [0, 1]
        normalized = (x - min_vals) / (max_vals - min_vals + 1e-10)  # 避免除零
        return normalized

    def p_mean_variance(self, x_t, t, condition_info):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps = self.model(x_t, t, condition_info)
        nonEps = self.model(x_t, t, torch.zeros_like(condition_info).to(condition_info.device))
        eps = (1. + self.w) * eps - self.w * nonEps
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T, condition_info):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t, condition_info=condition_info)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return self.batch_minmax_normalize(x_0)

def GaussianDiffusionSamplerTest():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = 32
    T = 1000
    img_H = 256
    img_W = 256
    beta_1 = 1e-4
    beta_T = 0.028
    w = 1.8
    condition_info = torch.randn(batch_size, 4, img_H, img_W).to(device)
    x_0 = torch.randn(batch_size, 1, img_H, img_W).to(device)
    t = torch.randint(1000, size=[batch_size]).to(device)
    BTM_ghost_UNet_input_shape = [condition_info.shape[1] + 1, condition_info.shape[2], condition_info.shape[3]]
    BTM_ghost_UNet_output_shape = [1, condition_info.shape[2], condition_info.shape[3]]
    C_down_list = [64, 128, 256, 512]
    C_list_attn = torch.tensor([64, 64, 128, 128, 128])
    attn_params = [C_list_attn, C_list_attn // 2, C_list_attn // 2, C_list_attn // 2]

    net_model = noise_UNet(T, BTM_ghost_UNet_input_shape, BTM_ghost_UNet_output_shape, C_down_list, attn_params).to(device)
    sampler = GaussianDiffusionSampler(model = net_model,beta_1 =  beta_1,beta_T =  beta_T,T = T, w=w).to(device)

    noisyImage = torch.randn(size=(batch_size, 1, img_H, img_W), device=device)
    # saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
    net_model.eval()
    with torch.no_grad():
        sampledImgs = sampler(noisyImage, condition_info)

    # sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
    print(sampledImgs.shape)


if __name__ == '__main__':

    # GaussianDiffusionTrainer_test()
    GaussianDiffusionSamplerTest()