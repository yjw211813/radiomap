import time
import torch
from torch import nn, Tensor
from volecity_predict import MLP
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.distributions import Independent, Normal
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='torch')

def inf_train_gen(batch_size: int = 200, device: str = "cpu"):
    x1 = torch.rand(batch_size, device=device) * 4 - 2
    x2_ = torch.rand(batch_size, device=device) - torch.randint(high=2, size=(batch_size,), device=device) * 2
    x2 = x2_ + (torch.floor(x1) % 2)

    data = 1.0 * torch.cat([x1[:, None], x2[:, None]], dim=1) / 0.45

    return data.float()

# Model wrapper class
class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)


# Training function
def train(device, lr, batch_size, iterations, print_every, hidden_dim):
    vf = MLP(input_dim=2, time_dim=1, hidden_dim=hidden_dim).to(device)
    path = AffineProbPath(scheduler=CondOTScheduler())
    optim = torch.optim.Adam(vf.parameters(), lr=lr)

    start_time = time.time()
    for i in range(iterations):
        optim.zero_grad()
        x_1 = inf_train_gen(batch_size=batch_size, device=device)
        x_0 = torch.randn_like(x_1).to(device)
        t = torch.rand(x_1.shape[0]).to(device)
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        loss = torch.pow(vf(path_sample.x_t, path_sample.t) - path_sample.dx_t, 2).mean()
        loss.backward()
        optim.step()

        if (i + 1) % print_every == 0:
            elapsed = time.time() - start_time
            print(f'| iter {i + 1:6d} | {elapsed * 1000 / print_every:5.2f} ms/step | loss {loss.item():8.3f}')
            start_time = time.time()

    return vf, path


# Inference function
def infer(model, path, device, step_size, batch_size, T, grid_size, num_acc):
    wrapped_vf = WrappedModel(model)
    solver = ODESolver(velocity_model=wrapped_vf)

    # Sampling and visualization
    x_init = torch.randn((batch_size, 2), device=device)
    sol = solver.sample(time_grid=T, x_init=x_init, method='midpoint',
                        step_size=step_size, return_intermediates=True).cpu().numpy()
    T_cpu = T.cpu()

    fig, axs = plt.subplots(1, len(T), figsize=(20, 20))
    for i, t_val in enumerate(T_cpu):
        H = axs[i].hist2d(sol[i, :, 0], sol[i, :, 1], 300, range=((-5, 5), (-5, 5)))
        cmax = torch.quantile(torch.from_numpy(H[0]), 0.99).item()
        norm = cm.colors.Normalize(vmax=cmax, vmin=0.0)
        axs[i].hist2d(sol[i, :, 0], sol[i, :, 1], 300, range=((-5, 5), (-5, 5)), norm=norm)
        axs[i].set_aspect('equal')
        axs[i].axis('off')
        axs[i].set_title(f't= {t_val:.2f}')
    plt.tight_layout()
    plt.show()

    # Likelihood computation and visualization
    x_grid = torch.stack(torch.meshgrid(torch.linspace(-5, 5, grid_size),
                                        torch.linspace(-5, 5, grid_size)), dim=-1
                         ).reshape(-1, 2).to(device)
    gaussian_log_density = Independent(Normal(torch.zeros(2, device=device),
                                              torch.ones(2, device=device)), 1).log_prob

    log_p_acc = 0
    for _ in range(num_acc):
        _, log_p = solver.compute_likelihood(
            x_1=x_grid, method='midpoint', step_size=step_size,
            exact_divergence=False, log_p0=gaussian_log_density
        )
        log_p_acc += log_p
    log_p_acc /= num_acc

    _, exact_log_p = solver.compute_likelihood(
        x_1=x_grid, method='midpoint', step_size=step_size,
        exact_divergence=True, log_p0=gaussian_log_density
    )

    likelihood = log_p_acc.exp().cpu().reshape(grid_size, grid_size).numpy()
    exact_likelihood = exact_log_p.exp().cpu().reshape(grid_size, grid_size).numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    cmax = 1 / 32
    norm = cm.colors.Normalize(vmax=cmax, vmin=0.0)
    for ax, data, title in zip(axs, [likelihood, exact_likelihood],
                               [f'Model Likelihood (Hutchinson, #acc={num_acc})',
                                'Exact Model Likelihood']):
        im = ax.imshow(data, extent=(-5, 5, -5, 5), origin='lower', cmap='viridis', norm=norm)
        ax.set_title(title)
    fig.colorbar(im, ax=axs, orientation='horizontal', label='density')
    plt.show()


if __name__ == "__main__":
    # Set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}')
    torch.manual_seed(42)

    # Training hyperparameters
    train_params = {
        'lr': 0.001,
        'batch_size': 4096,
        'iterations': 20001,
        'print_every': 2000,
        'hidden_dim': 512
    }

    # Inference hyperparameters
    infer_params = {
        'step_size': 0.05,
        'batch_size': 50000,
        'T': torch.linspace(0, 1, 10).to(device),
        'grid_size': 200,
        'num_acc': 10
    }

    # Execute training
    model, path = train(device, **train_params)

    # Execute inference
    infer(model, path, device, **infer_params)

