import torch

from torch import nn

""" 7.5.3从零实现 """
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled():  # inference
        X_hat = ((X - moving_mean) / torch.sqrt(moving_var + eps))
    else:  # train
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:  # full-connected layer
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        elif len(X.shape) == 4:     # convolution layer
            mean = X.mean(dim=(0, 2, 3), keepdim=True)    # keepdim，求平均的维度不是去掉，而是变为1
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = ((X - mean) / torch.sqrt(var + eps))
        moving_mean = momentum * moving_mean + (1 - momentum) * mean
        moving_var = momentum * moving_var + (1 - momentum) * var
    X_hat = gamma * X_hat + beta
    return X_hat, moving_mean, moving_var


class BatchNorm(nn.Module):
    def __init__(self, num_dims, num_features,):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        elif num_dims == 4:
            shape = (1, num_features, 1, 1)
        shape = (1, num_features) if num_dims == 2 else (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta, self.moving_mean, self.moving_var,
                                                            eps=1e-5, momentum=.9)
        return Y

