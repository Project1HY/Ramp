import torch


def torch_stabilized_log_sum_exp(x):
    max_x = torch.max(x, dim=2)[0]
    x = x - max_x.unsqueeze(2)
    ret = torch.log(torch.sum(torch.exp(x), dim=2)) + max_x
    return ret
