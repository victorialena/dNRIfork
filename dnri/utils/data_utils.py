import numpy as np
import torch
import sys, pdb


# Code from NRI.
def normalize(data, data_max, data_min):
	return (data - data_min) * 2 / (data_max - data_min) - 1


def unnormalize(data, data_max, data_min):
	return (data + 1) * (data_max - data_min) / 2. + data_min


def get_edge_inds(num_vars):
	edges = []
	for i in range(num_vars):
		for j in range(num_vars):
			if i == j:
				continue
			edges.append([i, j])
	return edges

def print_logs(suffix, nll_log, kl_log, mse_log, ade_log, fde_log, outfile=sys.stdout):
    print('nll_'+suffix+': {:.10f}'.format(np.mean(nll_log)),
          'kl_'+suffix+': {:.10f}'.format(np.mean(kl_log)),
          'mse_'+suffix+': {:.10f}'.format(np.mean(mse_log)),
          'ade_'+suffix+': {:.10f}'.format(np.min(ade_log)),
          'fde_'+suffix+': {:.10f}'.format(np.min(fde_log)),
          file=outfile)


def mse(recons, target, mask):
    assert target.dim() == 4
    mse = torch.pow(target-recons, 2).sum(-1) * mask
    return mse.sum() / mask.sum()


def ade(recons, target, mask):
    """ ADE = 1/T * sum(sqrt(dx_i^2 + dy_i^2))
        |target| = [bs, T, n_vars, d]
    """
    assert target.dim() == 4
    ade = torch.pow(target-recons, 2).sum(-1).sqrt() * mask
    return ade.sum() / mask.sum()


def fde(recons, target, mask):
    """ FDE = sqrt(dx_T^2 + dy_T^2)
        |target| = [bs, T, n_vars, d]
    """
    assert target.dim() == 4
    fde = torch.pow(target[:, -1]-recons[:, -1], 2).sum(-1).sqrt() * mask[:, -1]
    return fde.sum() / mask[:, -1].sum()