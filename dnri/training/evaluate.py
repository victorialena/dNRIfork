from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from dnri.utils import data_utils
import os
import numpy as np
import pdb

from dnri.training.train_dynamicvars import DATA_PATH
from dnri.utils.data_utils import unnormalize, ade, fde, mse, print_logs
from dnri.datasets.ind_data import ind_collate_fn


def eval_forward_prediction_dynamicvars(model, dataset, params):
    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1000)
    collate_fn = params.get('collate_fn', None)
    data_loader = DataLoader(dataset, batch_size=1, pin_memory=gpu, collate_fn=collate_fn)
    model.eval()
    total_se = 0
    batch_count = 0
    final_errors = torch.zeros(0)
    final_counts = torch.zeros(0)
    bad_count = 0

    mse_eval = []
    fde_eval = []
    ade_eval = []

    min_feats, max_feats = torch.load(os.path.join(DATA_PATH, 'train_data_stats'))

    for batch_ind, batch in enumerate(data_loader):
        inputs = batch['inputs']
        gt_preds = inputs[0, 1:]
        masks = batch['masks']
        node_inds = batch.get('node_inds', None)
        graph_info = batch.get('graph_info', None)
        burn_in_masks = batch['burn_in_masks']
        pred_masks = (masks.float() - burn_in_masks)[0, 1:]
        with torch.no_grad():
            if gpu:
                inputs = inputs.cuda(non_blocking=True)
                masks = masks.cuda(non_blocking=True)
                burn_in_masks = burn_in_masks.cuda(non_blocking=True)
            model_preds = model.predict_future(inputs, masks, node_inds, graph_info, burn_in_masks)
            
            # ------------- NEW CODE
            # pdb.set_trace()
            _output, _target = model_preds[..., :2], inputs[:, 1:, :, :2]
            _masks = ((masks[:, :-1] == 1)*(masks[:, 1:] == 1)).float()

            _output[..., 0] = unnormalize(_output[..., 0], max_feats[0], min_feats[0])
            _output[..., 1] = unnormalize(_output[..., 1], max_feats[1], min_feats[1])
            _target[..., 0] = unnormalize(_target[..., 0], max_feats[0], min_feats[0])
            _target[..., 1] = unnormalize(_target[..., 1], max_feats[1], min_feats[1])

            mse_eval.append(mse(_output, _target, _masks).item())
            ade_eval.append(ade(_output, _target, _masks).item())
            fde_eval.append(fde(_output, _target, _masks).item())

            print_logs('test', [0], [0], mse_eval, ade_eval, fde_eval)
            # -------------
            
    #         max_len = pred_masks.sum(dim=0).max().int().item()

    #         if max_len > len(final_errors):
    #             final_errors = torch.cat([final_errors, torch.zeros(max_len - len(final_errors))])
    #             final_counts = torch.cat([final_counts, torch.zeros(max_len - len(final_counts))])
    #         for var in range(masks.size(-1)):
    #             var_gt = gt_preds[:, var]
    #             var_preds = model_preds[:, var]
    #             var_pred_masks = pred_masks[:, var]
    #             var_losses = F.mse_loss(var_preds, var_gt, reduction='none').mean(dim=-1)*var_pred_masks
    #             tmp_inds = torch.nonzero(var_pred_masks)
    #             if len(tmp_inds) == 0:
    #                 continue
    #             for i in range(len(tmp_inds)-1):
    #                 if tmp_inds[i+1] - tmp_inds[i] != 1:
    #                     bad_count += 1
    #                     break
    #             num_entries = var_pred_masks.sum().int().item()
    #             final_errors[:num_entries] += var_losses[tmp_inds[0].item():tmp_inds[0].item()+num_entries]
    #             final_counts[:num_entries] += var_pred_masks[tmp_inds[0]:tmp_inds[0]+num_entries]
    # print("FINAL BAD COUNT: ",bad_count)
    # return final_errors/final_counts, final_counts