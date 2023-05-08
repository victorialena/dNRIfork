from dnri.utils.flags import build_flags
import dnri.models.model_builder as model_builder
from dnri.datasets.ind_data import IndData, ind_collate_fn
import dnri.training.train_dynamicvars as train
import dnri.training.train_utils as train_utils
import dnri.training.evaluate as evaluate
import dnri.utils.misc as misc

# from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os, sys
import pickle
import pdb

from datetime import date

# Save model and meta-data. Always saves in a new sub-folder.
def get_save_folder(args):
    save_folder = args.working_dir    
    base_str = date.today().strftime("%m-%d-%y")+'_exp'
    iter = max([int(file.replace('_', '.').split('.')[1][3:]) for file in os.listdir(save_folder) if base_str in file], default=0)
    return save_folder+base_str+str(iter+1)+'/'


def print_mse(pretext, mse, counts, outfile=sys.stdout):
    print(pretext, file=outfile)
    print("\t1 STEP:  ", mse[0].item(), counts[0].item(), file=outfile)
    print("\t20 STEP: ", mse[19].item(), counts[19].item(), file=outfile)
    print("\t40 STEP: ", mse[39].item(), counts[39].item(), file=outfile)
    

if __name__ == '__main__':
    parser = build_flags()
    parser.add_argument('--data_path')
    parser.add_argument('--error_out_name', default='val_prediction_errors.npy')
    parser.add_argument('--train_data_len', type=int, default=-1)
    parser.add_argument('--prior_variance', type=float, default=5e-5)
    parser.add_argument('--expand_train', action='store_true')
    parser.add_argument('--final_test', action='store_true')
    parser.add_argument('--test_short_sequences', action='store_true')
    parser.add_argument('--load_folder', type=str, default="")

    args = parser.parse_args()
    params = vars(args)

    # ----------- save files
    if args.mode == 'train':
        save_folder = get_save_folder(args)
        os.mkdir(save_folder)

        meta_file = os.path.join(save_folder, 'metadata.pkl')
        pickle.dump({'args': args}, open(meta_file, "wb"))

        log_file = os.path.join(save_folder, 'log.txt')
        log = open(log_file, 'w')
        params['working_dir'] = save_folder
    # ----------- 

    misc.seed(args.seed)

    params['input_size'] = 4
    params['nll_loss_type'] = 'gaussian'
    params['dynamic_vars'] = True
    params['collate_fn'] = ind_collate_fn

    model = model_builder.build_model(params)
    
    if args.mode == 'train':
        pdb.set_trace()
        train_data = IndData(args.data_path, 'train', params)
        val_data = IndData(args.data_path, 'valid', params)
        train.train(model, train_data, val_data, params, log)
    else:
        path = os.path.join(args.working_dir, args.load_folder)
        model.load(path)
    
    test_data = IndData(args.data_path, 'test', params)
    test_mse, counts = evaluate.eval_forward_prediction_dynamicvars(model, test_data, params)

    print_mse("TEST FORWARD PRED RESULTS:", test_mse, counts)

    if args.mode == 'train':
        print_mse("TEST FORWARD PRED RESULTS:", test_mse, counts, log)
        log.flush()
    
        