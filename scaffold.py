# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: fed_avg_algo.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2020-01-13 (YYYY-MM-DD)
-----------------------------------------------
"""
import numpy as np
import h5py
import tqdm
import copy
import torch
import pandas as pd
import sys
import random
import matplotlib
import matplotlib.pyplot as plt

sys.path.append('../')
# DualFedAtt.
from utils.misc import args_parser, average_weights
from utils.misc import get_data, process_isolated
from utils.models import LSTM
from utils.fed_update import LocalUpdate, test_inference
from sklearn import metrics

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data, _, selected_cells, mean, std, _, _ = get_data(args)

    device = 'cuda' if args.gpu else 'cpu'

    parameter_list = 'FedAvg-data-{:}-type-{:}-'.format(args.file, args.type)
    parameter_list += '-frac-{:.2f}-le-{:}-lb-{:}-seed-{:}'.format(args.frac, args.local_epoch,
                                                                   args.local_bs,
                                                                   args.seed)
    log_id = args.directory + parameter_list
    train, val, test = process_isolated(args, data)

    s_global_model = LSTM(args).to(device)
    s_control_global = LSTM(args).to(device)
    s_global_model.train()
    print(s_global_model)

    s_control_weights = s_control_global.state_dict()
    s_global_weights = copy.deepcopy(s_global_model.state_dict())
    s_c_glob = copy.deepcopy(s_control_weights)

    s_c_local = {}
    for idx in selected_cells:
        s_c_local[idx] = LSTM(args).to(device)
        s_c_local[idx].load_state_dict(s_control_weights)
    s_delta_c = copy.deepcopy(s_global_model.state_dict())

    s_weight = copy.deepcopy(s_global_weights)
    for w in s_global_weights:
        s_weight[w] = s_global_weights[w] - s_global_weights[w]
    # s_ww = copy.deepcopy(s_weight)
    # w_locals = [copy.deepcopy(weight) for i in range(11455)]


    s_best_val_loss = None
    s_val_loss = []
    s_val_acc = []
    s_cell_loss = []
    s_loss_hist = []


    for epoch in tqdm.tqdm(range(args.epochs)):
        s_local_weights, s_local_losses = [], []
        # local_grad_np = []
        for i in s_delta_c:
            s_delta_c[i] = 0.0
        # print(f'\n | Global Training Round: {epoch+1} |\n')
        s_global_model.train()

        m = max(int(args.frac * args.bs), 1)
        cell_idx = random.sample(selected_cells, m)
        # print(cell_idx)

        for cell in cell_idx:
            cell_train, cell_test = train[cell], test[cell]
            local_model = LocalUpdate(args, cell_train, cell_test)

            s_global_model.load_state_dict(s_global_weights)
            s_global_model.train()

            s_w, s_loss, s_epoch_loss, s_local_delta_c, s_local_delta, s_control_local_w = local_model.update_weights_scaffold(model=copy.deepcopy(s_global_model),
                                                              control_local=copy.deepcopy(s_c_local[cell]), control_global = copy.deepcopy(s_control_global))
            if epoch != 0:
                s_c_local[cell].load_state_dict(s_control_local_w)

            # local_weights.append(copy.deepcopy(w))
            s_local_weights.append(copy.deepcopy(s_w))
            s_local_losses.append(copy.deepcopy(s_loss))
            s_cell_loss.append(s_loss)

            for i in s_delta_c:
                 s_delta_c[i] += s_local_delta_c[i]

        s_loss_hist.append(sum(s_cell_loss)/len(s_cell_loss))

        for i in s_delta_c:
            s_delta_c[i] /= m

        # Update global model

        s_global_weights = average_weights(s_local_weights)
        # for w in s_weight:
        #     s_global_weights[w] += s_new_global_weight[w]

        s_control_global_w = copy.deepcopy(s_control_global.state_dict())
        for i in s_control_global_w:
            # if epoch != 0:
            s_control_global_w[i] += (m / len(selected_cells)) * s_delta_c[i]
                # control_global_w[i] += (m / args.num_users) * delta_c[i]

        s_global_model.load_state_dict(s_global_weights)
        s_control_global.load_state_dict(s_control_global_w)


    # Test model accuracy
    s_pred, s_truth = {}, {}
    s_test_loss_list = []
    s_test_mse_list = []
    s_nrmse = 0.0

    s_global_model.load_state_dict(s_global_weights)

    for cell in selected_cells:
        cell_test = test[cell]
        s_test_loss, s_test_mse, s_test_nrmse, s_pred[cell], s_truth[cell] = test_inference(args, s_global_model, cell_test)
        # print(f'Cell {cell} MSE {test_mse:.4f}')
        s_nrmse += s_test_nrmse

        s_test_loss_list.append(s_test_loss)
        s_test_mse_list.append(s_test_mse)

    s_df_pred = pd.DataFrame.from_dict(s_pred)
    s_df_truth = pd.DataFrame.from_dict(s_truth)

    s_mse = metrics.mean_squared_error(s_df_pred.values.ravel(), s_df_truth.values.ravel())
    s_mae = metrics.mean_absolute_error(s_df_pred.values.ravel(), s_df_truth.values.ravel())
    s_nrmse = s_nrmse / len(selected_cells)
    print('Scaffold File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, s_mse, s_mae,
                                                                                     s_nrmse))
    plt.figure()
    plt.plot(range(len(s_loss_hist)), s_loss_hist, label='Scaffold')
    plt.ylabel('train_loss')
    plt.xlabel('epochs')
    plt.legend(loc='best')
    plt.savefig(
        './save/scaffold_{}.png'.format('train_loss'))
