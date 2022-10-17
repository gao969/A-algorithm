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
    # print(selected_cells)

    parameter_list = 'FedAvg-data-{:}-type-{:}-'.format(args.file, args.type)
    parameter_list += '-frac-{:.2f}-le-{:}-lb-{:}-seed-{:}'.format(args.frac, args.local_epoch,
                                                                   args.local_bs,
                                                                   args.seed)
    log_id = args.directory + parameter_list
    train, val, test = process_isolated(args, data)

    topk_global_model = LSTM(args).to(device)
    topk_control_global = LSTM(args).to(device)
    topk_global_model.train()
    print(topk_global_model)
    # print(selected_cells)

    topk_control_weights = topk_control_global.state_dict()
    topk_global_weights = topk_global_model.state_dict()
    topk_c_glob = copy.deepcopy(topk_control_weights)

    # c_local = [LSTM(args).to(device) for i in range(11455)]
    # for net in c_local:
    #     net.load_state_dict(control_weights)
    topk_c_local = {}
    for idx in selected_cells:
        topk_c_local[idx] = LSTM(args).to(device)
        topk_c_local[idx].load_state_dict(topk_control_weights)
    topk_delta_c = copy.deepcopy(topk_global_model.state_dict())

    topk_weight = copy.deepcopy(topk_global_weights)
    for w in topk_global_weights:
        topk_weight[w] = topk_global_weights[w] - topk_global_weights[w]
    topk_ww = copy.deepcopy(topk_weight)
    # w_locals = [copy.deepcopy(weight) for i in range(11455)]
    topk_w_locals = {}
    for idx in selected_cells:
        topk_w_locals[idx] = copy.deepcopy(topk_weight)

    lay_shape = {}
    for k in topk_weight:
        lay_shape[k] = torch.ones(1)
        for i in range(len(topk_weight[k].shape)):
            lay_shape[k] *= topk_weight[k].shape[i]

    topk_best_val_loss = None
    topk_val_loss = []
    topk_val_acc = []
    topk_cell_loss = []
    topk_loss_hist = []
    # epoch_pearson = []
    topk_epoch_pearson_dgc = []

    for epoch in tqdm.tqdm(range(args.epochs)):
        topk_local_weights, topk_local_losses = [], []
        topk_local_grad_np_dgc = []
        # local_grad_np = []
        for i in topk_delta_c:
            topk_delta_c[i] = 0.0
        # print(f'\n | Global Training Round: {epoch+1} |\n')
        topk_global_model.train()

        m = max(int(args.frac * args.bs), 1)
        cell_idx = random.sample(selected_cells, m)
        # print(cell_idx)

        for cell in cell_idx:
            cell_train, cell_test = train[cell], test[cell]
            local_model = LocalUpdate(args, cell_train, cell_test)

            topk_global_model.load_state_dict(topk_global_weights)
            topk_global_model.train()

            topk_w, topk_loss, topk_epoch_loss, topk_local_delta_c, topk_local_delta, topk_control_local_w, topk_ww, topk_grad_np_dgc = local_model.update_weights_topk(model=copy.deepcopy(topk_global_model),
                                                             global_round=epoch, control_local=topk_c_local[cell], control_global = topk_control_global, w_loc = topk_w_locals[cell])
            if epoch != 0:
                topk_c_local[cell].load_state_dict(topk_control_local_w)

            # local_weights.append(copy.deepcopy(w))
            topk_local_weights.append(copy.deepcopy(topk_local_delta))
            topk_local_losses.append(copy.deepcopy(topk_loss))
            topk_cell_loss.append(topk_loss)
            # local_grad_np.append(grad_np)
            topk_local_grad_np_dgc.append(topk_grad_np_dgc)

            for i in topk_delta_c:
                # if iter != 0:
                # if epoch == 0:
                #     delta_c[i] += w[i]
                # else:
                 topk_delta_c[i] += topk_local_delta_c[i]
            for k in topk_ww:
                # print(ww[k])
                topk_w_locals[cell][k] = topk_ww[k]

        topk_loss_hist.append(sum(topk_cell_loss)/len(topk_cell_loss))

        # pearson
        # global_grad_np = []
        # for l in range(len(local_grad_np)):
        #     b = local_grad_np[l]
        #     global_grad_np = np.concatenate([global_grad_np, local_grad_np[l]])
        #     # global_grad_np = np.vstack((global_grad_np, local_grad_np[l]))
        # global_grad_np = global_grad_np.reshape(m, -1)
        # epoch_pearson.append(np.corrcoef(global_grad_np))

        topk_global_grad_np_dgc = []
        for l in range(len(topk_local_grad_np_dgc)):
            topk_b = topk_local_grad_np_dgc[l]
            topk_global_grad_np_dgc = np.concatenate([topk_global_grad_np_dgc, topk_local_grad_np_dgc[l]])
            # global_grad_np = np.vstack((global_grad_np, local_grad_np[l]))
        topk_global_grad_np_dgc = topk_global_grad_np_dgc.reshape(m, -1)
        topk_epoch_pearson_dgc.append(np.corrcoef(topk_global_grad_np_dgc))

        #weighted
        topk_new_global_np_dgc = np.dot(np.corrcoef(topk_global_grad_np_dgc ), topk_global_grad_np_dgc)
        topk_a = np.zeros_like(topk_grad_np_dgc)
        for l in range(m):
            topk_a += topk_new_global_np_dgc[l]
        topk_a = topk_a/m
        topk_new_global_weight = {}
        # topk_aa = []
        topk_b = []
        topk_count = 0
        for k in lay_shape:
            topk_c = int(lay_shape[k].item())
            topk_b = topk_a[topk_count:topk_count+topk_c]
            topk_d = torch.from_numpy(topk_b)
            topk_new_global_weight[k] = topk_d.reshape(topk_weight[k].shape)
            # aa.append(d.reshape(weight[k].shape))
            topk_count += topk_c





        for i in topk_delta_c:
            topk_delta_c[i] /= m

        # Update global model

        # f_global_weights = average_weights(local_weights)
        for w in topk_weight:
            topk_global_weights[w] += topk_new_global_weight[w]

        topk_control_global_w = topk_control_global.state_dict()
        for i in topk_control_global_w:
            # if epoch != 0:
            topk_control_global_w[i] += (1 / m) * topk_delta_c[i]
                # control_global_w[i] += (m / args.num_users) * delta_c[i]

        topk_global_model.load_state_dict(topk_global_weights)


    # Test model accuracy
    topk_pred, topk_truth = {}, {}
    topk_test_loss_list = []
    topk_test_mse_list = []
    topk_nrmse = 0.0

    topk_global_model.load_state_dict(topk_global_weights)

    for cell in selected_cells:
        cell_test = test[cell]
        topk_test_loss, topk_test_mse, topk_test_nrmse, topk_pred[cell], topk_truth[cell] = test_inference(args, topk_global_model, cell_test)
        # print(f'Cell {cell} MSE {test_mse:.4f}')
        topk_nrmse += topk_test_nrmse

        topk_test_loss_list.append(topk_test_loss)
        topk_test_mse_list.append(topk_test_mse)

    topk_df_pred = pd.DataFrame.from_dict(topk_pred)
    topk_df_truth = pd.DataFrame.from_dict(topk_truth)

    # txt
    # with open('pearson.txt', 'w') as a:
    #     for i in range(len(epoch_pearson)):
    #         # print(i + ":" + "\r" + epoch_pearson[i] + "\r")
    #         a.write(str(i) + ":" + "\r" + str(epoch_pearson[i]) + "\r")
    with open('pearson_dgc.txt', 'w') as a:
        for i in range(len(topk_epoch_pearson_dgc)):
            a.write(str(i) + ":" + "\r" + str(topk_epoch_pearson_dgc[i]) + "\r")

    topk_mse = metrics.mean_squared_error(topk_df_pred.values.ravel(), topk_df_truth.values.ravel())
    topk_mae = metrics.mean_absolute_error(topk_df_pred.values.ravel(), topk_df_truth.values.ravel())
    topk_nrmse = topk_nrmse / len(selected_cells)
    print('Scaffold_TopK File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, topk_mse, topk_mae,
                                                                                     topk_nrmse))
    plt.figure()
    plt.plot(range(len(topk_loss_hist)), topk_loss_hist, label='Scaffold-Topk')
    plt.ylabel('train_loss')
    plt.xlabel('epochs')
    plt.legend(loc='best')
    plt.savefig(
        './save/scaffold_topk_{}.png'.format('train_loss'))
