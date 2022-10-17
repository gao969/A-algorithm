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

    global_model = LSTM(args).to(device)
    control_global = LSTM(args).to(device)
    global_model.train()
    print(global_model)
    # print(selected_cells)

    control_weights = control_global.state_dict()
    global_weights = global_model.state_dict()
    c_glob = copy.deepcopy(control_weights)


    c_local = {}
    c_local_epoch = {}
    for idx in selected_cells:
        c_local[idx] = LSTM(args).to(device)
        c_local[idx].load_state_dict(control_weights)
        c_local_epoch[idx] = 0
    delta_c = copy.deepcopy(global_model.state_dict())

    weight = copy.deepcopy(global_weights)
    for w in global_weights:
        weight[w] = global_weights[w] - global_weights[w]
    ww = copy.deepcopy(weight)

    w_locals = {}
    for idx in selected_cells:
        w_locals[idx] = copy.deepcopy(weight)

    lay_shape = {}
    for k in weight:
        lay_shape[k] = torch.ones(1)
        for i in range(len(weight[k].shape)):
            lay_shape[k] *= weight[k].shape[i]

    best_val_loss = None
    val_loss = []
    val_acc = []
    cell_loss = []
    loss_hist = []
    # epoch_pearson = []
    epoch_pearson_dgc = []

    for epoch in tqdm.tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        local_grad_np_dgc = []
        # local_grad_np = []
        for i in delta_c:
            delta_c[i] = 0.0
        # print(f'\n | Global Training Round: {epoch+1} |\n')
        global_model.train()

        m = max(int(args.frac * args.bs), 1)
        cell_idx = random.sample(selected_cells, m)
        # print(cell_idx)

        for cell in cell_idx:
            cell_train, cell_test = train[cell], test[cell]
            local_model = LocalUpdate(args, cell_train, cell_test)

            global_model.load_state_dict(global_weights)
            global_model.train()

            w, loss, epoch_loss, local_delta_c, local_delta, control_local_w, ww, grad_np_dgc, c_local_epoch[cell] = local_model.update_weights_topkp(model=copy.deepcopy(global_model),
                                                             global_round=epoch, control_local=c_local[cell], control_global = control_global, w_loc=w_locals[cell], c_local_epoch=c_local_epoch[idx])
            if epoch != 0:
                c_local[cell].load_state_dict(control_local_w)

            # local_weights.append(copy.deepcopy(w))
            local_weights.append(copy.deepcopy(local_delta))
            local_losses.append(copy.deepcopy(loss))
            cell_loss.append(loss)
            # local_grad_np.append(grad_np)
            local_grad_np_dgc.append(grad_np_dgc)

            for i in delta_c:
                # if iter != 0:
                # if epoch == 0:
                #     delta_c[i] += w[i]
                # else:
                 delta_c[i] += local_delta_c[i]
            for k in ww:
                # print(ww[k])
                w_locals[cell][k] = ww[k]

        loss_hist.append(sum(cell_loss)/len(cell_loss))

        # pearson
        # global_grad_np = []
        # for l in range(len(local_grad_np)):
        #     b = local_grad_np[l]
        #     global_grad_np = np.concatenate([global_grad_np, local_grad_np[l]])
        #     # global_grad_np = np.vstack((global_grad_np, local_grad_np[l]))
        # global_grad_np = global_grad_np.reshape(m, -1)
        # epoch_pearson.append(np.corrcoef(global_grad_np))

        mu = np.zeros((m,1))
        global_grad_np_dgc = []
        for l in range(len(local_grad_np_dgc)):
            b = local_grad_np_dgc[l]
            global_grad_np_dgc = np.concatenate([global_grad_np_dgc, local_grad_np_dgc[l]])
            # global_grad_np = np.vstack((global_grad_np, local_grad_np[l]))
        global_grad_np_dgc = global_grad_np_dgc.reshape(m, -1)
        epoch_pearson_dgc.append(np.corrcoef(global_grad_np_dgc))

        for i in range(10):
            a = 0
            for j in range(10):
                a += epoch_pearson_dgc[epoch][i][j]
            mu[i][0] = a - 1
        #weighted
        new_global_np_dgc = np.dot(np.corrcoef(global_grad_np_dgc ), global_grad_np_dgc)
        a = np.zeros_like(grad_np_dgc)
        for l in range(m):
            a += new_global_np_dgc[l]
        a = a/m
        new_global_weight = {}
        aa = []
        b = []
        count = 0
        for k in lay_shape:
            c = int(lay_shape[k].item())
            b = a[count:count+c]
            d = torch.from_numpy(b)
            new_global_weight[k] = d.reshape(weight[k].shape)
            # aa.append(d.reshape(weight[k].shape))
            count += c





        for i in delta_c:
            delta_c[i] /= m

        # Update global model

        # f_global_weights = average_weights(local_weights)
        for w in weight:
            global_weights[w] += new_global_weight[w]

        control_global_w = control_global.state_dict()
        for i in control_global_w:
            # if epoch != 0:
            control_global_w[i] += (1 / m) * delta_c[i]
                # control_global_w[i] += (m / args.num_users) * delta_c[i]

        global_model.load_state_dict(global_weights)


    # Test model accuracy
    pred, truth = {}, {}
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0

    global_model.load_state_dict(global_weights)

    for cell in selected_cells:
        cell_test = test[cell]
        test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference(args, global_model, cell_test)
        # print(f'Cell {cell} MSE {test_mse:.4f}')
        nrmse += test_nrmse

        test_loss_list.append(test_loss)
        test_mse_list.append(test_mse)

    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)

    # txt
    # with open('pearson.txt', 'w') as a:
    #     for i in range(len(epoch_pearson)):
    #         # print(i + ":" + "\r" + epoch_pearson[i] + "\r")
    #         a.write(str(i) + ":" + "\r" + str(epoch_pearson[i]) + "\r")
    with open('pearson_dgc.txt', 'w') as a:
        for i in range(len(epoch_pearson_dgc)):
            a.write(str(i) + ":" + "\r" + str(epoch_pearson_dgc[i]) + "\r")

    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    nrmse = nrmse / len(selected_cells)
    print('Scaffold_TopK.p File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse, mae,
                                                                                     nrmse))
    plt.figure()
    plt.plot(range(len(loss_hist)), loss_hist, label='Scaffold-TopK.p')
    plt.ylabel('train_loss')
    plt.xlabel('epochs')
    plt.legend(loc='best')
    plt.savefig(
        './save/scaffold_TopK.p_{}.png'.format('train_loss'))
