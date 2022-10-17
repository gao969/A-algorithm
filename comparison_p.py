
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
    # print(global_model)
    # print(selected_cells)
    global_model_p10 = LSTM(args).to(device)
    control_global_p10 = LSTM(args).to(device)
    global_model_p10.train()
    global_model_p0 = LSTM(args).to(device)
    control_global_p0 = LSTM(args).to(device)
    global_model_p0.train()
    global_model_p15 = LSTM(args).to(device)
    control_global_p15 = LSTM(args).to(device)
    global_model_p15.train()



    control_weights = control_global.state_dict()
    global_weights = global_model.state_dict()
    c_glob = copy.deepcopy(control_weights)
    control_weights_p10 = control_global_p10.state_dict()
    global_weights_p10 = global_model_p10.state_dict()
    c_globz_p10 = copy.deepcopy(control_weights_p10)
    control_weights_p0 = control_global_p0.state_dict()
    global_weights_p0 = global_model_p0.state_dict()
    c_glob_p0 = copy.deepcopy(control_weights_p0)
    control_weights_p15 = control_global_p15.state_dict()
    global_weights_p15 = global_model_p15.state_dict()
    c_glob_p15 = copy.deepcopy(control_weights_p15)


    c_local = {}
    c_local_epoch = {}
    c_local_p10 = {}
    c_local_epoch_p10 = {}
    c_local_p0 = {}
    c_local_epoch_p0 = {}
    c_local_p15 = {}
    c_local_epoch_p15 = {}
    for idx in selected_cells:
        c_local[idx] = LSTM(args).to(device)
        c_local[idx].load_state_dict(control_weights)
        c_local_epoch[idx] = 0
        c_local_p10[idx] = LSTM(args).to(device)
        c_local_p10[idx].load_state_dict(control_weights_p10)
        c_local_epoch_p10[idx] = 0
        c_local_p0[idx] = LSTM(args).to(device)
        c_local_p0[idx].load_state_dict(control_weights_p0)
        c_local_epoch_p0[idx] = 0
        c_local_p15[idx] = LSTM(args).to(device)
        c_local_p15[idx].load_state_dict(control_weights_p15)
        c_local_epoch_p15[idx] = 0
    delta_c = copy.deepcopy(global_model.state_dict())
    delta_c_p10 = copy.deepcopy(global_model_p10.state_dict())
    delta_c_p0 = copy.deepcopy(global_model_p0.state_dict())
    delta_c_p15 = copy.deepcopy(global_model_p15.state_dict())

    weight = copy.deepcopy(global_weights)
    weight_p10 = copy.deepcopy(global_weights_p10)
    weight_p0 = copy.deepcopy(global_weights_p0)
    weight_p15 = copy.deepcopy(global_weights_p15)
    for w in global_weights:
        weight[w] = global_weights[w] - global_weights[w]
        weight_p10[w] = global_weights_p10[w] - global_weights_p10[w]
        weight_p0[w] = global_weights_p0[w] - global_weights_p0[w]
        weight_p15[w] = global_weights_p15[w] - global_weights_p15[w]
    ww = copy.deepcopy(weight)
    ww_p10 = copy.deepcopy(weight_p10)
    ww_p0 = copy.deepcopy(weight_p0)
    ww_p15 = copy.deepcopy(weight_p15)

    w_locals = {}
    w_locals_p10 = {}
    w_locals_p0 = {}
    w_locals_p15 = {}
    for idx in selected_cells:
        w_locals[idx] = copy.deepcopy(weight)
        w_locals_p10[idx] = copy.deepcopy(weight_p10)
        w_locals_p0[idx] = copy.deepcopy(weight_p0)
        w_locals_p15[idx] = copy.deepcopy(weight_p15)

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
    best_val_loss_p10 = None
    val_loss_p10= []
    val_acc_p10 = []
    cell_loss_p10 = []
    loss_hist_p10 = []
    epoch_pearson_dgc_p10 = []
    best_val_loss_p0 = None
    val_loss_p0 = []
    val_acc_p0 = []
    cell_loss_p0 = []
    loss_hist_p0 = []
    epoch_pearson_dgc_p0 = []
    best_val_loss_p15 = None
    val_loss_p15 = []
    val_acc_p15 = []
    cell_loss_p15 = []
    loss_hist_p15 = []
    epoch_pearson_dgc_p15 = []

    for epoch in tqdm.tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        local_grad_np_dgc = []
        local_weights_p10, local_losses_p10 = [], []
        local_grad_np_dgc_p10 = []
        local_weights_p0, local_losses_p0 = [], []
        local_grad_np_dgc_p0 = []
        local_weights_p15, local_losses_p15 = [], []
        local_grad_np_dgc_p15 = []
        # local_grad_np = []
        for i in delta_c:
            delta_c[i] = 0.0
            delta_c_p10[i] = 0.0
            delta_c_p0[i] = 0.0
            delta_c_p15[i] = 0.0
        # print(f'\n | Global Training Round: {epoch+1} |\n')
        global_model.train()
        global_model_p10.train()
        global_model_p0.train()
        global_model_p15.train()

        m = max(int(args.frac * args.bs), 1)
        cell_idx = random.sample(selected_cells, m)
        # print(cell_idx)

        for cell in cell_idx:
            cell_train, cell_test = train[cell], test[cell]
            local_model = LocalUpdate(args, cell_train, cell_test)

            global_model.load_state_dict(global_weights)
            global_model.train()
            global_model_p10.load_state_dict(global_weights_p10)
            global_model_p10.train()
            global_model_p0.load_state_dict(global_weights_p0)
            global_model_p0.train()
            global_model_p15.load_state_dict(global_weights_p15)
            global_model_p15.train()

            w, loss, epoch_loss, local_delta_c, local_delta, control_local_w, ww, grad_np_dgc, c_local_epoch[cell] = local_model.update_weights_topkp(model=copy.deepcopy(global_model),
                                                             global_round=epoch, control_local=c_local[cell], control_global = control_global, w_loc=w_locals[cell], c_local_epoch=c_local_epoch[idx])
            w_p10, loss_p10, epoch_loss_p10, local_delta_c_p10, local_delta_p10, control_local_w_p10, ww_p10, grad_np_dgc_p10, c_local_epoch_p10[cell] = local_model.update_weights_topkp_p10(model=copy.deepcopy(global_model_p10),
                                                        global_round=epoch, control_local=c_local_p10[cell],
                                                        control_global=control_global_p10, w_loc=w_locals_p10[cell],
                                                        c_local_epoch=c_local_epoch_p10[idx])
            w_p0, loss_p0, epoch_loss_p0, local_delta_c_p0, local_delta_p0, control_local_w_p0, ww_p0, grad_np_dgc_p0, \
            c_local_epoch_p0[cell] = local_model.update_weights_topkp_p0(model=copy.deepcopy(global_model_p0),
                                                                        global_round=epoch,
                                                                        control_local=c_local_p0[cell],
                                                                        control_global=control_global_p0,
                                                                        w_loc=w_locals_p0[cell],
                                                                        c_local_epoch=c_local_epoch_p0[idx])
            w_p15, loss_p15, epoch_loss_p15, local_delta_c_p15, local_delta_p15, control_local_w_p15, ww_p15, grad_np_dgc_p15, \
            c_local_epoch_p15[cell] = local_model.update_weights_topkp_p15(model=copy.deepcopy(global_model_p15),
                                                                          global_round=epoch,
                                                                          control_local=c_local_p15[cell],
                                                                          control_global=control_global_p15,
                                                                          w_loc=w_locals_p15[cell],
                                                                          c_local_epoch=c_local_epoch_p15[idx])

            if epoch != 0:
                c_local[cell].load_state_dict(control_local_w)
                c_local_p10[cell].load_state_dict(control_local_w_p10)
                c_local_p0[cell].load_state_dict(control_local_w_p0)
                c_local_p15[cell].load_state_dict(control_local_w_p15)

            # local_weights.append(copy.deepcopy(w))
            local_weights.append(copy.deepcopy(local_delta))
            local_losses.append(copy.deepcopy(loss))
            cell_loss.append(loss)
            # local_grad_np.append(grad_np)
            local_grad_np_dgc.append(grad_np_dgc)
            local_weights_p10.append(copy.deepcopy(local_delta_p10))
            local_losses_p10.append(copy.deepcopy(loss_p10))
            cell_loss_p10.append(loss_p10)
            local_grad_np_dgc_p10.append(grad_np_dgc_p10)
            local_weights_p0.append(copy.deepcopy(local_delta_p0))
            local_losses_p0.append(copy.deepcopy(loss_p0))
            cell_loss_p0.append(loss_p0)
            local_grad_np_dgc_p0.append(grad_np_dgc_p0)
            local_weights_p15.append(copy.deepcopy(local_delta_p15))
            local_losses_p15.append(copy.deepcopy(loss_p15))
            cell_loss_p15.append(loss_p15)
            local_grad_np_dgc_p15.append(grad_np_dgc_p15)

            for i in delta_c:
                # if iter != 0:
                # if epoch == 0:
                #     delta_c[i] += w[i]
                # else:
                delta_c[i] += local_delta_c[i]
                delta_c_p10[i] += local_delta_c_p10[i]
                delta_c_p0[i] += local_delta_c_p0[i]
                delta_c_p15[i] += local_delta_c_p15[i]
            for k in ww:
                # print(ww[k])
                w_locals[cell][k] = ww[k]
                w_locals_p10[cell][k] = ww_p10[k]
                w_locals_p0[cell][k] = ww_p0[k]
                w_locals_p15[cell][k] = ww_p15[k]

        loss_hist.append(sum(cell_loss)/len(cell_loss))
        loss_hist_p10.append(sum(cell_loss_p10) / len(cell_loss_p10))
        loss_hist_p0.append(sum(cell_loss_p0)/len(cell_loss_p0))
        loss_hist_p15.append(sum(cell_loss_p15) / len(cell_loss_p15))

        # pearson
        # global_grad_np = []
        # for l in range(len(local_grad_np)):
        #     b = local_grad_np[l]
        #     global_grad_np = np.concatenate([global_grad_np, local_grad_np[l]])
        #     # global_grad_np = np.vstack((global_grad_np, local_grad_np[l]))
        # global_grad_np = global_grad_np.reshape(m, -1)
        # epoch_pearson.append(np.corrcoef(global_grad_np))

        mu = np.zeros((m, 1))
        global_grad_np_dgc = []
        global_grad_np_dgc_p10 = []
        global_grad_np_dgc_p0 = []
        global_grad_np_dgc_p15 = []
        for l in range(len(local_grad_np_dgc)):
            b = local_grad_np_dgc[l]
            global_grad_np_dgc = np.concatenate([global_grad_np_dgc, local_grad_np_dgc[l]])
            b_p10 = local_grad_np_dgc_p10[l]
            global_grad_np_dgc_p10 = np.concatenate([global_grad_np_dgc_p10, local_grad_np_dgc_p10[l]])
            b_p0 = local_grad_np_dgc_p0[l]
            global_grad_np_dgc_p0 = np.concatenate([global_grad_np_dgc_p0, local_grad_np_dgc_p0[l]])
            b_p15 = local_grad_np_dgc_p15[l]
            global_grad_np_dgc_p15 = np.concatenate([global_grad_np_dgc_p15, local_grad_np_dgc_p15[l]])
            # global_grad_np = np.vstack((global_grad_np, local_grad_np[l]))
        global_grad_np_dgc = global_grad_np_dgc.reshape(m, -1)
        epoch_pearson_dgc.append(np.corrcoef(global_grad_np_dgc))
        global_grad_np_dgc_p10 = global_grad_np_dgc_p10.reshape(m, -1)
        epoch_pearson_dgc_p10.append(np.corrcoef(global_grad_np_dgc_p10))
        global_grad_np_dgc_p0 = global_grad_np_dgc_p0.reshape(m, -1)
        epoch_pearson_dgc_p0.append(np.corrcoef(global_grad_np_dgc_p0))
        global_grad_np_dgc_p15 = global_grad_np_dgc_p15.reshape(m, -1)
        epoch_pearson_dgc_p15.append(np.corrcoef(global_grad_np_dgc_p15))

        for i in range(10):
            a = 0
            for j in range(10):
                a += epoch_pearson_dgc[epoch][i][j]
            mu[i][0] = a - 1
        #weighted
        new_global_np_dgc = np.dot(np.corrcoef(global_grad_np_dgc ), global_grad_np_dgc)
        a = np.zeros_like(grad_np_dgc)
        new_global_np_dgc_p10 = np.dot(np.corrcoef(global_grad_np_dgc_p10), global_grad_np_dgc_p10)
        a_p10 = np.zeros_like(grad_np_dgc_p10)
        new_global_np_dgc_p0 = np.dot(np.corrcoef(global_grad_np_dgc_p0), global_grad_np_dgc_p0)
        a_p0 = np.zeros_like(grad_np_dgc_p0)
        new_global_np_dgc_p15 = np.dot(np.corrcoef(global_grad_np_dgc_p15), global_grad_np_dgc_p15)
        a_p15 = np.zeros_like(grad_np_dgc_p15)
        for l in range(m):
            a += new_global_np_dgc[l]
            a_p10 += new_global_np_dgc_p10[l]
            a_p0 += new_global_np_dgc_p0[l]
            a_p15 += new_global_np_dgc_p15[l]
        a = a/m
        new_global_weight = {}
        aa = []
        b = []
        count = 0
        a_p10 = a_p10 / m
        new_global_weight_p10 = {}
        b_p10 = []
        count_p10 = 0
        a_p0 = a_p0 / m
        new_global_weight_p0 = {}
        b_p0 = []
        count_p0 = 0
        a_p15 = a_p15 / m
        new_global_weight_p15 = {}
        aa_p15 = []
        b_p15 = []
        count_p15 = 0
        for k in lay_shape:
            c = int(lay_shape[k].item())
            b = a[count:count+c]
            d = torch.from_numpy(b)
            new_global_weight[k] = d.reshape(weight[k].shape)
            # aa.append(d.reshape(weight[k].shape))
            count += c
            c_p10 = int(lay_shape[k].item())
            b_p10 = a_p10[count_p10:count_p10 + c_p10]
            d_p10 = torch.from_numpy(b_p10)
            new_global_weight_p10[k] = d_p10.reshape(weight_p10[k].shape)
            count_p10 += c_p10
            c_p0 = int(lay_shape[k].item())
            b_p0 = a_p0[count_p0:count_p0 + c_p0]
            d_p0 = torch.from_numpy(b_p0)
            new_global_weight_p0[k] = d_p0.reshape(weight_p0[k].shape)
            count_p0 += c_p0
            c_p15 = int(lay_shape[k].item())
            b_p15 = a_p15[count_p15:count_p15 + c_p15]
            d_p15 = torch.from_numpy(b_p15)
            new_global_weight_p15[k] = d_p15.reshape(weight_p15[k].shape)
            count_p15 += c_p15





        for i in delta_c:
            delta_c[i] /= m
            delta_c_p10[i] /= m
            delta_c_p0[i] /= m
            delta_c_p15[i] /= m

        # Update global model

        # f_global_weights = average_weights(local_weights)
        for w in weight:
            global_weights[w] += new_global_weight[w]
            global_weights_p10[w] += new_global_weight_p10[w]
            global_weights_p0[w] += new_global_weight_p0[w]
            global_weights_p15[w] += new_global_weight_p15[w]

        control_global_w = control_global.state_dict()
        control_global_w_p10 = control_global_p10.state_dict()
        control_global_w_p0 = control_global_p0.state_dict()
        control_global_w_p15 = control_global_p15.state_dict()
        for i in control_global_w:
            # if epoch != 0:
            control_global_w[i] += (1 / m) * delta_c[i]
            control_global_w_p10[i] += (1 / m) * delta_c_p10[i]
            control_global_w_p0[i] += (1 / m) * delta_c_p0[i]
            control_global_w_p15[i] += (1 / m) * delta_c_p15[i]
                # control_global_w[i] += (m / args.num_users) * delta_c[i]

        global_model.load_state_dict(global_weights)
        global_model_p10.load_state_dict(global_weights_p10)
        global_model_p0.load_state_dict(global_weights_p0)
        global_model_p15.load_state_dict(global_weights_p15)


    # Test model accuracy
    pred, truth = {}, {}
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0
    pred_p10, truth_p10 = {}, {}
    test_loss_list_p10 = []
    test_mse_list_p10 = []
    nrmse_p10 = 0.0
    pred_p0, truth_p0 = {}, {}
    test_loss_list_p0 = []
    test_mse_list_p0 = []
    nrmse_p0 = 0.0
    pred_p15, truth_p15 = {}, {}
    test_loss_list_p15 = []
    test_mse_list_p15 = []
    nrmse_p15 = 0.0

    global_model.load_state_dict(global_weights)
    global_model_p10.load_state_dict(global_weights_p10)
    global_model_p0.load_state_dict(global_weights_p0)
    global_model_p15.load_state_dict(global_weights_p15)

    for cell in selected_cells:
        cell_test = test[cell]
        test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference(args, global_model, cell_test)
        test_loss_p10, test_mse_p10, test_nrmse_p10, pred_p10[cell], truth_p10[cell] = test_inference(args, global_model_p10, cell_test)
        test_loss_p0, test_mse_p0, test_nrmse_p0, pred_p0[cell], truth_p0[cell] = test_inference(args, global_model_p0, cell_test)
        test_loss_p15, test_mse_p15, test_nrmse_p15, pred_p15[cell], truth_p15[cell] = test_inference(args, global_model_p15, cell_test)
        # print(f'Cell {cell} MSE {test_mse:.4f}')
        nrmse += test_nrmse
        nrmse_p10 += test_nrmse_p10
        nrmse_p0 += test_nrmse_p0
        nrmse_p15 += test_nrmse_p15

        test_loss_list.append(test_loss)
        test_mse_list.append(test_mse)
        test_loss_list_p10.append(test_loss_p10)
        test_mse_list_p10.append(test_mse_p10)
        test_loss_list_p0.append(test_loss_p0)
        test_mse_list_p0.append(test_mse_p0)
        test_loss_list_p15.append(test_loss_p15)
        test_mse_list_p15.append(test_mse_p15)

    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)
    df_pred_p10 = pd.DataFrame.from_dict(pred_p10)
    df_truth_p10 = pd.DataFrame.from_dict(truth_p10)
    df_pred_p0 = pd.DataFrame.from_dict(pred_p0)
    df_truth_p0 = pd.DataFrame.from_dict(truth_p0)
    df_pred_p15 = pd.DataFrame.from_dict(pred_p15)
    df_truth_p15 = pd.DataFrame.from_dict(truth_p15)

    # txt
    # with open('pearson.txt', 'w') as a:
    #     for i in range(len(epoch_pearson)):
    #         # print(i + ":" + "\r" + epoch_pearson[i] + "\r")
    #         a.write(str(i) + ":" + "\r" + str(epoch_pearson[i]) + "\r")
    with open('pearson_dgc.txt', 'w') as a:
        for i in range(len(epoch_pearson_dgc)):
            a.write(str(i) + ":" + "\r" + str(epoch_pearson_dgc[i]) + "\r")

    mse_p0 = metrics.mean_squared_error(df_pred_p0.values.ravel(), df_truth_p0.values.ravel())
    mae_p0 = metrics.mean_absolute_error(df_pred_p0.values.ravel(), df_truth_p0.values.ravel())
    nrmse_p0 = nrmse_p0 / len(selected_cells)
    print('Scaffold_TopK.p_p0 File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type,
                                                                                                 mse_p0, mae_p0,
                                                                                                 nrmse_p0))
    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    nrmse = nrmse / len(selected_cells)
    print('Scaffold_TopK.p_p5 File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse, mae,
                                                                                     nrmse))
    mse_p10 = metrics.mean_squared_error(df_pred_p10.values.ravel(), df_truth_p10.values.ravel())
    mae_p10 = metrics.mean_absolute_error(df_pred_p10.values.ravel(), df_truth_p10.values.ravel())
    nrmse_p10 = nrmse_p10 / len(selected_cells)
    print('Scaffold_TopK.p_p10 File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse_p10,
                                                                                              mae_p10,
                                                                                              nrmse_p10))

    mse_p15 = metrics.mean_squared_error(df_pred_p15.values.ravel(), df_truth_p15.values.ravel())
    mae_p15 = metrics.mean_absolute_error(df_pred_p15.values.ravel(), df_truth_p15.values.ravel())
    nrmse_p15 = nrmse_p15 / len(selected_cells)
    print('Scaffold_TopK.p_p15 File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type,
                                                                                              mse_p15, mae_p15,
                                                                                              nrmse_p15))
    plt.figure()
    plt.plot(range(len(loss_hist)), loss_hist_p0, label='p=0')
    plt.plot(range(len(loss_hist)), loss_hist, label='p=5')
    plt.plot(range(len(loss_hist)), loss_hist_p10, label='p=10')
    plt.plot(range(len(loss_hist)), loss_hist_p15, label='p=15')
    plt.ylabel('train_loss')
    plt.xlabel('epochs')
    plt.legend(loc='best')
    plt.savefig(
        './save/scaffold_TopK.p_p=051015_{}.png'.format('train_loss'))

