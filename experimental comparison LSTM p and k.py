

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

    global_model = LSTM(args).to(device)
    control_global = LSTM(args).to(device)
    global_model.train()
    # print(global_model)
    topk_global_model = LSTM(args).to(device)
    topk_control_global = LSTM(args).to(device)
    topk_global_model.train()
    # print(topk_global_model)
    fed_global_model = LSTM(args).to(device)
    fed_global_model.train()
    s_global_model = LSTM(args).to(device)
    s_control_global = LSTM(args).to(device)
    s_global_model.train()
    top_global_model = LSTM(args).to(device)
    top_control_global = LSTM(args).to(device)
    top_global_model.train()

    control_weights = control_global.state_dict()
    global_weights = global_model.state_dict()
    c_glob = copy.deepcopy(control_weights)
    topk_control_weights = topk_control_global.state_dict()
    topk_global_weights = topk_global_model.state_dict()
    topk_c_glob = copy.deepcopy(topk_control_weights)
    fed_global_weights = fed_global_model.state_dict()
    s_control_weights = s_control_global.state_dict()
    s_global_weights = copy.deepcopy(s_global_model.state_dict())
    s_c_glob = copy.deepcopy(s_control_weights)
    top_control_weights = top_control_global.state_dict()
    top_global_weights = copy.deepcopy(top_global_model.state_dict())
    top_c_glob = copy.deepcopy(top_control_weights)

    c_local = {}
    c_local_epoch = {}
    topk_c_local = {}
    s_c_local = {}
    top_c_local = {}
    for idx in selected_cells:
        c_local[idx] = LSTM(args).to(device)
        c_local[idx].load_state_dict(control_weights)
        c_local_epoch[idx] = 0
        topk_c_local[idx] = LSTM(args).to(device)
        topk_c_local[idx].load_state_dict(topk_control_weights)
        s_c_local[idx] = LSTM(args).to(device)
        s_c_local[idx].load_state_dict(s_control_weights)
        top_c_local[idx] = LSTM(args).to(device)
        top_c_local[idx].load_state_dict(top_control_weights)
    top_delta_c = copy.deepcopy(top_global_model.state_dict())
    delta_c = copy.deepcopy(global_model.state_dict())
    topk_delta_c = copy.deepcopy(topk_global_model.state_dict())
    s_delta_c = copy.deepcopy(s_global_model.state_dict())

    weight = copy.deepcopy(global_weights)
    topk_weight = copy.deepcopy(topk_global_weights)
    s_weight = copy.deepcopy(s_global_weights)
    top_weight = copy.deepcopy(top_global_weights)
    for w in global_weights:
        weight[w] = global_weights[w] - global_weights[w]
        topk_weight[w] = topk_global_weights[w] - topk_global_weights[w]
        s_weight[w] = s_global_weights[w] - s_global_weights[w]
        top_weight[w] = top_global_weights[w] - top_global_weights[w]
    ww = copy.deepcopy(weight)
    topk_ww = copy.deepcopy(topk_weight)
    top_ww = copy.deepcopy(top_weight)

    w_locals = {}
    topk_w_locals = {}
    top_w_locals = {}
    for idx in selected_cells:
        w_locals[idx] = copy.deepcopy(weight)
        topk_w_locals[idx] = copy.deepcopy(topk_weight)
        top_w_locals[idx] = copy.deepcopy(top_weight)

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
    topk_best_val_loss = None
    topk_val_loss = []
    topk_val_acc = []
    topk_cell_loss = []
    topk_loss_hist = []
    # epoch_pearson = []
    topk_epoch_pearson_dgc = []
    fed_best_val_loss = None
    fed_val_loss = []
    fed_val_acc = []
    fed_cell_loss = []
    fed_loss_hist = []
    s_best_val_loss = None
    s_val_loss = []
    s_val_acc = []
    s_cell_loss = []
    s_loss_hist = []
    top_best_val_loss = None
    top_val_loss = []
    top_val_acc = []
    top_cell_loss = []
    top_loss_hist = []


    for epoch in tqdm.tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        local_grad_np_dgc = []
        topk_local_weights, topk_local_losses = [], []
        topk_local_grad_np_dgc = []
        fed_local_weights, fed_local_losses = [], []
        fed_global_model.train()
        s_local_weights, s_local_losses = [], []
        top_local_weights, top_local_losses = [], []
        # local_grad_np = []
        for i in delta_c:
            delta_c[i] = 0.0
            topk_delta_c[i] = 0.0
            s_delta_c[i] = 0.0
            top_delta_c[i] = 0.0
        # print(f'\n | Global Training Round: {epoch+1} |\n')
        global_model.train()
        topk_global_model.train()
        s_global_model.train()
        top_global_model.train()

        m = max(int(args.frac * args.bs), 1)
        cell_idx = random.sample(selected_cells, m)
        # print(cell_idx)

        for cell in cell_idx:
            cell_train, cell_test = train[cell], test[cell]
            local_model = LocalUpdate(args, cell_train, cell_test)

            global_model.load_state_dict(global_weights)
            global_model.train()
            topk_global_model.load_state_dict(topk_global_weights)
            topk_global_model.train()
            fed_global_model.load_state_dict(fed_global_weights)
            fed_global_model.train()
            s_global_model.load_state_dict(s_global_weights)
            s_global_model.train()
            top_global_model.load_state_dict(top_global_weights)
            top_global_model.train()

            w, loss, epoch_loss, local_delta_c, local_delta, control_local_w, ww, grad_np_dgc, c_local_epoch[cell] = local_model.update_weights_topkp_p10(model=copy.deepcopy(global_model),
                                                             global_round=epoch, control_local=c_local[cell], control_global = control_global, w_loc=w_locals[cell], c_local_epoch=c_local_epoch[cell] + 1)
            topk_w, topk_loss, topk_epoch_loss, topk_local_delta_c, topk_local_delta, topk_control_local_w, topk_ww, topk_grad_np_dgc = local_model.update_weights_topk(
                model=copy.deepcopy(topk_global_model),
                global_round=epoch, control_local=topk_c_local[cell], control_global=topk_control_global,
                w_loc=topk_w_locals[cell])
            fed_w, fed_loss, fed_epoch_loss = local_model.update_weights_fed(model=copy.deepcopy(fed_global_model),
                                                                         global_round=epoch)
            s_w, s_loss, s_epoch_loss, s_local_delta_c, s_local_delta, s_control_local_w = local_model.update_weights_scaffold(
                model=copy.deepcopy(s_global_model),
                control_local=copy.deepcopy(s_c_local[cell]), control_global=copy.deepcopy(s_control_global))
            top_w, top_loss, top_epoch_loss, top_local_delta_c, top_local_delta, top_control_local_w, top_ww = local_model.update_weights_topkg(
                model=copy.deepcopy(top_global_model), global_round=epoch,
                control_local=copy.deepcopy(top_c_local[cell]), control_global=copy.deepcopy(top_control_global),
                w_loc=top_w_locals[cell])

            if epoch != 0:
                c_local[cell].load_state_dict(control_local_w)
                topk_c_local[cell].load_state_dict(topk_control_local_w)
                s_c_local[cell].load_state_dict(s_control_local_w)
                top_c_local[cell].load_state_dict(top_control_local_w)

            # local_weights.append(copy.deepcopy(w))
            local_weights.append(copy.deepcopy(local_delta))
            local_losses.append(copy.deepcopy(loss))
            cell_loss.append(loss)
            # local_grad_np.append(grad_np)
            local_grad_np_dgc.append(grad_np_dgc)
            topk_local_weights.append(copy.deepcopy(topk_local_delta))
            topk_local_losses.append(copy.deepcopy(topk_loss))
            topk_cell_loss.append(topk_loss)
            topk_local_grad_np_dgc.append(topk_grad_np_dgc)
            fed_local_weights.append(copy.deepcopy(fed_w))
            fed_local_losses.append(copy.deepcopy(fed_loss))
            fed_cell_loss.append(fed_loss)
            s_local_weights.append(copy.deepcopy(s_w))
            s_local_losses.append(copy.deepcopy(s_loss))
            s_cell_loss.append(s_loss)
            top_local_weights.append(copy.deepcopy(top_local_delta))
            top_local_losses.append(copy.deepcopy(top_loss))
            top_cell_loss.append(top_loss)

            for i in delta_c:
                delta_c[i] += local_delta_c[i]
                topk_delta_c[i] += topk_local_delta_c[i]
                s_delta_c[i] += s_local_delta_c[i]
                top_delta_c[i] += top_local_delta_c[i]
            for k in ww:
                w_locals[cell][k] = ww[k]
                topk_w_locals[cell][k] = topk_ww[k]
                top_w_locals[cell][k] = top_ww[k]

        loss_hist.append(sum(cell_loss) / len(cell_loss))
        topk_loss_hist.append(sum(topk_cell_loss) / len(topk_cell_loss))
        fed_loss_hist.append(sum(fed_cell_loss) / len(fed_cell_loss))
        s_loss_hist.append(sum(s_cell_loss) / len(s_cell_loss))
        top_loss_hist.append(sum(top_cell_loss) / len(top_cell_loss))

        # pearson
        mu = np.zeros((m, 1))
        global_grad_np_dgc = []
        topk_global_grad_np_dgc = []
        for l in range(len(local_grad_np_dgc)):
            b = local_grad_np_dgc[l]
            global_grad_np_dgc = np.concatenate([global_grad_np_dgc, local_grad_np_dgc[l]])
            topk_b = topk_local_grad_np_dgc[l]
            topk_global_grad_np_dgc = np.concatenate([topk_global_grad_np_dgc, topk_local_grad_np_dgc[l]])
            # global_grad_np = np.vstack((global_grad_np, local_grad_np[l]))
        global_grad_np_dgc = global_grad_np_dgc.reshape(m, -1)
        epoch_pearson_dgc.append(np.corrcoef(global_grad_np_dgc))
        topk_global_grad_np_dgc = topk_global_grad_np_dgc.reshape(m, -1)
        topk_epoch_pearson_dgc.append(np.corrcoef(topk_global_grad_np_dgc))
        for i in range(10):
            a = 0
            for j in range(10):
                a += epoch_pearson_dgc[epoch][i][j]
            mu[i][0] = a - 1

        # weighted
        new_global_np_dgc = np.dot(np.corrcoef(global_grad_np_dgc), global_grad_np_dgc)
        a = np.zeros_like(grad_np_dgc)
        topk_new_global_np_dgc = np.dot(np.corrcoef(topk_global_grad_np_dgc), topk_global_grad_np_dgc)
        topk_a = np.zeros_like(topk_grad_np_dgc)
        for l in range(m):
            a += new_global_np_dgc[l]
            topk_a += topk_new_global_np_dgc[l]
        a = a / m
        new_global_weight = {}
        topk_a = topk_a / m
        topk_new_global_weight = {}
        # aa = []
        b = []
        count = 0
        topk_b = []
        topk_count = 0
        for k in lay_shape:
            c = int(lay_shape[k].item())
            b = a[count:count + c]
            d = torch.from_numpy(b)
            new_global_weight[k] = d.reshape(weight[k].shape)
            # aa.append(d.reshape(weight[k].shape))
            count += c
            topk_c = int(lay_shape[k].item())
            topk_b = topk_a[topk_count:topk_count + topk_c]
            topk_d = torch.from_numpy(topk_b)
            topk_new_global_weight[k] = topk_d.reshape(topk_weight[k].shape)
            topk_count += topk_c

        for i in delta_c:
            delta_c[i] /= m
            topk_delta_c[i] /= m
            s_delta_c[i] /= m
            top_delta_c[i] /= m

        # Update global model

        s_global_weights = average_weights(s_local_weights)
        s_control_global_w = copy.deepcopy(s_control_global.state_dict())
        top_global_weights = copy.deepcopy(top_global_model.state_dict())
        for w in weight:
            global_weights[w] += new_global_weight[w]
            topk_global_weights[w] += topk_new_global_weight[w]
            top_global_weights[w] += average_weights(top_local_weights)[w]

        control_global_w = control_global.state_dict()
        topk_control_global_w = topk_control_global.state_dict()
        top_control_global_w = copy.deepcopy(top_control_global.state_dict())
        for i in control_global_w:
            # if epoch != 0:
            control_global_w[i] += (1 / m) * delta_c[i]
            topk_control_global_w[i] += (1 / m) * topk_delta_c[i]
            s_control_global_w[i] += (m / len(selected_cells)) * s_delta_c[i]
            top_control_global_w[i] += (m / len(selected_cells)) * top_delta_c[i]
            # control_global_w[i] += (m / args.num_users) * delta_c[i]

        global_model.load_state_dict(global_weights)
        topk_global_model.load_state_dict(topk_global_weights)

        fed_global_weights = average_weights(fed_local_weights)
        fed_global_model.load_state_dict(fed_global_weights)
        s_global_model.load_state_dict(s_global_weights)
        s_control_global.load_state_dict(s_control_global_w)
        top_global_model.load_state_dict(top_global_weights)
        top_control_global.load_state_dict(top_control_global_w)


    # Test model accuracy
    pred, truth = {}, {}
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0
    topk_pred, topk_truth = {}, {}
    topk_test_loss_list = []
    topk_test_mse_list = []
    topk_nrmse = 0.0
    fed_pred, fed_truth = {}, {}
    fed_test_loss_list = []
    fed_test_mse_list = []
    fed_nrmse = 0.0
    s_pred, s_truth = {}, {}
    s_test_loss_list = []
    s_test_mse_list = []
    s_nrmse = 0.0
    top_pred, top_truth = {}, {}
    top_test_lostop_list = []
    top_test_mse_list = []
    top_nrmse = 0.0

    global_model.load_state_dict(global_weights)
    topk_global_model.load_state_dict(topk_global_weights)
    fed_global_model.load_state_dict(fed_global_weights)
    s_global_model.load_state_dict(s_global_weights)
    top_global_model.load_state_dict(top_global_weights)

    for cell in selected_cells:
        cell_test = test[cell]
        test_loss, test_mse, test_nrmse, pred[cell], truth[cell] = test_inference(args, global_model, cell_test)
        # print(f'Cell {cell} MSE {test_mse:.4f}')
        nrmse += test_nrmse
        topk_test_loss, topk_test_mse, topk_test_nrmse, topk_pred[cell], topk_truth[cell] = test_inference(args, topk_global_model, cell_test)
        topk_nrmse += topk_test_nrmse
        fed_test_loss, fed_test_mse, fed_test_nrmse, fed_pred[cell], fed_truth[cell] = test_inference(args, fed_global_model, cell_test)
        fed_nrmse += fed_test_nrmse
        s_test_loss, s_test_mse, s_test_nrmse, s_pred[cell], s_truth[cell] = test_inference(args, s_global_model, cell_test)
        s_nrmse += s_test_nrmse
        top_test_loss, top_test_mse, top_test_nrmse, top_pred[cell], top_truth[cell] = test_inference(args, top_global_model, cell_test)
        top_nrmse += top_test_nrmse

        test_loss_list.append(test_loss)
        test_mse_list.append(test_mse)
        topk_test_loss_list.append(topk_test_loss)
        topk_test_mse_list.append(topk_test_mse)
        fed_test_loss_list.append(fed_test_loss)
        fed_test_mse_list.append(fed_test_mse)
        s_test_loss_list.append(s_test_loss)
        s_test_mse_list.append(s_test_mse)
        top_df_pred = pd.DataFrame.from_dict(top_pred)
        top_df_truth = pd.DataFrame.from_dict(top_truth)

    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)
    topk_df_pred = pd.DataFrame.from_dict(topk_pred)
    topk_df_truth = pd.DataFrame.from_dict(topk_truth)
    fed_df_pred = pd.DataFrame.from_dict(fed_pred)
    fed_df_truth = pd.DataFrame.from_dict(fed_truth)
    s_df_pred = pd.DataFrame.from_dict(s_pred)
    s_df_truth = pd.DataFrame.from_dict(s_truth)

    # txt
    # with open('pearson.txt', 'w') as a:
    #     for i in range(len(epoch_pearson)):
    #         # print(i + ":" + "\r" + epoch_pearson[i] + "\r")
    #         a.write(str(i) + ":" + "\r" + str(epoch_pearson[i]) + "\r")
    with open('pearson_dgc.txt', 'w') as a:
        for i in range(len(epoch_pearson_dgc)):
            a.write(str(i) + ":" + "\r" + str(epoch_pearson_dgc[i]) + "\r")

    fed_mse = metrics.mean_squared_error(fed_df_pred.values.ravel(), fed_df_truth.values.ravel())
    fed_mae = metrics.mean_absolute_error(fed_df_pred.values.ravel(), fed_df_truth.values.ravel())
    fed_nrmse = fed_nrmse / len(selected_cells)
    print('FedAvg File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, fed_mse,
                                                                                     fed_mae,
                                                                                     fed_nrmse))
    s_mse = metrics.mean_squared_error(s_df_pred.values.ravel(), s_df_truth.values.ravel())
    s_mae = metrics.mean_absolute_error(s_df_pred.values.ravel(), s_df_truth.values.ravel())
    s_nrmse = s_nrmse / len(selected_cells)
    print(
        'Scaffold File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, s_mse, s_mae,
                                                                                     s_nrmse))
    top_mse = metrics.mean_squared_error(top_df_pred.values.ravel(), top_df_truth.values.ravel())
    top_mae = metrics.mean_absolute_error(top_df_pred.values.ravel(), top_df_truth.values.ravel())
    top_nrmse = top_nrmse / len(selected_cells)
    print(
        'Scaffold_TopK File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, top_mse,
                                                                                     top_mae,
                                                                                     top_nrmse))
    topk_mse = metrics.mean_squared_error(topk_df_pred.values.ravel(), topk_df_truth.values.ravel())
    topk_mae = metrics.mean_absolute_error(topk_df_pred.values.ravel(), topk_df_truth.values.ravel())
    topk_nrmse = topk_nrmse / len(selected_cells)
    print('Scaffold_TopKG File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type,
                                                                                            topk_mse, topk_mae,
                                                                                            topk_nrmse))
    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    nrmse = nrmse / len(selected_cells)
    print('Scaffold_TopKG.p File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, mse,
                                                                                              mae,
                                                                                              nrmse))

    plt.figure()
    plt.plot(range(len(fed_loss_hist)), fed_loss_hist, label='FedAvg')
    plt.plot(range(len(s_loss_hist)), s_loss_hist, label='Scaffold')
    plt.plot(range(len(top_loss_hist)), top_loss_hist, label='Scaffold-TopK')
    plt.plot(range(len(topk_loss_hist)), topk_loss_hist, label='Scaffold-TopKG')
    plt.plot(range(len(loss_hist)), loss_hist, label='Scaffold-TopKG.p')
    plt.ylabel('train_loss')
    plt.xlabel('epochs')
    plt.legend(loc='best')
    plt.savefig(
        './save/experimental_comparison_{}.png'.format('train_loss'))