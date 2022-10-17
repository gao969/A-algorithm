

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
from torch.nn import init

sys.path.append('../')
# DualFedAtt.
from utils.misc import args_parser, average_weights
from utils.misc import get_data, process_isolated, initialize_parameters_zeros
from utils.models import MLP
from utils.fed_update import LocalUpdate, test_inference
from sklearn import metrics

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(5)
    np.random.seed(args.seed)
    data, _, selected_cells, mean, std, _, _ = get_data(args)

    device = 'cuda' if args.gpu else 'cpu'

    parameter_list = 'FedAvg-data-{:}-type-{:}-'.format(args.file, args.type)
    parameter_list += '-frac-{:.2f}-le-{:}-lb-{:}-seed-{:}'.format(args.frac, args.local_epoch,
                                                                   args.local_bs,
                                                                   args.seed)
    log_id = args.directory + parameter_list
    train, val, test = process_isolated(args, data)



    #初始化模型
    fed_global_model = MLP(args).to(device)
    s_global_model = MLP(args).to(device)
    sg_global_model = MLP(args).to(device)

    fed_global_model.train()
    s_global_model.train()
    sg_global_model.train()

    fed_global_weights = fed_global_model.state_dict()
    s_global_weights = s_global_model.state_dict()
    sg_global_weights = sg_global_model.state_dict()


    #初始化控制变量
    s_control_global = MLP(args).to(device)
    sg_control_global = MLP(args).to(device)

    s_control_weights = initialize_parameters_zeros(s_control_global)
    sg_control_weights = initialize_parameters_zeros(sg_control_global)

    s_c_local = {}
    sg_c_local = {}
    for idx in selected_cells:
        s_c_local[idx] = MLP(args).to(device)
        s_c_local[idx].load_state_dict(s_control_weights)
        sg_c_local[idx] = MLP(args).to(device)
        sg_c_local[idx].load_state_dict(sg_control_weights)

    s_delta_c = copy.deepcopy(s_control_global.state_dict())
    sg_delta_c = copy.deepcopy(sg_control_global.state_dict())


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

    sg_best_val_loss = None
    sg_val_loss = []
    sg_val_acc = []
    sg_cell_loss = []
    sg_loss_hist = []


    for epoch in tqdm.tqdm(range(args.epochs)):
        fed_local_weights, fed_local_losses = [], []
        s_local_weights, s_local_losses = [], []
        sg_local_weights, sg_local_losses = [], []
        sg_local_pear_data = []

        for i in s_delta_c:
            s_delta_c[i] = 0.0
            sg_delta_c[i] = 0.0

        m = max(int(args.frac * args.bs), 1)
        cell_idx = random.sample(selected_cells, m)

        for cell in cell_idx:
            cell_train, cell_test = train[cell], test[cell]
            local_model = LocalUpdate(args, cell_train, cell_test)

            fed_global_model.load_state_dict(fed_global_weights)
            fed_global_model.train()
            s_global_model.load_state_dict(s_global_weights)
            s_global_model.train()
            sg_global_model.load_state_dict(sg_global_weights)
            sg_global_model.train()

            s_w, s_loss, s_epoch_loss, s_local_delta_c, s_local_delta, s_control_local_w = local_model.update_weights_scaffold(
                model=copy.deepcopy(s_global_model), control_local=copy.deepcopy(s_c_local[cell]),
                control_global=copy.deepcopy(s_control_global))

            sg_w, sg_loss, sg_epoch_loss, sg_local_delta_c, sg_local_delta, sg_control_local_w, sg_pear_data = local_model.update_weights_sg(
                model=copy.deepcopy(sg_global_model), global_round=epoch, control_local=sg_c_local[cell],
                control_global=sg_control_global, )

            fed_w, fed_loss, fed_epoch_loss = local_model.update_weights_fed(model=copy.deepcopy(fed_global_model),
                                                                             global_round=epoch)

            s_c_local[cell].load_state_dict(s_control_local_w)
            sg_c_local[cell].load_state_dict(sg_control_local_w)


            # local_weights.append(copy.deepcopy(w))
            fed_local_weights.append(copy.deepcopy(fed_w))
            fed_local_losses.append(copy.deepcopy(fed_loss))
            fed_cell_loss.append(fed_loss)
            s_local_weights.append(copy.deepcopy(s_w))
            s_local_losses.append(copy.deepcopy(s_loss))
            s_cell_loss.append(s_loss)
            sg_local_weights.append(copy.deepcopy(sg_w))
            sg_local_losses.append(copy.deepcopy(sg_loss))
            sg_cell_loss.append(sg_loss)
            sg_local_pear_data.append(sg_pear_data)

            for i in s_delta_c:
                s_delta_c[i] += s_local_delta_c[i]
                sg_delta_c[i] += s_local_delta_c[i]

        fed_loss_hist.append(sum(fed_cell_loss) / len(fed_cell_loss))
        s_loss_hist.append(sum(s_cell_loss) / len(s_cell_loss))
        sg_loss_hist.append(sum(sg_cell_loss) / len(sg_cell_loss))

        for i in s_delta_c:
            s_delta_c[i] /= m
            sg_delta_c[i] /= m


        #pearson
        sg_pearson = []
        for l in range(len(sg_local_pear_data)):
            data = sg_local_pear_data[l]
            sg_pearson = np.concatenate([sg_pearson, sg_local_pear_data[l]])
        sg_pearson = sg_pearson.reshape(m, -1)
        sg_pearson = np.corrcoef(sg_pearson)

        #weighted
        sg_global = np.dot(sg_pearson, sg_local_pear_data)
        a = np.zeros_like(sg_pear_data)
        for i in range(len(sg_global)):
            a += sg_global[i]
        a /= m
        new_sg_global_weights = {}
        b = []
        count = 0
        lay_shape = {}
        for k in sg_global_weights:
            lay_shape[k] = torch.ones(1)
            for i in range(len(sg_global_weights[k].shape)):
                lay_shape[k] *= sg_global_weights[k].shape[i]
        for k in lay_shape:
            c = int(lay_shape[k].item())
            b = a[count:count + c]
            d = torch.from_numpy(b)
            new_sg_global_weights[k] = d.reshape(sg_global_weights[k].shape)
            count += c




        # Update global model
        fed_global_weights = average_weights(fed_local_weights)
        s_global_weights = average_weights(s_local_weights)
        for k in new_sg_global_weights:
            sg_global_weights[k] = new_sg_global_weights[k] + sg_global_weights[k]

        fed_global_model.load_state_dict(fed_global_weights)
        s_global_model.load_state_dict(s_global_weights)
        sg_global_model.load_state_dict(sg_global_weights)


        #Update global c
        s_control_global_w = copy.deepcopy(s_control_global.state_dict())
        sg_control_global_w = copy.deepcopy(s_control_global.state_dict())
        control_global_w = s_control_global.state_dict()
        for i in control_global_w:
            s_control_global_w[i] += (m / len(selected_cells)) * s_delta_c[i]
            sg_control_global_w[i] += (m / len(selected_cells)) * sg_delta_c[i]
            # control_global_w[i] += (m / args.num_users) * delta_c[i]
        s_control_global.load_state_dict(s_control_global_w)
        sg_control_global.load_state_dict(sg_control_global_w)


    # Test model accuracy
    fed_pred, fed_truth = {}, {}
    fed_test_loss_list = []
    fed_test_mse_list = []
    fed_nrmse = 0.0

    s_pred, s_truth = {}, {}
    s_test_loss_list = []
    s_test_mse_list = []
    s_nrmse = 0.0

    sg_pred, sg_truth = {}, {}
    sg_test_loss_list = []
    sg_test_mse_list = []
    sg_nrmse = 0.0



    for cell in selected_cells:
        cell_test = test[cell]
        fed_test_loss, fed_test_mse, fed_test_nrmse, fed_pred[cell], fed_truth[cell] = test_inference(args, fed_global_model, cell_test)
        fed_nrmse += fed_test_nrmse

        s_test_loss, s_test_mse, s_test_nrmse, s_pred[cell], s_truth[cell] = test_inference(args, s_global_model, cell_test)
        s_nrmse += s_test_nrmse

        sg_test_loss, sg_test_mse, sg_test_nrmse, sg_pred[cell], sg_truth[cell] = test_inference(args, sg_global_model,cell_test)
        sg_nrmse += sg_test_nrmse

        fed_test_loss_list.append(fed_test_loss)
        fed_test_mse_list.append(fed_test_mse)
        s_test_loss_list.append(s_test_loss)
        s_test_mse_list.append(s_test_mse)
        sg_test_loss_list.append(sg_test_loss)
        sg_test_mse_list.append(sg_test_mse)

    fed_df_pred = pd.DataFrame.from_dict(fed_pred)
    fed_df_truth = pd.DataFrame.from_dict(fed_truth)
    s_df_pred = pd.DataFrame.from_dict(s_pred)
    s_df_truth = pd.DataFrame.from_dict(s_truth)
    sg_df_pred = pd.DataFrame.from_dict(sg_pred)
    sg_df_truth = pd.DataFrame.from_dict(sg_truth)


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
    sg_mse = metrics.mean_squared_error(sg_df_pred.values.ravel(), sg_df_truth.values.ravel())
    sg_mae = metrics.mean_absolute_error(sg_df_pred.values.ravel(), sg_df_truth.values.ravel())
    sg_nrmse = sg_nrmse / len(selected_cells)
    print(
        'Scaffold File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, sg_mse, sg_mae,
                                                                                     sg_nrmse))

    plt.figure()
    plt.plot(range(len(fed_loss_hist)), fed_loss_hist, label='FedAvg')
    plt.plot(range(len(s_loss_hist)), s_loss_hist, label='Scaffold')
    plt.plot(range(len(sg_loss_hist)), sg_loss_hist, label='Scaffold-G')
    plt.ylabel('train_loss')
    plt.xlabel('epochs')
    plt.legend(loc='best')
    plt.savefig(
        './save/t_FedAvg_Scaffold_ScaffoldG_{}.png'.format('train_loss'))