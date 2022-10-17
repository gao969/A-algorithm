

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
from utils.models import MLP
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

    fed_global_model = MLP(args).to(device)
    fed_global_model.train()
    top_global_model = MLP(args).to(device)
    top_control_global = MLP(args).to(device)
    top_global_model.train()

    fed_global_weights = fed_global_model.state_dict()
    top_control_weights = top_control_global.state_dict()
    top_global_weights = copy.deepcopy(top_global_model.state_dict())
    top_c_glob = copy.deepcopy(top_control_weights)

    top_c_local = {}
    for idx in selected_cells:
        top_c_local[idx] = MLP(args).to(device)
        top_c_local[idx].load_state_dict(top_control_weights)
    top_delta_c = copy.deepcopy(top_global_model.state_dict())

    top_weight = copy.deepcopy(top_global_weights)
    for w in top_global_weights:
        top_weight[w] = top_global_weights[w] - top_global_weights[w]

    top_ww = copy.deepcopy(top_weight)
    top_w_locals = {}
    for idx in selected_cells:
        top_w_locals[idx] = copy.deepcopy(top_weight)

    fed_best_val_loss = None
    fed_val_loss = []
    fed_val_acc = []
    fed_cell_loss = []
    fed_loss_hist = []
    top_best_val_loss = None
    top_val_loss = []
    top_val_acc = []
    top_cell_loss = []
    top_loss_hist = []


    for epoch in tqdm.tqdm(range(args.epochs)):
        fed_local_weights, fed_local_losses = [], []
        fed_global_model.train()
        top_local_weights, top_local_losses = [], []
        # local_grad_np = []
        for i in top_delta_c:
            top_delta_c[i] = 0.0
        # print(f'\n | Global Training Round: {epoch+1} |\n')
        top_global_model.train()

        m = max(int(args.frac * args.bs), 1)
        cell_idx = random.sample(selected_cells, m)
        # print(cell_idx)

        for cell in cell_idx:
            cell_train, cell_test = train[cell], test[cell]
            local_model = LocalUpdate(args, cell_train, cell_test)

            fed_global_model.load_state_dict(fed_global_weights)
            fed_global_model.train()
            top_global_model.load_state_dict(top_global_weights)
            top_global_model.train()

            fed_w, fed_loss, fed_epoch_loss = local_model.update_weights_fed(model=copy.deepcopy(fed_global_model),
                                                                         global_round=epoch)
            top_w, top_loss, top_epoch_loss, top_local_delta_c, top_local_delta, top_control_local_w, top_ww = local_model.update_weights_topkg(
                model=copy.deepcopy(top_global_model), global_round=epoch,
                control_local=copy.deepcopy(top_c_local[cell]), control_global=copy.deepcopy(top_control_global), w_loc=top_w_locals[cell])

            if epoch != 0:
                top_c_local[cell].load_state_dict(top_control_local_w)

            # local_weights.append(copy.deepcopy(w))
            fed_local_weights.append(copy.deepcopy(fed_w))
            fed_local_losses.append(copy.deepcopy(fed_loss))
            fed_cell_loss.append(fed_loss)
            top_local_weights.append(copy.deepcopy(top_local_delta))
            top_local_losses.append(copy.deepcopy(top_loss))
            top_cell_loss.append(top_loss)

            for i in top_delta_c:
                top_delta_c[i] += top_local_delta_c[i]
            for k in top_ww:
                top_w_locals[cell][k] = top_ww[k]

        fed_loss_hist.append(sum(fed_cell_loss) / len(fed_cell_loss))
        top_loss_hist.append(sum(top_cell_loss) / len(top_cell_loss))

        for i in top_delta_c:
            top_delta_c[i] /= m

        # Update global model
        top_global_weights = copy.deepcopy(top_global_model.state_dict())
        for k in top_global_weights:
            top_global_weights[k] += average_weights(top_local_weights)[k]
        top_control_global_w = copy.deepcopy(top_control_global.state_dict())

        control_global_w = top_control_global.state_dict()
        for i in control_global_w:
            top_control_global_w[i] += (m / len(selected_cells)) * top_delta_c[i]
            # control_global_w[i] += (m / args.num_users) * delta_c[i]

        fed_global_weights = average_weights(fed_local_weights)
        fed_global_model.load_state_dict(fed_global_weights)
        top_global_model.load_state_dict(top_global_weights)
        top_control_global.load_state_dict(top_control_global_w)


    # Test model accuracy
    fed_pred, fed_truth = {}, {}
    fed_test_loss_list = []
    fed_test_mse_list = []
    fed_nrmse = 0.0
    top_pred, top_truth = {}, {}
    top_test_lostop_list = []
    top_test_mse_list = []
    top_nrmse = 0.0

    fed_global_model.load_state_dict(fed_global_weights)
    top_global_model.load_state_dict(top_global_weights)

    for cell in selected_cells:
        cell_test = test[cell]
        fed_test_loss, fed_test_mse, fed_test_nrmse, fed_pred[cell], fed_truth[cell] = test_inference(args, fed_global_model, cell_test)
        fed_nrmse += fed_test_nrmse
        top_test_loss, top_test_mse, top_test_nrmse, top_pred[cell], top_truth[cell] = test_inference(args, top_global_model, cell_test)
        top_nrmse += top_test_nrmse

        fed_test_loss_list.append(fed_test_loss)
        fed_test_mse_list.append(fed_test_mse)
        top_test_lostop_list.append(top_test_loss)
        top_test_mse_list.append(top_test_mse)

    fed_df_pred = pd.DataFrame.from_dict(fed_pred)
    fed_df_truth = pd.DataFrame.from_dict(fed_truth)
    top_df_pred = pd.DataFrame.from_dict(top_pred)
    top_df_truth = pd.DataFrame.from_dict(top_truth)


    fed_mse = metrics.mean_squared_error(fed_df_pred.values.ravel(), fed_df_truth.values.ravel())
    fed_mae = metrics.mean_absolute_error(fed_df_pred.values.ravel(), fed_df_truth.values.ravel())
    fed_nrmse = fed_nrmse / len(selected_cells)
    print('FedAvg File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, fed_mse,
                                                                                     fed_mae,
                                                                                     fed_nrmse))
    top_mse = metrics.mean_squared_error(top_df_pred.values.ravel(), top_df_truth.values.ravel())
    top_mae = metrics.mean_absolute_error(top_df_pred.values.ravel(), top_df_truth.values.ravel())
    top_nrmse = top_nrmse / len(selected_cells)
    print(
        'Scaffold File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}, NRMSE: {:.4f}'.format(args.file, args.type, top_mse, top_mae,
                                                                                     top_nrmse))

    plt.figure()
    plt.plot(range(len(fed_loss_hist)), fed_loss_hist, label='FedAvg')
    plt.plot(range(len(top_loss_hist)), top_loss_hist, label='Scaffold')
    plt.ylabel('train_loss')
    plt.xlabel('epochs')
    plt.legend(loc='best')
    plt.savefig(
        './save/t_scaffold{}.png'.format('train_loss'))