import copy
import math
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics

torch.manual_seed(2020)
np.random.seed(2020)


class LocalUpdate(object):
    def __init__(self, args, train, test):
        self.args = args
        self.train_loader = self.process_data(train)
        self.test_loader = self.process_data(test)
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.MSELoss().to(self.device)

    def process_data(self, dataset):
        data = list(zip(*dataset))
        if self.args.fedsgd == 1:
            loader = DataLoader(data, shuffle=False, batch_size=len(data))
        else:
            loader = DataLoader(data, shuffle=False, batch_size=self.args.local_bs)
        return loader

    def update_weights_A(self, model, global_round, control_local, control_global, w_loc, c_local_epoch):
        global_weights = copy.deepcopy(model.state_dict())
        control_global_w = copy.deepcopy(control_global.state_dict())
        control_local_w = control_local.state_dict()
        count = 0
        model.train()
        epoch_loss = []

        d_weight = copy.deepcopy(model.state_dict())
        w_weight = copy.deepcopy(w_loc)

        lr = self.args.lr

        if self.args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=self.args.momentum)

        for iter in range(self.args.local_epoch):
            batch_loss = []

            for batch_idx, (xc, xp, y) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)

                local_delta = copy.deepcopy(model.state_dict())
                local_weight = model.state_dict()
                for w in local_delta:
                    local_delta[w]=local_weight[w]- local_weight[w]
                model.zero_grad()
                pred = model(xc, xp)

                loss = self.criterion(y, pred)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
                optimizer.step()

                batch_loss.append(loss.item())

                local_weight = model.state_dict()
                for w in local_weight:
                    local_weight[w] = local_weight[w] - self.args.lr * (control_global_w[w] - control_local_w[w])
                model.load_state_dict(local_weight)
                count += 1

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        local_weight = model.state_dict()
        for w in local_weight:
            local_delta[w] = (local_weight[w] - global_weights[w])/self.train_loader.batch_size
            w_weight[w] += local_delta[w]

        if global_round != 99:
            if c_local_epoch % 2 == 0:
                for w in local_weight:
                    d_weight[w] = d_weight[w] - d_weight[w]
                    d_weight[w] += w_weight[w]
                    w_weight[w] -= d_weight[w]
            else:
                for w in local_weight:
                    d_weight[w] = d_weight[w] - d_weight[w]
                    sample_ratio = 0.01
                    compress_ratio = 0.01
                    numel = local_weight[w].numel()
                    num_samples = numel
                    # pct_numel = int(math.ceil(numel * sample_ratio))
                    # cpr_numel = int(math.ceil(2 / compress_ratio))
                    top_k_samples = int(math.ceil(num_samples * compress_ratio))
                    num_selects = int(math.ceil(numel * compress_ratio))
                    a = local_weight[w].shape
                    # a = tensor
                    tensor = w_weight[w].view(-1)
                    importance = tensor.abs()
                    threshold = torch.min(torch.topk(importance, top_k_samples, 0, largest=True, sorted=False)[0])
                    mask = torch.ge(importance, threshold)
                    indices = mask.nonzero(as_tuple =False).view(-1)
                    num_indices = indices.numel()
                    indices = indices[:num_selects]
                    values = tensor[indices]
                    d_values = d_weight[w].view(-1)
                    d_values[indices] = values
                    shape = local_weight[w].shape
                    num_shape = shape.numel()
                    d_weight[w] = d_values.reshape(a)

                    #d_weight:筛选交换的权重变化
                    #w_weight:筛选剩下的权重变化
                    w_weight[w] -= d_weight[w]

                    # d_weight[w].reshape((local_weight[w].shape[0], local_weight[w].shape[1]))
        else:
            d_weight = w_weight


        new_control_local_w = control_local.state_dict()
        control_delta = copy.deepcopy(control_local_w)
        net_weights = model.state_dict()
        local_delta = copy.deepcopy(net_weights)
        for w in net_weights:
            new_control_local_w[w] = new_control_local_w[w] - control_global_w[w] + (
                        global_weights[w] - net_weights[w]) / (count * self.args.lr)
            control_delta[w] = new_control_local_w[w] - control_local_w[w]
            local_delta[w] -= global_weights[w]


        #grad_np
        weight_diff = copy.deepcopy(net_weights)
        for k in model.state_dict():
            weight_diff[k] = net_weights[k] - global_weights[k]

        # grad_np = []
        # a = [v.view(-1).cpu().numpy() for k, v in weight_diff.items()]
        # for l in range(len(a)):
        #     grad_np = np.concatenate([grad_np, a[l]], axis=0)
        for k in d_weight:
            d_weight[k] /= self.train_loader.batch_size
        grad_np_dgc = []
        a = [v.view(-1).cpu().numpy() for k, v in d_weight.items()]
        for l in range(len(a)):
            grad_np_dgc = np.concatenate([grad_np_dgc, a[l]], axis=0)


        return model.state_dict(), sum(epoch_loss)/len(epoch_loss), epoch_loss, control_delta, d_weight, new_control_local_w,  w_weight, grad_np_dgc, c_local_epoch

    def update_weights_topkG(self, model, global_round, control_local, control_global, w_loc):
        global_weights = copy.deepcopy(model.state_dict())
        control_global_w = copy.deepcopy(control_global.state_dict())
        control_local_w = control_local.state_dict()
        count = 0
        model.train()
        epoch_loss = []

        d_weight = copy.deepcopy(model.state_dict())
        w_weight = copy.deepcopy(w_loc)

        lr = self.args.lr

        if self.args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=self.args.momentum)

        for iter in range(self.args.local_epoch):
            batch_loss = []

            for batch_idx, (xc, xp, y) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)

                local_delta = copy.deepcopy(model.state_dict())
                local_weight = model.state_dict()
                for w in local_delta:
                    local_delta[w]=local_weight[w]- local_weight[w]
                model.zero_grad()
                pred = model(xc, xp)

                loss = self.criterion(y, pred)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
                optimizer.step()

                batch_loss.append(loss.item())

                local_weight = model.state_dict()
                for w in local_weight:
                    local_weight[w] = local_weight[w] - self.args.lr * (control_global_w[w] - control_local_w[w])
                model.load_state_dict(local_weight)
                count += 1

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        local_weight = model.state_dict()
        for w in local_weight:
            local_delta[w] = (local_weight[w] - global_weights[w])/self.train_loader.batch_size
            w_weight[w] += local_delta[w]

        if global_round != 99 :
            for w in local_weight:
                d_weight[w] = d_weight[w] - d_weight[w]
                sample_ratio = 0.01
                compress_ratio = 0.01
                numel = local_weight[w].numel()
                num_samples = numel
                # pct_numel = int(math.ceil(numel * sample_ratio))
                # cpr_numel = int(math.ceil(2 / compress_ratio))
                top_k_samples = int(math.ceil(num_samples * compress_ratio))
                num_selects = int(math.ceil(numel * compress_ratio))
                a = local_weight[w].shape
                # a = tensor
                tensor = w_weight[w].view(-1)
                importance = tensor.abs()
                threshold = torch.min(torch.topk(importance, top_k_samples, 0, largest=True, sorted=False)[0])
                mask = torch.ge(importance, threshold)
                indices = mask.nonzero(as_tuple =False).view(-1)
                num_indices = indices.numel()
                indices = indices[:num_selects]
                values = tensor[indices]
                d_values = d_weight[w].view(-1)
                d_values[indices] = values
                shape = local_weight[w].shape
                num_shape = shape.numel()
                d_weight[w] = d_values.reshape(a)
                w_weight[w] -= d_weight[w]
                # print(w_weight)

                # d_weight[w].reshape((local_weight[w].shape[0], local_weight[w].shape[1]))
        else:
            d_weight = w_weight


        new_control_local_w = control_local.state_dict()
        control_delta = copy.deepcopy(control_local_w)
        net_weights = model.state_dict()
        local_delta = copy.deepcopy(net_weights)
        for w in net_weights:
            new_control_local_w[w] = new_control_local_w[w] - control_global_w[w] + (
                        global_weights[w] - net_weights[w]) / (count * self.args.lr)
            control_delta[w] = new_control_local_w[w] - control_local_w[w]
            local_delta[w] -= global_weights[w]


        #grad_np
        weight_diff = copy.deepcopy(net_weights)
        for k in model.state_dict():
            weight_diff[k] = net_weights[k] - global_weights[k]

        # grad_np = []
        # a = [v.view(-1).cpu().numpy() for k, v in weight_diff.items()]
        # for l in range(len(a)):
        #     grad_np = np.concatenate([grad_np, a[l]], axis=0)
        for k in d_weight:
            d_weight[k] /= self.train_loader.batch_size
        grad_np_dgc = []
        a = [v.view(-1).cpu().numpy() for k, v in d_weight.items()]
        for l in range(len(a)):
            grad_np_dgc = np.concatenate([grad_np_dgc, a[l]], axis=0)

        return model.state_dict(), sum(epoch_loss)/len(epoch_loss), epoch_loss, control_delta, \
               d_weight, new_control_local_w,  w_weight, grad_np_dgc

    def update_weights_fed(self, model, global_round):
        model.train()
        epoch_loss = []
        lr = self.args.lr

        if self.args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=self.args.momentum)

        for iter in range(self.args.local_epoch):
            batch_loss = []

            for batch_idx, (xc, xp, y) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)

                model.zero_grad()
                pred = model(xc, xp)

                loss = self.criterion(y, pred)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss)/len(epoch_loss), epoch_loss

    def update_weights_scaffold(self, model, control_local, control_global):
        global_weights = copy.deepcopy(model.state_dict())
        control_global_w = copy.deepcopy(control_global.state_dict())
        control_local_w = control_local.state_dict()
        count = 0
        model.train()
        epoch_loss = []

        d_weight = copy.deepcopy(model.state_dict())

        lr = self.args.lr

        if self.args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=self.args.momentum)

        for iter in range(self.args.local_epoch):
            batch_loss = []

            for batch_idx, (xc, xp, y) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)

                local_delta = copy.deepcopy(model.state_dict())
                local_weight = model.state_dict()
                for w in local_delta:
                    local_delta[w]=local_weight[w]- local_weight[w]
                model.zero_grad()
                pred = model(xc, xp)

                loss = self.criterion(y, pred)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
                optimizer.step()

                batch_loss.append(loss.item())

                local_weight = model.state_dict()
                for w in local_weight:
                    local_weight[w] = local_weight[w] - self.args.lr * (control_global_w[w] - control_local_w[w])
                model.load_state_dict(local_weight)
                count += 1

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        local_weight = model.state_dict()
        for w in local_weight:
            local_delta[w] = (local_weight[w] - global_weights[w])/count


        new_control_local_w = control_local.state_dict()
        control_delta = copy.deepcopy(control_local_w)
        net_weights = model.state_dict()
        local_delta = copy.deepcopy(net_weights)
        for w in net_weights:
            new_control_local_w[w] = new_control_local_w[w] - control_global_w[w] + (
                        global_weights[w] - net_weights[w]) / (count * self.args.lr)
            control_delta[w] = new_control_local_w[w] - control_local_w[w]
            local_delta[w] -= global_weights[w]

        return model.state_dict(), sum(epoch_loss)/len(epoch_loss), epoch_loss, control_delta, \
               local_delta, new_control_local_w

    def update_weights_topk(self, model, global_round, control_local, control_global, w_loc):
        global_weights = copy.deepcopy(model.state_dict())
        control_global_w = copy.deepcopy(control_global.state_dict())
        control_local_w = control_local.state_dict()
        count = 0
        model.train()
        epoch_loss = []

        d_weight = copy.deepcopy(model.state_dict())
        w_weight = copy.deepcopy(w_loc)

        lr = self.args.lr

        if self.args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=self.args.momentum)

        for iter in range(self.args.local_epoch):
            batch_loss = []

            for batch_idx, (xc, xp, y) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)

                local_delta = copy.deepcopy(model.state_dict())
                local_weight = model.state_dict()
                for w in local_delta:
                    local_delta[w]=local_weight[w]- local_weight[w]
                model.zero_grad()
                pred = model(xc, xp)

                loss = self.criterion(y, pred)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
                optimizer.step()

                batch_loss.append(loss.item())

                local_weight = model.state_dict()
                for w in local_weight:
                    local_weight[w] = local_weight[w] - self.args.lr * (control_global_w[w] - control_local_w[w])
                model.load_state_dict(local_weight)
                count += 1

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        local_weight = model.state_dict()
        for w in local_weight:
            local_delta[w] = (local_weight[w] - global_weights[w])/self.train_loader.batch_size
            w_weight[w] += local_delta[w]

        if global_round != 99:
            for w in local_weight:
                d_weight[w] = d_weight[w] - d_weight[w]
                sample_ratio = 0.01
                compress_ratio = 0.2
                numel = local_weight[w].numel()
                num_samples = numel
                # pct_numel = int(math.ceil(numel * sample_ratio))
                # cpr_numel = int(math.ceil(2 / compress_ratio))
                top_k_samples = int(math.ceil(num_samples * compress_ratio))
                num_selects = int(math.ceil(numel * compress_ratio))
                a = local_weight[w].shape
                # a = tensor
                tensor = w_weight[w].view(-1)
                importance = tensor.abs()
                threshold = torch.min(torch.topk(importance, top_k_samples, 0, largest=True, sorted=False)[0])
                mask = torch.ge(importance, threshold)
                indices = mask.nonzero(as_tuple =False).view(-1)
                num_indices = indices.numel()
                indices = indices[:num_selects]
                values = tensor[indices]
                d_values = d_weight[w].view(-1)
                d_values[indices] = values
                shape = local_weight[w].shape
                num_shape = shape.numel()
                d_weight[w] = d_values.reshape(a)
                w_weight[w] -= d_weight[w]
        else:
            d_weight = w_weight


        new_control_local_w = control_local.state_dict()
        control_delta = copy.deepcopy(control_local_w)
        net_weights = model.state_dict()
        local_delta = copy.deepcopy(net_weights)
        for w in net_weights:
            new_control_local_w[w] = new_control_local_w[w] - control_global_w[w] + (
                        global_weights[w] - net_weights[w]) / (count * self.args.lr)
            control_delta[w] = new_control_local_w[w] - control_local_w[w]
            local_delta[w] -= global_weights[w]

        return model.state_dict(), sum(epoch_loss)/len(epoch_loss), epoch_loss, control_delta, \
               d_weight, new_control_local_w,  w_weight

    def update_weights_GCAU(self, model, global_round, control_local, control_global, w_loc, c_local_epoch):
        global_weights = copy.deepcopy(model.state_dict())
        control_global_w = copy.deepcopy(control_global.state_dict())
        control_local_w = control_local.state_dict()
        count = 0
        model.train()
        epoch_loss = []

        d_weight = copy.deepcopy(model.state_dict())
        w_weight = copy.deepcopy(w_loc)

        lr = self.args.lr

        if self.args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=self.args.momentum)

        for iter in range(self.args.local_epoch):
            batch_loss = []

            for batch_idx, (xc, xp, y) in enumerate(self.train_loader):
                xc, xp = xc.float().to(self.device), xp.float().to(self.device)
                y = y.float().to(self.device)

                local_delta = copy.deepcopy(model.state_dict())
                local_weight = model.state_dict()
                for w in local_delta:
                    local_delta[w] = local_weight[w] - local_weight[w]
                model.zero_grad()
                pred = model(xc, xp)

                loss = self.criterion(y, pred)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
                optimizer.step()

                batch_loss.append(loss.item())

                local_weight = model.state_dict()
                for w in local_weight:
                    local_weight[w] = local_weight[w] - self.args.lr * (control_global_w[w] - control_local_w[w])
                model.load_state_dict(local_weight)
                count += 1

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        local_weight = model.state_dict()
        for w in local_weight:
            local_delta[w] = (local_weight[w] - global_weights[w]) / self.train_loader.batch_size
            w_weight[w] += local_delta[w]

        if global_round != 99:
            if c_local_epoch % 2 == 0:
                for w in local_weight:
                    d_weight[w] = d_weight[w] - d_weight[w]
                    d_weight[w] += w_weight[w]
                    w_weight[w] -= d_weight[w]
            else:
                for w in local_weight:
                    d_weight[w] = d_weight[w] - d_weight[w]
                    sample_ratio = 0.01
                    compress_ratio = 0.01
                    numel = local_weight[w].numel()
                    num_samples = numel
                    # pct_numel = int(math.ceil(numel * sample_ratio))
                    # cpr_numel = int(math.ceil(2 / compress_ratio))
                    top_k_samples = int(math.ceil(num_samples * compress_ratio))
                    num_selects = int(math.ceil(numel * compress_ratio))
                    a = local_weight[w].shape
                    # a = tensor
                    tensor = w_weight[w].view(-1)
                    importance = tensor.abs()
                    threshold = torch.min(torch.topk(importance, top_k_samples, 0, largest=True, sorted=False)[0])
                    mask = torch.ge(importance, threshold)
                    indices = mask.nonzero(as_tuple=False).view(-1)
                    num_indices = indices.numel()
                    indices = indices[:num_selects]
                    values = tensor[indices]
                    d_values = d_weight[w].view(-1)
                    d_values[indices] = values
                    shape = local_weight[w].shape
                    num_shape = shape.numel()
                    d_weight[w] = d_values.reshape(a)

                    # d_weight:筛选交换的权重变化
                    # w_weight:筛选剩下的权重变化
                    w_weight[w] -= d_weight[w]

                    # d_weight[w].reshape((local_weight[w].shape[0], local_weight[w].shape[1]))
        else:
            d_weight = w_weight

        new_control_local_w = control_local.state_dict()
        control_delta = copy.deepcopy(control_local_w)
        net_weights = model.state_dict()
        local_delta = copy.deepcopy(net_weights)
        for w in net_weights:
            new_control_local_w[w] = new_control_local_w[w] - control_global_w[w] + (
                    global_weights[w] - net_weights[w]) / (count * self.args.lr)
            control_delta[w] = new_control_local_w[w] - control_local_w[w]
            local_delta[w] -= global_weights[w]

        for k in d_weight:
            d_weight[k] /= self.train_loader.batch_size

        return model.state_dict(), sum(epoch_loss) / len(
            epoch_loss), epoch_loss, control_delta, d_weight, new_control_local_w, w_weight,  c_local_epoch

def test_inference(args, model, dataset):
    model.eval()
    loss, mse = 0.0, 0.0
    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.MSELoss().to(device)
    data_loader = DataLoader(list(zip(*dataset)), batch_size=args.local_bs, shuffle=False)
    pred_list, truth_list = [], []

    with torch.no_grad():
        for batch_idx, (xc, xp, y) in enumerate(data_loader):
            xc, xp = xc.float().to(device), xp.float().to(device)
            y = y.float().to(device)
            pred = model(xc, xp)

            batch_loss = criterion(y, pred)
            loss += batch_loss.item()

            batch_mse = torch.mean((pred - y) ** 2)
            mse += batch_mse.item()

            pred_list.append(pred.detach().cpu())
            truth_list.append(y.detach().cpu())

    final_prediction = np.concatenate(pred_list).ravel()
    final_truth = np.concatenate(truth_list).ravel()
    nrmse = (metrics.mean_squared_error(final_truth, final_prediction) ** 0.5) / (
                max(final_truth) - min(final_truth))
    avg_loss = loss / len(data_loader)
    avg_mse = mse / len(data_loader)

    return avg_loss, avg_mse, nrmse, final_prediction, final_truth