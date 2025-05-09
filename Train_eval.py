from typing import Dict
import torch
from Data import Data
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from Scheduler import GradualWarmupScheduler
from Model import SpectralGroupAttention
import os
from Tools import checkFile, standard
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from ts_generation import ts_generation
from sklearn import metrics
import random
import time
from draw_a_b import draw_3d
# from torchinfo import summary
# from thop import profile
from draw import save_image
from ptflops import get_model_complexity_info

def seed_torch(seed=1):
    '''
    Keep the seed fixed thus the results can keep stable
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def spectral_group(x, n, m):
    ### divide the spectrum into n overlapping groups
    pad_size = m // 2
    new_sample = np.pad(x, ((0, 0), (pad_size, pad_size)),
                        mode='symmetric')
    b = x.shape[0]
    group_spectra = np.zeros([b, n, m])
    for i in range(n):
        group_spectra[:, i, :] = np.squeeze(new_sample[:, i:i + m])

    return torch.from_numpy(group_spectra).float()


def cosin_similarity(x, y):
    assert x.shape[1] == y.shape[1]
    x_norm = torch.sqrt(torch.sum(x ** 2, dim=1))
    y_norm = torch.sqrt(torch.sum(y ** 2, dim=1))
    x_y_dot = torch.sum(torch.multiply(x, y), dim=1)
    return x_y_dot / (x_norm * y_norm + 1e-8)


def cosin_similarity_numpy(x, y):
    assert x.shape[1] == y.shape[1]
    x_norm = np.sqrt(np.sum(x ** 2, axis=1))
    y_norm = np.sqrt(np.sum(y ** 2, axis=1))
    x_y = np.sum(np.multiply(x, y), axis=1)
    return x_y / (x_norm * y_norm + 1e-8)


def isia_loss(x, batch_size, margin=1.0, lambd=1):
    '''
    This function is used to calculate the intercategory separation and intracategory aggregation loss
    It includes the triplet loss and cross-entropy loss
    '''
    positive, negative, prior = x[:batch_size], x[batch_size:2 * batch_size], x[2 * batch_size:]
    p_sim = cosin_similarity(positive, prior)
    n_sim1 = cosin_similarity(negative, prior)
    n_sim2 = cosin_similarity(negative, positive)
    max_n_sim = torch.maximum(n_sim1, n_sim2)

    ## triplet loss to maximize the feature distance between anchor and positive samples
    ## while minimizing the feature distance between anchor and negative samples
    triplet_loss = margin + max_n_sim - p_sim
    triplet_loss = torch.relu(triplet_loss)
    triplet_loss = torch.mean(triplet_loss)

    ## binary cross-entropy loss to distinguish pixels of background and target
    p_sim = torch.sigmoid(p_sim)
    n_sim = torch.sigmoid(1 - n_sim1)
    bce_loss = -0.5 * torch.mean(torch.log(p_sim + 1e-8) + torch.log(n_sim + 1e-8))

    isia_loss = triplet_loss + lambd * bce_loss

    return isia_loss
#   这个改法效果不好。
# def isia_loss(x, batch_size, margin=1.0, lambd=1):
#     '''
#     Intercategory Separation and Intracategory Aggregation Loss
#     包含 Hard Negative Mining 的 Triplet Loss 和原有 BCE Loss
#     '''
#     positive = x[:batch_size]          # 正样本
#     negative = x[batch_size:2 * batch_size]  # 负样本
#     prior = x[2 * batch_size:]         # 锚点样本
#
#     # ---- Hard Negative Mining ----
#     # 计算 Prior 与 Negative 全部两两相似度矩阵
#     prior_norm = prior / (prior.norm(dim=1, keepdim=True) + 1e-8)        # shape: [B, C]
#     negative_norm = negative / (negative.norm(dim=1, keepdim=True) + 1e-8)  # shape: [B, C]
#     sim_matrix = torch.matmul(prior_norm, negative_norm.T)               # shape: [B, B]
#
#     # 每个 prior 找到最相似的负样本，得到 Hard Negative
#     max_sim_values, max_indices = torch.max(sim_matrix, dim=1)           # shape: [B]
#     hard_negatives = negative[max_indices]                               # shape: [B, C]
#
#     # ---- Triplet Loss ----
#     # 正样本与 Prior 的相似度
#     p_sim = cosin_similarity(positive, prior)                            # shape: [B]
#
#     # Hard Negative 与 Prior 的相似度
#     n_sim = cosin_similarity(hard_negatives, prior)                      # shape: [B]
#
#     # triplet loss: margin + hardest_negative - positive
#     triplet_loss = margin + n_sim - p_sim
#     triplet_loss = torch.relu(triplet_loss)
#     triplet_loss = torch.mean(triplet_loss)
#
#     # ---- BCE Loss (不变) ----
#     p_sim_sigmoid = torch.sigmoid(p_sim)       # 正样本预测概率
#     n_sim1 = cosin_similarity(negative, prior) # 负样本与prior相似度
#     n_sim_sigmoid = torch.sigmoid(1 - n_sim1)  # 负样本预测概率（越接近0越好）
#     bce_loss = -0.5 * torch.mean(torch.log(p_sim_sigmoid + 1e-8) + torch.log(n_sim_sigmoid + 1e-8))
#
#
#
#     # ---- 最终 Loss ----
#     isia_loss = triplet_loss + lambd * bce_loss
#
#     return isia_loss

def paintTrend(losslist, epochs=100, stride=10):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.title('loss-trend')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xticks(np.arange(0, epochs, stride))
    plt.xlim(0, epochs)
    plt.plot(losslist, color='r')
    plt.show()


def train(modelConfig: Dict):
    seed_torch(modelConfig['seed'])
    device = torch.device(modelConfig["device"])
    dataset = Data(modelConfig["path"],modelConfig["alphamax"],modelConfig["betamax"])
    dataloader = DataLoader(dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=16, drop_last=True,
                            pin_memory=True)
    # model setup
    net_model = SpectralGroupAttention(band=modelConfig['band'], m=modelConfig['group_length'],
                                       d=modelConfig['channel'], depth=modelConfig['depth'],heads=modelConfig['heads'],
                                       dim_head=modelConfig['dim_head'], mlp_dim=modelConfig['mlp_dim'], adjust=modelConfig['adjust']).to(device)


    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["training_load_weight"]), map_location=device), strict=False)
        print("Model weight load down.")
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    path = modelConfig["save_dir"] + '/' + modelConfig['path'] + '/'
    checkFile(path)

    # start training
    net_model.train()
    loss_list = []
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for positive, negative in tqdmDataLoader:
                # train
                combined_vectors = np.concatenate([positive, negative, dataset.target_spectrum], axis=0)
                combined_groups = spectral_group(combined_vectors, modelConfig['band'], modelConfig['group_length'])
                optimizer.zero_grad()
                x_0 = combined_groups.to(device)
                features = net_model(x_0)
                loss = isia_loss(features, modelConfig['batch_size'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        torch.save(net_model.state_dict(), os.path.join(
            path, 'ckpt_' + str(e) + "_.pt"))
        loss_list.append(loss.item())
    # paintTrend(loss_list, epochs=modelConfig['epoch'], stride=5)

def select_best(modelConfig: Dict,alpha=0.98,beta=0.98):
    seed_torch(modelConfig['seed'])
    device = torch.device(modelConfig["device"])
    opt_epoch = 0
    max_auc = 0
    path = modelConfig["save_dir"] + '/' + modelConfig['path'] + '/'
    for e in range(modelConfig['epoch']):
        with torch.no_grad():
            if modelConfig["path"] == "Indian_pines_corrected.mat":
                mat = sio.loadmat(modelConfig["path"])
                data = mat['indian_pines_corrected']
                mat1 = sio.loadmat("Indian_pines_gt.mat")
                temp_gt = mat1['indian_pines_gt']
                map = (temp_gt == 16).astype(int)
            else:
                mat = sio.loadmat(modelConfig["path"])
                data = mat['data']
                map = mat['map']


            data = standard(data)
            data = np.float32(data)
            target_spectrum = ts_generation(data, map, 7)
            h, w, c = data.shape
            numpixel = h * w
            data_matrix = np.reshape(data, [-1, c], order='F')
            model = SpectralGroupAttention(band=modelConfig['band'], m=modelConfig['group_length'],
                                           d=modelConfig['channel'], depth=modelConfig['depth'],
                                           heads=modelConfig['heads'],
                                           dim_head=modelConfig['dim_head'], mlp_dim=modelConfig['mlp_dim'],
                                           adjust=modelConfig['adjust']).to(device)
            ckpt = torch.load(os.path.join(
                path, "ckpt_%s_.pt" % e), map_location=device)
            model.load_state_dict(ckpt)
            # print("model load weight done.%s" % e)
            model.eval()

            batch_size = modelConfig['batch_size']
            detection_map = np.zeros([numpixel])
            target_prior = spectral_group(target_spectrum.T, modelConfig['band'], modelConfig['group_length'])
            target_prior = target_prior.to(device)
            target_features = model(target_prior)
            target_features = target_features.cpu().detach().numpy()

            for i in range(0, numpixel - batch_size, batch_size):
                pixels = data_matrix[i:i + batch_size]
                pixels = spectral_group(pixels, modelConfig['band'], modelConfig['group_length'])
                pixels = pixels.to(device)
                features = model(pixels)
                features = features.cpu().detach().numpy()
                detection_map[i:i + batch_size] = cosin_similarity_numpy(features, target_features)

            left_num = numpixel % batch_size
            if left_num != 0:
                pixels = data_matrix[-left_num:]
                pixels = spectral_group(pixels, modelConfig['band'], modelConfig['group_length'])
                pixels = pixels.to(device)
                features = model(pixels)
                features = features.cpu().detach().numpy()
                detection_map[-left_num:] = cosin_similarity_numpy(features, target_features)

            detection_map = np.reshape(detection_map, [h, w], order='F')
            detection_map = standard(detection_map)
            detection_map = np.clip(detection_map, 0, 1)
            y_l = np.reshape(map, [-1, 1], order='F')
            y_p = np.reshape(detection_map, [-1, 1], order='F')

            ## calculate the AUC value
            fpr, tpr, _ = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
            fpr = fpr[1:]
            tpr = tpr[1:]
            auc = round(metrics.auc(fpr, tpr), modelConfig['epision'])
            if auc > max_auc:
                max_auc = auc
                opt_epoch = e
    print("max_accuracy:"+str(max_auc))
    print("epoch:"+str(opt_epoch))
    if opt_epoch is not None:
        best_ckpt_path = os.path.join(path, f"ckpt_{opt_epoch}_.pt")
        new_ckpt_path = os.path.join(path, f"best_model_{alpha:.2f}_{beta:.2f}.pt".replace(".", "_"))
        os.rename(best_ckpt_path, new_ckpt_path)
        print(f"Renamed {best_ckpt_path} to {new_ckpt_path}")
    else:
        print("No optimal checkpoint found.")
    return max_auc
def calculate_flops(model, band, m):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 输入形状 (batch_size, band, m)
    input_tensor = torch.randn(1, band, m).to(device)

    # 使用 ptflops 计算 FLOPs 和参数数量
    macs, params = get_model_complexity_info(model, (band, m), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    print(f'FLOPs: {macs}, Params: {params}')
def eval(modelConfig: Dict):
    seed_torch(modelConfig['seed'])
    device = torch.device(modelConfig["device"])
    path = modelConfig["save_dir"] + '/' + modelConfig['path'] + '/'
    # weight_path = modelConfig["save_dir"]  + modelConfig['weight_path'] + '/'
    with torch.no_grad():
        if modelConfig["path"] == "Indian_pines_corrected.mat":
            mat = sio.loadmat(modelConfig["path"])
            data = mat['indian_pines_corrected']
            mat1 = sio.loadmat("Indian_pines_gt.mat")
            temp_gt = mat1['indian_pines_gt']
            map = (temp_gt == 16).astype(int)
        else:
            mat = sio.loadmat(modelConfig["path"])
            data = mat['data']
            map = mat['map']

        data = standard(data)
        data = np.float32(data)
        target_spectrum = ts_generation(data, map, 7)
        h, w, c = data.shape
        numpixel = h * w
        data_matrix = np.reshape(data, [-1, c], order='F')
        model = SpectralGroupAttention(band=modelConfig['band'], m=modelConfig['group_length'],
                                       d=modelConfig['channel'], depth=modelConfig['depth'],heads=modelConfig['heads'],
                                       dim_head=modelConfig['dim_head'], mlp_dim=modelConfig['mlp_dim'], adjust=modelConfig['adjust']).to(device)
        ckpt = torch.load(os.path.join(
            path, modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        calculate_flops(model, band=modelConfig['band'], m=modelConfig['group_length'])

        start = time.time()

        batch_size = modelConfig['batch_size']
        detection_map = np.zeros([numpixel])
        target_prior = spectral_group(target_spectrum.T, modelConfig['band'], modelConfig['group_length'])
        target_prior = target_prior.to(device)
        target_features = model(target_prior)
        target_features = target_features.cpu().detach().numpy()

        for i in range(0, numpixel - batch_size, batch_size):
            pixels = data_matrix[i:i + batch_size]
            pixels = spectral_group(pixels, modelConfig['band'], modelConfig['group_length'])
            pixels = pixels.to(device)
            features = model(pixels)
            features = features.cpu().detach().numpy()
            detection_map[i:i + batch_size] = cosin_similarity_numpy(features, target_features)

        left_num = numpixel % batch_size
        if left_num != 0:
            pixels = data_matrix[-left_num:]
            pixels = spectral_group(pixels, modelConfig['band'], modelConfig['group_length'])
            pixels = pixels.to(device)
            features = model(pixels)
            features = features.cpu().detach().numpy()
            detection_map[-left_num:] = cosin_similarity_numpy(features, target_features)
        # detection_map = np.exp(-1 * (detection_map - 1) ** 2 / modelConfig['delta'])

        detection_map = np.reshape(detection_map, [h, w], order='F')
        detection_map = standard(detection_map)
        endtime = time.time()
        print("eval time:"+str(endtime-start))
        detection_map = np.clip(detection_map, 0, 1)
        # plt.imshow(detection_map)
        # plt.show()
        y_l = np.reshape(map, [-1, 1], order='F')
        y_p = np.reshape(detection_map, [-1, 1], order='F')

        ## calculate the AUC value
        fpr, tpr, threshold = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
        fpr = fpr[1:]
        tpr = tpr[1:]
        threshold = threshold[1:]
        auc1 = round(metrics.auc(fpr, tpr), modelConfig['epision'])
        auc2 = round(metrics.auc(threshold, fpr), modelConfig['epision'])
        auc3 = round(metrics.auc(threshold, tpr), modelConfig['epision'])
        auc4 = round(auc1 + auc3 - auc2, modelConfig['epision'])
        auc5 = round(auc3 / auc2, modelConfig['epision'])
        print('{:.{precision}f}'.format(auc1, precision=modelConfig['epision']))
        print('{:.{precision}f}'.format(auc2, precision=modelConfig['epision']))
        print('{:.{precision}f}'.format(auc3, precision=modelConfig['epision']))
        print('{:.{precision}f}'.format(auc4, precision=modelConfig['epision']))
        print('{:.{precision}f}'.format(auc5, precision=modelConfig['epision']))

        # 2a. 方法一：用 precision_recall_curve + auc 计算 PR-AUC
        precision, recall, pr_thresholds = metrics.precision_recall_curve(y_l, y_p)
        # precision_recall_curve 会多返回一个触发最低 recall=1.0 的点，通常直接用 recall,precision
        pr_auc1 = round(metrics.auc(recall, precision), modelConfig['epision'])

        # 2b. 方法二：直接用 average_precision_score
        # average_precision_score 本质也是对 PR 曲线积分，通常更推荐
        pr_auc2 = round(metrics.average_precision_score(y_l, y_p), modelConfig['epision'])

        print(f"ROC-AUC = {auc1}")
        print(f"PR-AUC (via auc) = {pr_auc1}")
        print(f"PR-AUC (via average_precision_score) = {pr_auc2}")
        roc_save_dir = 'picture/roc'
        os.makedirs(roc_save_dir, exist_ok=True)
        # 创建1行4列大图
        fig = plt.figure(figsize=(24, 5))

        # 图1: fpr-tpr
        ax1 = fig.add_subplot(1, 4, 1)
        ax1.plot(fpr, tpr, color='blue', lw=2, label=f'AUC1={auc1}')
        ax1.plot([0, 1], [0, 1], linestyle='--', color='grey')
        ax1.set_xlabel('FPR')
        ax1.set_ylabel('TPR')
        ax1.set_title('ROC Curve 1')
        ax1.legend(loc='lower right')
        ax1.grid(True)

        # 图2: fpr-threshold
        ax2 = fig.add_subplot(1, 4, 2)
        ax2.plot(threshold, fpr, color='green', lw=2, label=f'AUC2={auc2}')
        ax2.set_xlabel('FPR')
        ax2.set_ylabel('Threshold')
        ax2.set_title('ROC Curve 2')
        ax2.legend(loc='best')
        ax2.grid(True)

        # 图3: tpr-threshold
        ax3 = fig.add_subplot(1, 4, 3)
        ax3.plot(threshold, tpr, color='red', lw=2, label=f'AUC3={auc3}')
        ax3.set_xlabel('TPR')
        ax3.set_ylabel('Threshold')
        ax3.set_title('ROC Curve 3')
        ax3.legend(loc='best')
        ax3.grid(True)

        # 图4: 3D ROC曲面
        ax4 = fig.add_subplot(1, 4, 4, projection='3d')
        ax4.plot(fpr, threshold, tpr, color='purple', lw=2)
        ax4.set_xlabel('FPR')
        ax4.set_zlabel('TPR')
        ax4.set_ylabel('Threshold')
        ax4.set_title('3D ROC Curve')
        ax4.view_init(elev=20, azim=135)


        plt.tight_layout()
        plt.savefig(f'{roc_save_dir}/{modelConfig["name"]}-roc-all.png')
        plt.close()
        print("ROC Curves saved to '{}'".format(roc_save_dir))

        # 定义保存路径
        savepath = f"picture/result/tsttd/{modelConfig['name']}.png"

        # 保存图像
        save_image(detection_map, savepath)

        print(f"Saved detection image to {savepath}")
        # 指定保存路径
        save_path = f"csv/{modelConfig['path']}/Proposed.npz"
        print(len(fpr), len(tpr), len(threshold))
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path,
                 thresholds=threshold,
                 pd=tpr,
                 far=fpr)
        # 保存 npz 文件
        print(f"Saved npz file to {save_path}")

def select_params(modelConfig: Dict):
    Z = np.zeros((11, 11))
    # seed_torch(modelConfig['seed'])
    # device = torch.device(modelConfig["device"])
    alpha_list = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55]
    beta_list = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55]
    for i, alpha in enumerate(alpha_list):
        for j, beta in enumerate(beta_list):
            seed_torch(modelConfig['seed'])
            device = torch.device(modelConfig["device"])
            dataset = Data(modelConfig["path"],alpha,beta)
            # print(len(dataset))
            dataloader = DataLoader(dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=24, drop_last=True,
                                       pin_memory=True)
            # print(len(dataloader))
            # model setup
            net_model = SpectralGroupAttention(band=modelConfig['band'], m=modelConfig['group_length'],
                                               d=modelConfig['channel'], depth=modelConfig['depth'], heads=modelConfig['heads'],
                                               dim_head=modelConfig['dim_head'], mlp_dim=modelConfig['mlp_dim'],
                                               adjust=modelConfig['adjust']).to(device)
            if modelConfig["training_load_weight"] is not None:
                net_model.load_state_dict(torch.load(os.path.join(
                    modelConfig["save_dir"], modelConfig["training_load_weight"]), map_location=device), strict=False)
                print("Model weight load down.")
            optimizer = torch.optim.AdamW(
                    net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
            cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
            warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                                         warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
            path = modelConfig["save_dir"] + '/' + modelConfig['path'] + '/'
            checkFile(path)

            # start training
            net_model.train()
            loss_list = []
            for e in range(modelConfig["epoch"]):
                    with tqdm(dataloader, dynamic_ncols=True,disable=True) as tqdmDataLoader:
                        for positive, negative in tqdmDataLoader:
                            # train
                            combined_vectors = np.concatenate([positive, negative, dataset.target_spectrum], axis=0)
                            combined_groups = spectral_group(combined_vectors, modelConfig['band'], modelConfig['group_length'])
                            optimizer.zero_grad()
                            x_0 = combined_groups.to(device)
                            features = net_model(x_0)
                            loss = isia_loss(features, modelConfig['batch_size'])
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                net_model.parameters(), modelConfig["grad_clip"])
                            optimizer.step()
                            tqdmDataLoader.set_postfix(ordered_dict={
                                "epoch": e,
                                "loss: ": loss.item(),
                                "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                            })
                    warmUpScheduler.step()
                    torch.save(net_model.state_dict(), os.path.join(
                        path, 'ckpt_' + str(e) + "_.pt"))
                    loss_list.append(loss.item())
            Z[i, j] = select_best(modelConfig, alpha, beta)  # 赋值给矩阵对应位置
    draw_3d(Z)