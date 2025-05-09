import matplotlib.pyplot as plt
import torch.utils.data as data
import scipy.io as sio
import numpy as np
import torch
from ts_generation import ts_generation
from Tools import standard
from my_cem import cem
import os
import random


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


def delete_target(result, data):
    # 执行上述代码
    mask = result < 0.2
    selected_vectors = data[mask]
    result_array = selected_vectors.reshape(-1, selected_vectors.shape[-1])
    return result_array

def delete_background(result, data):
    # 执行上述代码
    mask = result >0.5 #sandiego2:0.5848  #beach 0.43  #beach 2 ace 0.995 normal 0.5
    selected_vectors = data[mask]
    result_array = selected_vectors.reshape(-1, selected_vectors.shape[-1])
    return result_array
#
# def delete_target(result, data, percentage=80):
#     """
#     删除result最小的百分之 percentage 的对应数据。
#     :param result: 预测概率数组，shape 与 data 相同，通常为二维。
#     :param data: 数据数组，shape 与 result 相同。
#     :param percentage: 百分比，表示剔除前百分之几的小值。
#     :return: 删除后的一维向量集合。
#     """
#     threshold_index = int(len(result.flatten()) * (1 - percentage / 100))
#     sorted_result = np.sort(result.flatten())
#     threshold = sorted_result[threshold_index]
#
#     mask = result <= threshold
#     selected_vectors = data[mask]
#     result_array = selected_vectors.reshape(-1, selected_vectors.shape[-1])
#     return result_array
#
# def delete_background(result, data, percentage=80):
#     """
#     删除result最大的百分之 percentage 的对应数据。
#     :param result: 预测概率数组，shape 与 data 相同，通常为二维。
#     :param data: 数据数组，shape 与 result 相同。
#     :param percentage: 百分比，表示剔除后百分之几的大值。
#     :return: 删除后的一维向量集合。
#     """
#     threshold_index = int(len(result.flatten()) * (percentage / 100))
#     sorted_result = np.sort(result.flatten())
#     threshold = sorted_result[threshold_index]
#
#     mask = result >= threshold
#     selected_vectors = data[mask]
#     result_array = selected_vectors.reshape(-1, selected_vectors.shape[-1])
#     return result_array
nums = [0,1,2,2,3,0,4,2]
val = 2


class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        for i,num in enumerate(nums):
            if num == val:
                for j in nums[i+1:]:
                    nums[j] = num[j+1]

#
# class Data(data.Dataset):#htdformer
#     def __init__(self, path,alpha_max, beta_max):
#         seed_torch(1)
#         if path == "Indian_pines_corrected.mat":
#             mat = sio.loadmat(path)
#             data = mat['indian_pines_corrected']
#             mat1 = sio.loadmat("Indian_pines_gt.mat")
#             temp_gt = mat1['indian_pines_gt']
#             gt = (temp_gt == 16).astype(int)
#         else:
#             mat = sio.loadmat(path)
#             data = mat['data']
#             gt = mat['map']
#         print("GT中值为1的像素个数：", np.sum(gt == 1))
#         print("GT中值为0的像素个数：", np.sum(gt == 0))
#
#
#         data = standard(data)
#
#         h, w, b = data.shape
#
#         ## get the target spectrum
#         target_spectrum = ts_generation(data, gt, 7)
#
#         result = cem(data, target_spectrum)
#         pre_background_samples = delete_target(result, data)
#         bg_pixel_nums=len(pre_background_samples)
#
#         pre_target_samples = delete_background(result, data)
#         target_pixel_nums=len(pre_target_samples)
#
#
#         new_gen_target_matrix = np.vstack(pre_target_samples)  # 形状为 (4*9900, 189)
#         new_gen_bg_matrix = np.vstack(pre_background_samples)  # 形状为 (4*9900, 189)
#
#
#         self.target_samples = new_gen_target_matrix
#         self.background_samples = new_gen_bg_matrix
#         self.target_spectrum = target_spectrum.T
#         self.nums = min(bg_pixel_nums, target_pixel_nums)
#         print(self.background_samples.shape)
#         print(self.target_samples.shape)
#
#     def __getitem__(self, index):
#         positive_samples = self.target_samples[index]
#         negative_samples = self.background_samples[index]
#         return positive_samples, negative_samples
#
#
#     def __len__(self):
#         return self.nums
def remove_field(text, field):
    return text.replace(field, '')

class Data(data.Dataset):#我的方法
    def __init__(self, path,alpha_max, beta_max):
        seed_torch(1)
        if path == "Indian_pines_corrected.mat":
            mat = sio.loadmat(path)
            data = mat['indian_pines_corrected']
            mat1 = sio.loadmat("Indian_pines_gt.mat")
            temp_gt = mat1['indian_pines_gt']
            gt = (temp_gt == 16).astype(int)
        else:
            mat = sio.loadmat(path)
            data = mat['data']
            gt = mat['map']
        print("GT中值为1的像素个数：", np.sum(gt == 1))
        print("GT中值为0的像素个数：", np.sum(gt == 0))
        path02 = remove_field(path, ".mat")

        data = standard(data)

        h, w, b = data.shape
        save_dir = "picture/targetsamples"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir,  "Sandiego2-0421.png")
        data_reshaped = data.reshape(-1, b)
        gt_reshaped = gt.flatten()

        plt.figure(figsize=(9, 6))
        ax = plt.gca()
        for i in range(len(gt_reshaped)):
            if gt_reshaped[i] == 1:
                plt.plot(data_reshaped[i], alpha=0.7,linewidth = 1.5)  # 透明度0.3，避免太密
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体支持中文
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        # plt.title('GT=1的光谱曲线',fontsize=24)
        # plt.xlabel('波段',fontsize=36)
        # plt.ylabel('反射率',fontsize=36)
        plt.grid(True)
        plt.tick_params(axis='both', which='major', labelsize=36)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(2.5)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()


        print(f"图片已保存到 {save_path}")
        ## get the target spectrum
        target_spectrum = ts_generation(data, gt, 7)

        result = cem(data, target_spectrum)
        pre_background_samples = delete_target(result, data)
        bg_pixel_nums=len(pre_background_samples)

        pre_target_samples = delete_background(result, data)
        target_pixel_nums=len(pre_target_samples)

        new_gen_target_list = []
        new_gen_bg_list = []

        for pre_target_sample in pre_target_samples:  # 遍历每个目标光谱
            alphas = np.random.uniform(0, alpha_max, bg_pixel_nums)
            alphas = alphas[:, None]
            betas = np.random.uniform(0, beta_max, bg_pixel_nums)
            betas = betas[:, None]
            pre_target_sample = pre_target_sample[:, None]
            new_gen_tar = alphas * pre_background_samples + (1 - alphas) * pre_target_sample.T  # 逐像素混合
            new_gen_bg = (1 - betas) * pre_background_samples + betas * pre_target_sample.T
            new_gen_target_list.append(new_gen_tar)
            new_gen_bg_list.append(new_gen_bg)
        new_gen_target_matrix = np.vstack(new_gen_target_list)  # 形状为 (4*9900, 189)
        new_gen_bg_matrix = np.vstack(new_gen_bg_list)  # 形状为 (4*9900, 189)


        self.target_samples = new_gen_target_matrix
        self.background_samples = new_gen_bg_matrix
        self.target_spectrum = target_spectrum.T
        self.nums = bg_pixel_nums*target_pixel_nums
        print(self.background_samples.shape)
        print(self.target_samples.shape)

    def __getitem__(self, index):
        positive_samples = self.target_samples[index]
        negative_samples = self.background_samples[index]
        return positive_samples, negative_samples

    def __len__(self):
        return self.nums
# class Data(data.Dataset):
#     def __init__(self, path,alpha_max, beta_max):
#
#         if path == "Indian_pines_corrected.mat":
#             mat = sio.loadmat(path)
#             data = mat['indian_pines_corrected']
#             mat1 = sio.loadmat("Indian_pines_gt.mat")
#             temp_gt = mat1['indian_pines_gt']
#             gt = (temp_gt == 16).astype(int)
#         else:
#             mat = sio.loadmat(path)
#             data = mat['data']
#             gt = mat['map']
#         data = standard(data)
#
#         h, w, b = data.shape
#         pixel_nums = h * w
#
#         ## get the target spectrum
#         target_spectrum = ts_generation(data, gt, 7)
#         '''
#         CUT
#         '''
#         result = cem(data, target_spectrum)
#         background_samples = delete_target(result, data)
#         pixel_nums=len(background_samples)
#         '''
#         CUT
#         '''
#
#         ## regard all the pixels as background pixels
#         # background_samples1 = np.reshape(data, [-1, b], order='F')
#
#         ## randomly generate target samples by linear representation
#         alphas = np.random.uniform(0, 0.35, pixel_nums)
#         alphas = alphas[:, None]
#         target_samples = alphas * background_samples + (1 - alphas) * target_spectrum.T
#         #T means transposed
#         self.target_samples = target_samples
#         self.background_samples = background_samples
#         self.target_spectrum = target_spectrum.T
#         self.nums = pixel_nums
#         print(self.background_samples.shape)
#         print(self.target_samples.shape)
#
#     def __getitem__(self, index):
#         positive_samples = self.target_samples[index]
#         negative_samples = self.background_samples[index]
#         return positive_samples, negative_samples
#
#     def __len__(self):
#         return self.nums

# class Data(data.Dataset):#最原始版本
#     def __init__(self, path,alpha_max, beta_max):
#         if path == "Indian_pines_corrected.mat":
#             mat = sio.loadmat(path)
#             data = mat['indian_pines_corrected']
#             mat1 = sio.loadmat("Indian_pines_gt.mat")
#             temp_gt = mat1['indian_pines_gt']
#             gt = (temp_gt == 16).astype(int)
#         else:
#             mat = sio.loadmat(path)
#             data = mat['data']
#             gt = mat['map']
#         print("GT中值为1的像素个数：", np.sum(gt == 1))
#         print("GT中值为0的像素个数：", np.sum(gt == 0))
#
#         data = standard(data)
#
#         h, w, b = data.shape
#         pixel_nums = h * w
#
#         ## get the target spectrum
#         target_spectrum = ts_generation(data, gt, 7)
#
#         ## regard all the pixels as background pixels
#         background_samples = np.reshape(data, [-1, b], order='F')
#
#         ## randomly generate target samples by linear representation
#         alphas = np.random.uniform(0, 0.1, pixel_nums)
#         alphas = alphas[:, None]
#         target_samples = alphas * background_samples + (1 - alphas) * target_spectrum.T
#
#         self.target_samples = target_samples
#         self.background_samples = background_samples
#         self.target_spectrum = target_spectrum.T
#         self.nums = pixel_nums
#
#     def __getitem__(self, index):
#         positive_samples = self.target_samples[index]
#         negative_samples = self.background_samples[index]
#         return positive_samples, negative_samples
#
#     def __len__(self):
#         return self.nums



if __name__ == '__main__':
    data = Data('Sandiego2.mat',0.35,0.1)
    target_samples = data.target_spectrum
    print(target_samples.shape)
    plt.plot(target_samples.T)
    # plt.show()
    # center, coded_vector = data.__getitem__(128)
    # plt.plot(center.T)
    # plt.plot(coded_vector.T)
    # plt.show()
