import torch
import torch.nn as nn
import numpy as np


class WaveletNet(nn.Module):
    def __init__(self):
        super(WaveletNet, self).__init__()
    # batch operation usng tensor slice
    def WaveletTransformAxisY(self, batch_img):
        odd_img = batch_img[:, 0::2]
        even_img = batch_img[:, 1::2]
        L = (odd_img + even_img) / 2.0
        H = np.abs(odd_img - even_img)
        return L, H

    def WaveletTransformAxisX(self, batch_img):
        # transpose + fliplr
        # tmp_batch = batch_img.permute(0, 2, 1)
        tmp_batch = np.transpose(batch_img, (0, 2, 1))[:, :, ::-1]
        _dst_L, _dst_H = self.WaveletTransformAxisY(tmp_batch)
        # transpose + flipud
        dst_L = np.transpose(_dst_L, (0, 2, 1))[:, ::-1, ...]
        dst_H = np.transpose(_dst_H, (0, 2, 1))[:, ::-1, ...]
        return dst_L, dst_H

    def forward(self, batch_image):
        device = batch_image.device
        batch_image = batch_image.cpu().numpy()
        # make channel first image
        # batch_image = batch_image.permute(0, 3, 1, 2)
        # batch_image = np.transpose(batch_image, (0, 3, 1, 2))
        r = batch_image[:, 0]
        g = batch_image[:, 1]
        b = batch_image[:, 2]

        # level 1 decomposition
        wavelet_L, wavelet_H = self.WaveletTransformAxisY(r)
        r_wavelet_LL, r_wavelet_LH = self.WaveletTransformAxisX(wavelet_L)
        r_wavelet_HL, r_wavelet_HH = self.WaveletTransformAxisX(wavelet_H)

        wavelet_L, wavelet_H = self.WaveletTransformAxisY(g)
        g_wavelet_LL, g_wavelet_LH = self.WaveletTransformAxisX(wavelet_L)
        g_wavelet_HL, g_wavelet_HH = self.WaveletTransformAxisX(wavelet_H)

        wavelet_L, wavelet_H = self.WaveletTransformAxisY(b)
        b_wavelet_LL, b_wavelet_LH = self.WaveletTransformAxisX(wavelet_L)
        b_wavelet_HL, b_wavelet_HH = self.WaveletTransformAxisX(wavelet_H)

        wavelet_data = [r_wavelet_LL, r_wavelet_LH, r_wavelet_HL, r_wavelet_HH,
                        g_wavelet_LL, g_wavelet_LH, g_wavelet_HL, g_wavelet_HH,
                        b_wavelet_LL, b_wavelet_LH, b_wavelet_HL, b_wavelet_HH]
        transform_batch = np.stack(wavelet_data, axis=1)

        # level 2 decomposition
        wavelet_L2, wavelet_H2 = self.WaveletTransformAxisY(r_wavelet_LL)
        r_wavelet_LL2, r_wavelet_LH2 = self.WaveletTransformAxisX(wavelet_L2)
        r_wavelet_HL2, r_wavelet_HH2 = self.WaveletTransformAxisX(wavelet_H2)

        wavelet_L2, wavelet_H2 = self.WaveletTransformAxisY(g_wavelet_LL)
        g_wavelet_LL2, g_wavelet_LH2 = self.WaveletTransformAxisX(wavelet_L2)
        g_wavelet_HL2, g_wavelet_HH2 = self.WaveletTransformAxisX(wavelet_H2)

        wavelet_L2, wavelet_H2 = self.WaveletTransformAxisY(b_wavelet_LL)
        b_wavelet_LL2, b_wavelet_LH2 = self.WaveletTransformAxisX(wavelet_L2)
        b_wavelet_HL2, b_wavelet_HH2 = self.WaveletTransformAxisX(wavelet_H2)

        wavelet_data_l2 = [r_wavelet_LL2, r_wavelet_LH2, r_wavelet_HL2, r_wavelet_HH2,
                           g_wavelet_LL2, g_wavelet_LH2, g_wavelet_HL2, g_wavelet_HH2,
                           b_wavelet_LL2, b_wavelet_LH2, b_wavelet_HL2, b_wavelet_HH2]
        transform_batch_l2 = np.stack(wavelet_data_l2, axis=1)

        # level 3 decomposition
        wavelet_L3, wavelet_H3 = self.WaveletTransformAxisY(r_wavelet_LL2)
        r_wavelet_LL3, r_wavelet_LH3 = self.WaveletTransformAxisX(wavelet_L3)
        r_wavelet_HL3, r_wavelet_HH3 = self.WaveletTransformAxisX(wavelet_H3)

        wavelet_L3, wavelet_H3 = self.WaveletTransformAxisY(g_wavelet_LL2)
        g_wavelet_LL3, g_wavelet_LH3 = self.WaveletTransformAxisX(wavelet_L3)
        g_wavelet_HL3, g_wavelet_HH3 = self.WaveletTransformAxisX(wavelet_H3)

        wavelet_L3, wavelet_H3 = self.WaveletTransformAxisY(b_wavelet_LL2)
        b_wavelet_LL3, b_wavelet_LH3 = self.WaveletTransformAxisX(wavelet_L3)
        b_wavelet_HL3, b_wavelet_HH3 = self.WaveletTransformAxisX(wavelet_H3)

        wavelet_data_l3 = [r_wavelet_LL3, r_wavelet_LH3, r_wavelet_HL3, r_wavelet_HH3,
                           g_wavelet_LL3, g_wavelet_LH3, g_wavelet_HL3, g_wavelet_HH3,
                           b_wavelet_LL3, b_wavelet_LH3, b_wavelet_HL3, b_wavelet_HH3]
        transform_batch_l3 = np.stack(wavelet_data_l3, axis=1)

        # level 4 decomposition
        wavelet_L4, wavelet_H4 = self.WaveletTransformAxisY(r_wavelet_LL3)
        r_wavelet_LL4, r_wavelet_LH4 = self.WaveletTransformAxisX(wavelet_L4)
        r_wavelet_HL4, r_wavelet_HH4 = self.WaveletTransformAxisX(wavelet_H4)

        wavelet_L4, wavelet_H4 = self.WaveletTransformAxisY(g_wavelet_LL3)
        g_wavelet_LL4, g_wavelet_LH4 = self.WaveletTransformAxisX(wavelet_L4)
        g_wavelet_HL4, g_wavelet_HH4 = self.WaveletTransformAxisX(wavelet_H4)

        wavelet_L4, wavelet_H4 = self.WaveletTransformAxisY(b_wavelet_LL3)
        b_wavelet_LL4, b_wavelet_LH4 = self.WaveletTransformAxisX(wavelet_L4)
        b_wavelet_HL4, b_wavelet_HH4 = self.WaveletTransformAxisX(wavelet_H4)

        wavelet_data_l4 = [r_wavelet_LL4, r_wavelet_LH4, r_wavelet_HL4, r_wavelet_HH4,
                           g_wavelet_LL4, g_wavelet_LH4, g_wavelet_HL4, g_wavelet_HH4,
                           b_wavelet_LL4, b_wavelet_LH4, b_wavelet_HL4, b_wavelet_HH4]
        transform_batch_l4 = np.stack(wavelet_data_l4, axis=1)

        # 新增
        # level 5 decomposition
        wavelet_L5, wavelet_H5 = self.WaveletTransformAxisY(r_wavelet_LL4)
        r_wavelet_LL5, r_wavelet_LH5 = self.WaveletTransformAxisX(wavelet_L5)
        r_wavelet_HL5, r_wavelet_HH5 = self.WaveletTransformAxisX(wavelet_H5)

        wavelet_L5, wavelet_H5 = self.WaveletTransformAxisY(g_wavelet_LL4)
        g_wavelet_LL5, g_wavelet_LH5 = self.WaveletTransformAxisX(wavelet_L5)
        g_wavelet_HL5, g_wavelet_HH5 = self.WaveletTransformAxisX(wavelet_H5)

        wavelet_L5, wavelet_H5 = self.WaveletTransformAxisY(b_wavelet_LL4)
        b_wavelet_LL5, b_wavelet_LH5 = self.WaveletTransformAxisX(wavelet_L5)
        b_wavelet_HL5, b_wavelet_HH5 = self.WaveletTransformAxisX(wavelet_H5)

        wavelet_data_l5 = [r_wavelet_LL5, r_wavelet_LH5, r_wavelet_HL5, r_wavelet_HH5,
                           g_wavelet_LL5, g_wavelet_LH5, g_wavelet_HL5, g_wavelet_HH5,
                           b_wavelet_LL5, b_wavelet_LH5, b_wavelet_HL5, b_wavelet_HH5]
        transform_batch_l5 = np.stack(wavelet_data_l5, axis=1)





        # print('shape before')
        # print(transform_batch.shape)
        # print(transform_batch_l2.shape)
        # print(transform_batch_l3.shape)
        # print(transform_batch_l4.shape)

        # 原始
        # decom_level_1 = np.transpose(transform_batch, (0, 2, 3, 1))
        # decom_level_2 = np.transpose(transform_batch_l2, (0, 2, 3, 1))
        # decom_level_3 = np.transpose(transform_batch_l3, (0, 2, 3, 1))
        # decom_level_4 = np.transpose(transform_batch_l4, (0, 2, 3, 1))

        decom_level_1 = torch.from_numpy(transform_batch).float().cuda(device)  # /2
        decom_level_2 = torch.from_numpy(transform_batch_l2).float().cuda(device)  # /4
        decom_level_3 = torch.from_numpy(transform_batch_l3).float().cuda(device)  # /8
        decom_level_4 = torch.from_numpy(transform_batch_l4).float().cuda(device)  # /16
        decom_level_5 = torch.from_numpy(transform_batch_l5).float().cuda(device)  # /32
        decom_level_1.requires_grad = True
        decom_level_2.requires_grad = True
        decom_level_3.requires_grad = True
        decom_level_4.requires_grad = True
        decom_level_5.requires_grad = True
        # print('shape after')
        # print(decom_level_1.shape)
        # print(decom_level_2.shape)
        # print(decom_level_3.shape)
        # print(decom_level_4.shape)
        return [decom_level_1,
                decom_level_2,
                decom_level_3,
                decom_level_4,
                decom_level_5]
