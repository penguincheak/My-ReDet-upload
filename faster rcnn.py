import torch
import torch.nn.functional as F
import numpy as np

pred = torch.tensor([[-1.0687e-02,  2.8639e-02,  1.7221e-02, -3.8597e-02,  1.7231e-02],
        [-1.2305e-02, -2.2052e-03,  7.8945e-03, -2.7401e-02,  2.4540e-02],
        [-2.0463e-02, -3.2636e-03,  1.2343e-02, -2.0135e-02,  2.9087e-02],
        [-2.5700e-02,  1.8988e-02,  1.1242e-02, -1.7086e-02,  1.2345e-02],
        [-2.1388e-02,  1.5671e-03,  1.5423e-02, -3.0389e-02,  2.6674e-02],
        [-4.3696e-02, -5.4154e-03,  3.1037e-02, -5.3827e-02, -1.9703e-02],
        [ 8.3128e-03, -4.5085e-04,  2.8037e-02, -2.2486e-02, -8.9536e-04],
        [-3.4912e-03,  3.1667e-03, -2.9347e-02, -1.7227e-02,  2.8174e-03],
        [-1.3461e-02,  3.6302e-02,  1.9997e-02, -3.3305e-02,  4.3863e-05],
        [ 2.0080e-02,  1.1253e-02,  6.1648e-02, -1.8661e-02,  4.5245e-04],
        [ 2.2702e-03,  2.8819e-03,  6.1002e-02, -2.4647e-02,  2.3481e-02],
        [ 1.1374e-02, -3.1773e-03, -2.3455e-02, -3.0648e-02, -2.5453e-02],
        [-1.1751e-02,  7.5973e-03,  5.6825e-02, -2.5304e-02,  4.1839e-03],
        [ 1.2333e-02, -1.7586e-04, -2.4054e-02, -9.8938e-03, -3.4205e-03],
        [-1.5145e-02,  1.2196e-02, -5.0858e-03, -1.0961e-02, -2.5393e-02],
        [ 1.3272e-02,  2.0108e-02, -6.2379e-03, -3.3396e-02,  3.7493e-03]])

target = torch.tensor([[ 1.4033e-02,  4.9579e-02, -2.2081e+00, -1.0878e+00,  7.7305e+00],
        [ 1.1424e-02, -1.4115e-01, -1.3333e+00, -1.9160e+00, -7.5874e+00],
        [ 7.8832e-03, -1.6516e-01, -1.2702e+00, -1.8439e+00, -7.1091e+00],
        [-1.0462e+00,  8.0313e-01, -4.1576e-02, -3.3299e-01, -7.1091e+00],
        [ 6.0598e-01, -1.8641e-02, -3.5793e-01, -6.3979e-01, -7.5874e+00],
        [-1.0936e-01,  2.4473e-01,  3.3467e-02, -7.5385e-01,  8.3137e-01],
        [ 3.9005e-02,  9.4831e-01, -2.7604e-01,  4.3918e-02,  1.2062e+00],
        [-5.4772e-02,  6.4632e-02, -4.8319e-01, -3.5642e-01,  1.1066e+00],
        [-1.1081e-03, -1.3596e+00, -1.6822e-01,  6.1639e-02,  1.0979e+00],
        [-3.5908e+00,  4.0232e-01,  2.5192e+00,  6.7600e-01,  1.2062e+00],
        [-2.1017e+00,  5.0694e-01, -4.3189e-01,  5.6091e-01,  1.2062e+00],
        [-2.0185e-01, -1.2323e+00, -2.9445e-01,  1.2885e+00,  1.1066e+00],
        [-1.4224e+00,  1.9369e+00,  1.8826e+00, -1.3545e+00,  1.2062e+00],
        [ 1.7204e+00,  7.4947e-02, -4.3482e-01,  1.5285e+00,  1.1066e+00],
        [-1.7372e+00,  5.7187e-01, -2.9888e-01,  1.2869e+00,  1.1066e+00],
        [-1.3886e+00,  2.2932e+00, -7.3855e-01, -1.9729e+00,  1.2062e+00]])

def smooth_l1_loss(pred, target, beta=1.0, reduction='mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0

    index = torch.tensor([[1, 0, -1, -2], [-1, -2, -3, 0],
                          [1, 2, 3, 0], [2, 1, 0, -1]])
    target_angle = target[:, 4] / 10 - (np.pi / 2)

    zeros = torch.zeros(target.size(0))
    ones = torch.ones(target.size(0))
    twos = torch.ones(target.size(0)) * 2
    threes = torch.ones(target.size(0)) * 3

    angle_field = torch.where(target_angle > 0, ones, zeros)
    angle_field = torch.where((angle_field == 1) & (target_angle > np.pi/2), ones, zeros)
    angle_field = torch.where((angle_field == 0) & (target_angle < (-np.pi/2)), twos, threes)

    angle_index = index[angle_field.long()]
    angle_aug = angle_index.float() * np.pi / 2
    target_angle = (target_angle.view(target_angle.size(0), 1).expand_as(angle_aug)\
                   + angle_aug + (np.pi / 2)) * 10
    angle_diff = torch.abs(pred[:, 4].view(target_angle.size(0), 1).expand_as(target_angle)
                           - target_angle)
    angle_diff = torch.min(angle_diff, dim=1).values

    diff = torch.abs(pred - target)
    diff[:, 4] = angle_diff
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / pred.numel()
    elif reduction_enum == 2:
        return loss.sum()


if __name__ == '__main__':
    loss = smooth_l1_loss(pred, target)
    print(loss)
