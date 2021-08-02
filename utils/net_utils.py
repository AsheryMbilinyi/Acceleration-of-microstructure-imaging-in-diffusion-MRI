"""
Definitions of loss
"""

import numpy as np

#from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.keras import backend as K


def loss_func(y_true, y_pred):
    """
    Use RMSE loss.
    """
    return loss_rmse(y_true, y_pred) 

def loss_func_1(y_true, y_pred):

    #return loss_rmse(y_true, y_pred)
    return 0.9 * loss_rmse(y_true, y_pred) + 0.1 * loss_ssim(y_true, y_pred)

def loss_func_2(y_true, y_pred):

    #return loss_rmse(y_true, y_pred)
    return 0.8 * loss_rmse(y_true, y_pred) + 0.2 * loss_ssim(y_true, y_pred)

def loss_func_3(y_true, y_pred):

    #return loss_rmse(y_true, y_pred)
    return 0.7 * loss_rmse(y_true, y_pred) + 0.3 * loss_ssim(y_true, y_pred)

def loss_func_4(y_true, y_pred):

    #return loss_rmse(y_true, y_pred)
    return 0.6 * loss_rmse(y_true, y_pred) + 0.4 * loss_ssim(y_true, y_pred)

def loss_func_5(y_true, y_pred):

    #return loss_rmse(y_true, y_pred)
    return 0.5 * loss_rmse(y_true, y_pred) + 0.5 * loss_ssim(y_true, y_pred)

def loss_func_6(y_true, y_pred):

    #return loss_rmse(y_true, y_pred)
    return 0.4 * loss_rmse(y_true, y_pred) + 0.6 * loss_ssim(y_true, y_pred)

def loss_func_7(y_true, y_pred):

    #return loss_rmse(y_true, y_pred)
    return 0.3 * loss_rmse(y_true, y_pred) + 0.7 * loss_ssim(y_true, y_pred)

def loss_func_8(y_true, y_pred):

    #return loss_rmse(y_true, y_pred)
    return 0.2 * loss_rmse(y_true, y_pred) + 0.8 * loss_ssim(y_true, y_pred)

def loss_func_9(y_true, y_pred):

    #return loss_rmse(y_true, y_pred)
    return 0.1 * loss_rmse(y_true, y_pred) + 0.9 * loss_ssim(y_true, y_pred)

def loss_func_10(y_true, y_pred):

    #return loss_rmse(y_true, y_pred)
    return loss_ssim(y_true, y_pred)

def loss_rmse(y_true, y_pred):
    """
    RMSE loss.
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), keepdims=True))

loss_funcs = [loss_func, loss_func_1, loss_func_2, loss_func_3, loss_func_4, loss_func_5, loss_func_6, loss_func_7, loss_func_8, loss_func_9, loss_func_10]


def loss_psnr(y_true, y_pred):
    """
    Use PSNR loss
    """
    return   -20 * K.log(K.max(y_true) / (K.sqrt(K.mean(K.square(y_pred - y_true)))) + K.epsilon())

def loss_ssim(y_true, y_pred):
    """
    SSIM
    """

    #return - K.mean(((2 * K.mean(y_true, keepdims=True) * K.mean(y_pred, keepdims=True)) * \
    #                   (2 * K.std(y_true, keepdims=True) * K.std(y_pred, keepdims=True)) \
    #                  ) / \
    #                  ((K.square(K.mean(y_true, keepdims=True)) + K.square(K.mean(y_pred, keepdims=True))) * \
    #                   (K.square(K.std(y_true, keepdims=True)) + K.square(K.std(y_pred, keepdims=True))) \
    #                  )
    #                 )

    return 1 - ((2 * K.mean(y_true) * K.mean(y_pred)) * \
                (2 * (K.mean(y_true * y_pred) - K.mean(y_pred) * K.mean(y_true))) \
               ) / \
               ((K.square(K.mean(y_true)) + K.square(K.mean(y_pred))) * \
                (K.square(K.std(y_true)) + K.square(K.std(y_pred))) \
               )

def calc_RMSE(pred, gt, mask, percentage=False, model='', index=None):
    """
    Calculate RMSE of a & b
    """
    if model[:6] == 'conv3d':
        #true_mask[:, :, 1:-1] = mask[:, :, 1:-1]
        mask[:, :, 0] = 0
        mask[:, :, -1] = 0
        if index is not None:
            index = index - 1

    if index is not None:
        pred = pred[:, :, index:index+1]
        gt = gt[:, :, index:index+1]
        mask = mask[:, :, index:index+1]

    pred = pred[mask > 0]
    gt = gt[mask > 0]

    if percentage:
        return np.sqrt((((pred - gt) / (gt + 1e-7)) ** 2).mean())
    else:
        return np.sqrt(((pred - gt) ** 2).mean())

def calc_psnr(a, b):
    """
    """
    return -20 * np.log(np.max(b) / (np.sqrt(np.mean(np.square(a - b)))) + 1e-8)

def calc_ssim(a, b):
    """
    A
    """
    lxy = (2 * a.mean() * b.mean()) / (a.mean() ** 2 + b.mean() ** 2)
    cxy = (2 * np.std(a) * np.std(b)) / (np.std(a) ** 2 + np.std(b) ** 2)
    sxy = (np.mean(a * b) - np.mean(a) * np.mean(b)) / (np.std(a) * np.std(b))

    #AA = (4 * a.mean() * b.mean() * np.std(a) * np.std(b))
    #BB = ((a.mean() ** 2) + (b.mean()) ** 2) * ((np.std(a) ** 2) + (np.std(b) ** 2))

    return lxy * cxy * sxy