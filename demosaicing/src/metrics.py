import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


def y(rgb):
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def mse(i, j, img_real, img_pred):
    return (y(img_real[i, j]) - y(img_pred[i, j])) ** 2


def all_mse(img_real, img_pred):
    h, w, _ = img_real.shape
    results = Parallel(n_jobs=4)(delayed(mse)(i, j, img_real, img_pred) for i in tqdm(range(h), desc='PSNR progress bar:') for j in range(w))
    return sum(results) / (w * h)


def psnr(img_real, img_pred):
    mse_ = all_mse(img_real, img_pred)
    return 10 * np.log10(255 ** 2 / mse_)
