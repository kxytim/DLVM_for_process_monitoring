"""
Source code of the paper "Deep Learning of Latent Variable Models for Industrial Process Monitoring"
"DOI: 10.1109/TII.2021.3093386"
"https://ieeexplore.ieee.org/document/9468377"
If you use this code in your work, please cite the above paper. Thank you!

@author: Xiangyin Kong
@github account: kxytim, https://github.com/kxytim
"""

import numpy as np
from sklearn.decomposition import FastICA
from sklearn import preprocessing
from scipy import linalg
from scipy import stats
import statsmodels.nonparametric.api as smnp


def load_data(filename):
    data_set = []
    with open(filename) as f:
        for line in f.readlines():
            temp = []
            curline = line.strip().split('  ')
            for i in curline:
                temp.append(float(i))
            data_set.append(temp)
    return np.array(data_set)


def neg(x):  # compute the neg-entropy
    n = x.shape[0]
    mu = np.mean(x)
    std = np.std(x)
    norm = stats.norm.rvs(mu, std, size=n)
    temp = np.mean(np.tanh(x)) - np.mean(np.tanh(norm))
    return abs(temp)


def get_larger_r_components(source, component):
    l1 = []
    l2 = []
    for j in range(source.shape[1]):
        l1.append(neg(source[:, j]))
    enu_list = list(enumerate(l1))
    enu_list.sort(key=lambda x: x[1], reverse=True)
    for i in range(component):
        l2.append(enu_list[i][0])
    return source[:, l2], l2


def ICA(data, ica_max_iter, r_num):
    ica = FastICA(max_iter=ica_max_iter, whiten=False)
    sources = ica.fit_transform(data)
    r_larger_sources, larger_index = get_larger_r_components(sources, component=r_num)
    return ica, sources, larger_index


def whiten(X):
    cov = np.dot(X.T, X) / X.shape[0]
    U, S, V = np.linalg.svd(cov)
    V= U/np.sqrt(S + 1e-5)
    Xwhite = np.dot(X, V)
    return Xwhite, V


def PCA(train_data, pc_num):
    cov = np.matmul(train_data.T, train_data)/(train_data.shape[0]-1)
    eig_vals, eig_vecs = linalg.eigh(cov)
    eig_vals = np.real(eig_vals)
    eig_vecs = np.real(eig_vecs)
    index = None
    if pc_num == 'mean':
        avg = np.mean(eig_vals)
        indice = np.argsort(-eig_vals)
        eig_vals = eig_vals[indice]
        eig_vecs = eig_vecs[:, indice]
        for i in range(eig_vals.shape[0]):
            if eig_vals[i] < avg:
                index = i
                break
    elif isinstance(pc_num, int):
        index = pc_num
    else:
        raise ValueError('Please enter a correct PC_num value.')
    s = np.diag(eig_vals[:index])
    p = eig_vecs[:, :index]
    return eig_vecs, s, p


def compute_T2andSPE_T(data, L, P_retained):
    t2 = np.zeros(data.shape[0])
    spe_t = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        t2[i] = np.matmul(np.matmul(data[i, :], P_retained.dot(np.linalg.pinv(L)).dot(P_retained.T)), data[i, :].T)
        spe_t[i] = np.power(np.linalg.norm(data[i, :] - np.matmul(P_retained.dot(P_retained.T), data[i, :])), 2)
    t2 = t2
    spe_t = spe_t
    return t2, spe_t


def compute_I2(independent_sources):
    i2 = np.zeros(independent_sources.shape[0])
    for i in range(independent_sources.shape[0]):
        i2[i] = np.matmul(independent_sources[i, :], independent_sources[i, :].T)
    i2 = i2
    return i2


def compute_SPE_I(rec_err):
    spe_I = np.zeros(rec_err.shape[0])
    for i in range(rec_err.shape[0]):
        spe_I[i] = np.power(np.linalg.norm(rec_err[i, :]), 2)
    spe_I = spe_I
    return spe_I


def kde_limit(moni_stas, confi_limit):
    kde = smnp.KDEUnivariate(moni_stas)
    kde.fit()
    index = np.argmin(np.abs(kde.cdf - confi_limit))
    limit = kde.support[index]
    return limit


def farandmdr(moni_vari, limit, fault_intro):
    fa = 0
    md = 0
    for i in range(fault_intro):
        if moni_vari[i] > limit:
            fa += 1
    far = fa/fault_intro
    for j in range(fault_intro, moni_vari.shape[0]):
        if moni_vari[j] <= limit:
            md += 1
    mdr = md/(moni_vari.shape[0]-fault_intro)
    return np.array((far, mdr))
