"""
Source code of the paper "Deep Learning of Latent Variable Models for Industrial Process Monitoring"
"DOI: 10.1109/TII.2021.3093386"
"https://ieeexplore.ieee.org/document/9468377"
If you use this code in your work, please cite the above paper. Thank you!

@author: Xiangyin Kong
@github account: kxytim
"""

import numpy as np
from sklearn import preprocessing
from scipy import linalg
from utils import *


# Hyper Parameters


class DPI(object):
    def __init__(self, layer_num, PC_num, IC_num, ica_max_iter, confi_limit, fault_intro):
        self.layer_num = layer_num
        self.PC_num = PC_num
        self.IC_num = IC_num
        self.ica_max_iter = ica_max_iter
        self.confi_limit = confi_limit
        self.fault_intro = fault_intro
        self.P_ary = None
        self.L_list = []
        self.P_retained_list = []
        self.scaler_PCA_list = []
        self.scaler_ICA = None
        self.V_ary = None
        self.ica_transformer_list = []
        self.larger_index_ary = np.zeros((self.layer_num, self.IC_num), dtype=np.int32)
        self.sts_limits = np.zeros((self.layer_num, 4))

    def fit(self, X):
        self.P_ary = np.zeros((self.layer_num, X.shape[1], X.shape[1]))
        self.V_ary = np.zeros(self.P_ary.shape)
        T_ary = np.zeros((self.layer_num, X.shape[0], X.shape[1]))
        S_ary = np.zeros(T_ary.shape)
        for layer in range(self.layer_num):
            if layer == 0:
                scaler_PCA = preprocessing.StandardScaler().fit(X)
                self.scaler_PCA_list.append(scaler_PCA)
                X = scaler_PCA.transform(X)
                P, L, P_retained = PCA(X, pc_num=self.PC_num)
                T = X.dot(P)
                self.P_ary[layer] = P
                self.L_list.append(L)
                self.P_retained_list.append(P_retained)
                T_ary[layer] = T
                X_residual = X-X.dot(P_retained.dot(P_retained.T))
                scaler_ICA = preprocessing.StandardScaler().fit(X_residual)
                self.scaler_ICA = scaler_ICA
                X_residual = self.scaler_ICA.transform(X_residual)
                X_whiten, V = whiten(X_residual)
                self.V_ary[layer] = V
                ica, S, lar_index = ICA(X_whiten, self.ica_max_iter, self.IC_num)
                self.ica_transformer_list.append(ica)
                S_ary[layer] = S
                self.larger_index_ary[layer] = lar_index
            else:
                scaler_PCA = preprocessing.StandardScaler().fit(T_ary[layer-1])
                self.scaler_PCA_list.append(scaler_PCA)
                T_std = scaler_PCA.transform(T_ary[layer-1])
                P, L, P_retained = PCA(T_std, pc_num=self.PC_num)
                T = T_std.dot(P)
                self.P_ary[layer] = P
                self.L_list.append(L)
                self.P_retained_list.append(P_retained)
                T_ary[layer] = T
                S_whiten, V = whiten(S_ary[layer-1])
                self.V_ary[layer] = V
                ica, S, lar_index = ICA(S_whiten, self.ica_max_iter, self.IC_num)
                self.ica_transformer_list.append(ica)
                S_ary[layer] = S
                self.larger_index_ary[layer] = lar_index
        return self

    def transform(self, X):
        moni_statis = np.zeros((self.layer_num, X.shape[0], 4))
        T, S = None, None
        for layer in range(self.layer_num):
            if layer == 0:
                X = self.scaler_PCA_list[layer].transform(X)
                T = X.dot(self.P_ary[layer])
                moni_statis[layer, :, 0] = compute_T2andSPE_T(X, self.L_list[layer], self.P_retained_list[layer])[0]
                moni_statis[layer, :, 1] = compute_T2andSPE_T(X, self.L_list[layer], self.P_retained_list[layer])[1]
                X_residual = X - X.dot(self.P_retained_list[layer].dot(self.P_retained_list[layer].T))
                X_residual = self.scaler_ICA.transform(X_residual)
                S = self.ica_transformer_list[layer].transform(X_residual.dot(self.V_ary[layer]))
                moni_statis[layer, :, 2] = compute_I2(S[:, self.larger_index_ary[layer]])
                recons = np.dot(np.dot(S[:, self.larger_index_ary[layer]], linalg.pinv(
                    self.ica_transformer_list[layer].components_.T[:, self.larger_index_ary[layer]])),
                                linalg.pinv(self.V_ary[layer]))
                recons_err = X_residual-recons
                moni_statis[layer, :, 3] = compute_SPE_I(recons_err)
            else:
                T_std = self.scaler_PCA_list[layer].transform(T)
                T = T_std.dot(self.P_ary[layer])
                moni_statis[layer, :, 0] =compute_T2andSPE_T(T_std, self.L_list[layer], self.P_retained_list[layer])[0]
                moni_statis[layer, :, 1] = compute_T2andSPE_T(T_std, self.L_list[layer], self.P_retained_list[layer])[1]
                S_previous = S
                S = self.ica_transformer_list[layer].transform(S_previous.dot(self.V_ary[layer]))
                moni_statis[layer, :, 2] = compute_I2(S[:, self.larger_index_ary[layer]])
                recons = np.dot(np.dot(S[:, self.larger_index_ary[layer]], linalg.pinv(
                    self.ica_transformer_list[layer].components_.T[:, self.larger_index_ary[layer]])),
                                linalg.pinv(self.V_ary[layer]))
                recons_err = S_previous - recons
                moni_statis[layer, :, 3] = compute_SPE_I(recons_err)
        return moni_statis

    def get_limits(self, normal_moni_statis):
        for layer in range(self.layer_num):
            for i in range(4):
                self.sts_limits[layer, i] = kde_limit(normal_moni_statis[layer, :, i], self.confi_limit)
        return self.sts_limits

    def detect(self, X):
        test_moni_statis =self.transform(X)
        detect_results = np.zeros((self.layer_num, 2, 4))
        for layer in range(self.layer_num):
            for i in range(4):
                detect_results[layer, :, i] = farandmdr(test_moni_statis[layer, :, i], self.sts_limits[layer, i], self.fault_intro)
        return test_moni_statis, detect_results

