"""
Source code of the paper "Deep Learning of Latent Variable Models for Industrial Process Monitoring"
"DOI: 10.1109/TII.2021.3093386"
"https://ieeexplore.ieee.org/document/9468377"
If you use this code in your work, please cite the above paper. Thank you!

@author: Xiangyin Kong
@github account: kxytim
"""

import numpy as np


def bayesian_fusion(sts_each_layer, sts_limits, mu, confi_limit, eta, past_y_samples):
    a = []
    b = []
    c = []
    d = []
    for i in range(len(sts_each_layer)):
        temp1 = []
        temp2 = []
        temp3 = []
        temp4 = []
        for j in range(len(sts_each_layer[0])):
            temp1.append(np.exp(-mu*sts_limits[i]/sts_each_layer[i][j]))
            temp2.append(np.exp(-mu*sts_each_layer[i][j]/sts_limits[i]))
            temp3.append((1-confi_limit)*temp1[j]+confi_limit*temp2[j])
            temp4.append((1-confi_limit)*temp1[j]/temp3[j])
        a.append(temp1)
        b.append(temp2)
        c.append(temp3)
        d.append(temp4)
    e = []
    for ii in range(len(sts_each_layer)):
        temp5 = []
        for jj in range(len(sts_each_layer[0])):
            w = np.ones(sts_each_layer[0][jj].shape) * eta
            for kk in range(past_y_samples, sts_each_layer[0][jj].shape[0]):
                count = 1
                sum1 = 0
                while count <= past_y_samples:
                    sum1 += d[ii][jj][kk - past_y_samples + count]
                    count += 1
                sum1 /= past_y_samples
                if (d[ii][jj][kk] > 1 - confi_limit) & (sum1 > 1 -confi_limit):
                    w[kk] = 1 / eta
            temp5.append(w)
        e.append(temp5)
    for iii in range(len(sts_each_layer[0])):
        sum2 = 0
        for jjj in range(len(sts_each_layer)):
            sum2 += e[jjj][iii]
        for kkk in range(len(sts_each_layer)):
            e[kkk][iii] /= sum2
    f = []
    for iii in range(len(sts_each_layer[0])):
        sum3 = 0
        for jjj in range(len(sts_each_layer)):
            sum3 += e[jjj][iii]*d[jjj][iii]
        f.append(sum3)
    return np.array(f), np.array(e)

