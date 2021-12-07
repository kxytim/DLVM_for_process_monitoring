"""
Source code of the paper "Deep Learning of Latent Variable Models for Industrial Process Monitoring"
"DOI: 10.1109/TII.2021.3093386"
"https://ieeexplore.ieee.org/document/9468377"
If you use this code in your work, please cite the above paper. Thank you!

@author: Xiangyin Kong
@github account: kxytim, https://github.com/kxytim
"""


import time
import numpy as np
import warnings
from utils import *
from deep_PCA_ICA import *
from Bayesian_Fusion import *

warnings.filterwarnings("ignore")


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

# Hyper Parameters
LAYERS_NUMBER = 3
PC_NUMBER = 'mean'
IC_NUMBER = 15
ICA_MAX_ITER = 3000
CONFI_LIMIT = 0.95
FAULT_INTRODUCE_AT = 160

MU = 1.3
ETA = 0.01
PAST_Y_SAMPLES = 5

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)  # For reproduction, fix the random seed.


print('Start running')
# Load TE process data
normal_data = load_data('TE_data/d00.dat').T
normal_data_validate = load_data('TE_data/d00_te.dat')
fault_mode = list()
for fault_num in range(1, 10):
    fault_mode.append(load_data('TE_data/d0{}_te.dat'.format(fault_num)))
for fault_num in range(10, 22):
    fault_mode.append(load_data('TE_data/d{}_te.dat'.format(fault_num)))

start_time = time.time()
# Build DPI model for monitoring TE process
dpi_3l = DPI(LAYERS_NUMBER, PC_NUMBER, IC_NUMBER, ICA_MAX_ITER, CONFI_LIMIT, FAULT_INTRODUCE_AT)
dpi_3l.fit(normal_data)
Normal_sts = dpi_3l.transform(normal_data_validate)
Sts_limits = dpi_3l.get_limits(Normal_sts)
Sum_Detect_results = 0
for fault in range(21):
    Sum_Detect_results += dpi_3l.detect(fault_mode[fault])[1]
Avg_Detect_results = Sum_Detect_results/21
# The meaning of Avg_Detect_results: shape[0] means layer, shape[1] means detect results (far, mdr),
# shape[2] means monitoring statistics [T2, SPE_T, I2, SPE_I]

# Use Bayesian inference and weighting strategy to generate deep Bayesian statistic
# Detect all faults at once
faults_total_num = 21
All_T2 = np.zeros((LAYERS_NUMBER, faults_total_num, fault_mode[0].shape[0]))
All_SPE_T = np.zeros(All_T2.shape)
All_I2 = np.zeros(All_T2.shape)
All_SPE_I = np.zeros(All_T2.shape)
for fault in range(faults_total_num):
    All_T2[:, fault, :] = dpi_3l.detect(fault_mode[fault])[0][:, :, 0]
    All_SPE_T[:, fault, :] = dpi_3l.detect(fault_mode[fault])[0][:, :, 1]
    All_I2[:, fault, :] = dpi_3l.detect(fault_mode[fault])[0][:, :, 2]
    All_SPE_I[:, fault, :] = dpi_3l.detect(fault_mode[fault])[0][:, :, 3]
DB_T2, Coef1 = bayesian_fusion(All_T2, dpi_3l.sts_limits[:, 0], MU, CONFI_LIMIT, ETA, PAST_Y_SAMPLES)
DB_SPE_T, Coef2 = bayesian_fusion(All_SPE_T, dpi_3l.sts_limits[:, 1], MU, CONFI_LIMIT, ETA, PAST_Y_SAMPLES)
DB_I2, Coef3 = bayesian_fusion(All_I2, dpi_3l.sts_limits[:, 2], MU, CONFI_LIMIT, ETA, PAST_Y_SAMPLES)
DB_SPE_I, Coef4 = bayesian_fusion(All_SPE_I, dpi_3l.sts_limits[:, 3], MU, CONFI_LIMIT, ETA, PAST_Y_SAMPLES)
ODBS, Coef5 = bayesian_fusion([DB_T2, DB_SPE_T, DB_I2, DB_SPE_I], [1-CONFI_LIMIT, 1-CONFI_LIMIT, 1-CONFI_LIMIT,
                                                                   1-CONFI_LIMIT], MU, CONFI_LIMIT, ETA, PAST_Y_SAMPLES)

DB_T2_results = np.zeros((len(DB_T2), 2))
DB_SPE_T_results = np.zeros(DB_T2_results.shape)
DB_I2_results = np.zeros(DB_T2_results.shape)
DB_SPE_I_results = np.zeros(DB_T2_results.shape)
ODBS_results = np.zeros(DB_T2_results.shape)
for fault in range(21):
    DB_T2_results[fault] = farandmdr(DB_T2[fault], 1-CONFI_LIMIT, 160)
    DB_SPE_T_results[fault] = farandmdr(DB_SPE_T[fault], 1-CONFI_LIMIT, 160)
    DB_I2_results[fault] = farandmdr(DB_I2[fault], 1-CONFI_LIMIT, 160)
    DB_SPE_I_results[fault] = farandmdr(DB_SPE_I[fault], 1-CONFI_LIMIT, 160)
    ODBS_results[fault] = farandmdr(ODBS[fault], 1-CONFI_LIMIT, 160)

print('The average FAR and MDR of DB_T2 are {:.3f}% and {:.3f}%.'.format(100*np.mean(DB_T2_results, axis=0)[0],
                                                                        100*np.mean(DB_T2_results, axis=0)[1]))
print('The average FAR and MDR of DB_SPE_T are {:.3f}% and {:.3f}%.'.format(100*np.mean(DB_SPE_T_results, axis=0)[0],
                                                                        100*np.mean(DB_SPE_T_results, axis=0)[1]))
print('The average FAR and MDR of DB_I2 are {:.3f}% and {:.3f}%.'.format(100*np.mean(DB_I2_results, axis=0)[0],
                                                                        100*np.mean(DB_I2_results, axis=0)[1]))
print('The average FAR and MDR of DB_SPE_I are {:.3f}% and {:.3f}%.'.format(100*np.mean(DB_SPE_I_results, axis=0)[0],
                                                                        100*np.mean(DB_SPE_I_results, axis=0)[1]))
print('The average FAR and MDR of ODBS are {:.3f}% and {:.3f}%.'.format(100*np.mean(ODBS_results, axis=0)[0],
                                                                        100*np.mean(ODBS_results, axis=0)[1]))

print('The total time of training and testing is: {:.3f}s.'.format(time.time()-start_time))
print('End')
