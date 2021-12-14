# DLVM_for_process_monitoring
Source code of the paper "Deep Learning of Latent Variable Models for Industrial Process Monitoring".

"DOI: 10.1109/TII.2021.3134251" "https://ieeexplore.ieee.org/document/9647968"

First author: Xiangyin Kong

Corresponding author: Prof. Zhiqiang Ge (Google Scholar: https://scholar.google.com/citations?user=g_EMkuMAAAAJ&hl=zh-CN)

**If the paper or the code is helpful to your work, please cite the above paper. Thank you!**

@author: Xiangyin Kong

@github account: kxytim, https://github.com/kxytim

**Code description:**

utils.py: the basic functions used in the model.

deep_PCA_ICA.py: the proposed deep PCA-ICA model.

Bayesian_Fusion.py: the proposed Bayesian fusion strategy to integrate the information at different layers.

DPI_monitor_TE.py: use the proposed model to monitor TE process.

**For a fast test, please execute:**

```python DPI_monitor_TE.py```

**Then you may get the following results:**

Start running

The average FAR and MDR of DB_T2 are 2.887% and 18.786%.

The average FAR and MDR of DB_SPE_T are 1.577% and 18.792%.

The average FAR and MDR of DB_I2 are 2.292% and 18.643%.

The average FAR and MDR of DB_SPE_I are 2.560% and 19.601%.

The average FAR and MDR of ODBS are 3.333% and 16.702%.

The total time of training and testing is: 80.858s.

End