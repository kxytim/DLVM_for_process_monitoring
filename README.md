# DLVM_for_process_monitoring
Source code of the paper "Deep Learning of Latent Variable Models for Industrial Process Monitoring".

"DOI: 10.1109/TII.2021.3093386"

"https://ieeexplore.ieee.org/document/9468377"

**If you use this code in your work, please cite the above paper. Thank you!**

@author: Xiangyin Kong

@github account: kxytim

**Code description:**

utils.py: the basic functions used in the model.

deep_PCA_ICA.py: the proposed deep PCA-ICA model.

Bayesian_Fusion.py: the proposed Bayesian fusion strategy to integrate the information at different layers.

DPI_monitor_TE.py: process monitoring for TE process.

**For a fast test, please execute:**

```python DPI_monitor_TE.py```

**Then you may get the following results:**

Start running

The average FAR and MDR of DB_T2 are 2.887% and 18.786%.

The average FAR and MDR of DB_SPE_T are 1.577% and 18.792%.

The average FAR and MDR of DB_I2 are 2.470% and 19.232%.

The average FAR and MDR of DB_SPE_I are 2.708% and 18.476%.

The average FAR and MDR of ODBS are 3.214% and 16.738%.

The total time of training and testing is: 87.223s.

End