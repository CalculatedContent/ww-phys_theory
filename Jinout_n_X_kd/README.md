### Colab Notebooks for  input output Jacobian, shaped by n X kd

(Also just called Jacobian in other folders)

- n:  number of test (or training) data ppints
- k:  number of output classes 
- d:  dimension of input

For CIFAR10, n=10000, k=10, and d=3*32*32

Resnet CIFAR10  models trained (in memory) (resnet20, 56, 110, 164bn, 272bn)

Jacobian ESDs computed over test data in full (using LAPAC eigh)

#### Diagonal Jaocobian

refers to diagonal elements of $\mathbf{J}^{T}\mathbf{J}$
and not the diag_j() methods in the jacobian.py lib


- [J_Diag_PLfits.ipynb](https://github.com/CalculatedContent/ww-phys_theory/blob/master/Jinout_n_X_kd/J_Diag_PLfits.ipynb) comparison of PL alpha fits on ResNet CIFAR10 models 
  Reads diagonal  $\mathbf{J}^{T}\mathbf{J}$ from local  csv files on Google Drive

- [Jacobians_Resnets_old.ipynb](https://github.com/CalculatedContent/ww-phys_theory/blob/master/Jinout_n_X_kd/Jacobians_Resnets_old.ipynb)  old code for computing Jacobian diagonal.  
  Did *not* set model.eval() so results may be off


#### Full Jacobians

- [Make_n_Save_JJMat_ResNets.ipynb](https://github.com/CalculatedContent/ww-phys_theory/blob/master/Jinout_n_X_kd/Make_n_Save_JJMat_ResNets.ipynb)   Make ResNet Jacobians correlation matrix JJMat in memory and save to Google Drive


#### Batched Calculations

Long calculations of very large Jacobians have been batches and the
correlation matrix $\mathbf{J}^{T}\mathbf{J}$ stored on Google Drive





