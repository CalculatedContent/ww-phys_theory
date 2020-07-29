### Colab Notebooks for  input output Jacobian, shaped by n X kd

(Also jut called Jacobian in other folders)

- n:  number of test (or training) data ppints
- k:  number of output classes 
- d:  dimension of input

For CIFAR10, n=10000, k=10, and d=3*32*32

Resnet CIFAR10  models trained (in memory) (resnet20, 56, 110, 164bn, 272bn)

Jacobian ESDs computed over test data in full (using LAPAC eigh)

#### Diagonal Jaocobian

refers to diagonal elements of $\mathbf{J}^{T}\mathbf{J}$
and not the diag_j() methods in the jacobian.py lib



