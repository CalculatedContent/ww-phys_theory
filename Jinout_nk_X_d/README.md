### Colab Notebooks for  input output Jacobian, shaped by nk X d

- n:  number of test (or training) data ppints
- k:  number of output classes 
- d:  dimension of input

For CIFAR10, n=10000, k=10, and d=3*32*32

Resnet CIFAR10  models trained (in memory) (resnet20, 56, 110, 164bn, 272bn)

Jacobian ESDs computed over test data in full (using LAPAC eigh)

ESD to fit to power law, near the peak of the ESD, and at the tail

