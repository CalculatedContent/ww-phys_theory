"""
Helper Functions to compute the spectrum of J^T @ J and J @ J^T,
were J is the Jacobian of the Neural Network J = df(x)/dx as a function of it's input (the data, not the weights).
We clarify some nomenclature. The diagonal Jacobian is constructed of terms only of the form dJ(x_i)/dx_i.
It will be of dimension training_data_size*(output_dim*data_dim). The full J will contains terms of the form
dJ(x_i)/dx_j, which will be of dimension training_data_size*(output_dim*training_data_size*data_dim)
The diagonal of M = J @ J^T or M = J^T @ J is the main diagonal of M.
"""

import torch
from torch.autograd.gradcheck import zero_gradients
import numpy as np
import math

def batch_diagJ(inputs, output):
	"""Computes the diagonal Jacobian by input batches.

	Input: Input for the function for which the Jacobian will
	computed. It will be batch_size*data_dim. Make sure that the
	input is flagged as requires_grad=True with the torch.autograd.Variable
	wrapper.

	Output: Output of the function for which the Jacobian will
	be computed. It will be batch_size*classes

	Return: Jacobian of dimension batch_size*classes*data_dim
	"""
	assert inputs.requires_grad

	num_classes = output.size()[1] #0 index is batch

	jacobian = torch.zeros(num_classes, *inputs.size())
	grad_output = torch.zeros(*output.size())
	if inputs.is_cuda:
		grad_output = grad_output.cuda()
		jacobian = jacobian.cuda()

	#zero out gradients
	#compute gradient for one of the classes
	for i in range(num_classes):
		zero_gradients(inputs)
		grad_output.zero_()
		grad_output[:,i] = 1
		output.backward(grad_output, retain_graph=True)
		jacobian[i] = inputs.grad.data

	return torch.transpose(jacobian, dim0=0, dim1=1)

def construct_diagJ(
		model,
		data_loader,
		batch_size,
		device='cuda:0',
		num_classes=10,
		data_dim = 3*32*32,
		class_label_filter = None,
		example_x_class_rowspace = False
		):
	"""Constructs the diagonal J matrix from batches.

	Input: Model, data_loader, batch_size, device, num_classes, data_dim.
	Optional Arguments: device, num_classes (default 10), data_dim (default: 3072), class_label_filter (default: None), example_x_class_rowspace (default: False)
	Return: Diagonal Jacobian of dimension (len(data_loader)*batch_size, num_classes*data_dim).
	"""
	Js = []
	model.eval()
	model = model.to(device)

	for batch, data in enumerate(data_loader):
		features, label = data
		if class_label_filter == None:
			inputs = features.to(device)
			inputs.requires_grad=True
			outputs = model(inputs)
			J = batch_diagJ(inputs, outputs)
			Js.append(J)
		else:
			indices = [i for i, x in enumerate(label) if x == class_label_filter]
			features = features[indices]
			inputs = features.to(device)
			inputs.requires_grad=True
			outputs = model(inputs)
			J = batch_diagJ(inputs, outputs)
			Js.append(J)
	#Need to double check this reshaping is correct.
	if class_label_filter == None:
		full_J = torch.stack(Js, dim=0)
		if example_x_class_rowspace == False:
			full_J = full_J.reshape(len(data_loader)*batch_size, num_classes*data_dim)
		else:
			full_J = full_J.reshape(len(data_loader)*batch_size*num_classes, data_dim)
	else:
		full_J = torch.cat(Js)
		examples = full_J.shape[0]
		if example_x_class_rowspace == False:
			full_J = full_J.reshape(examples, num_classes*data_dim)
		else:
			full_J = full_J.reshape(examples*num_classes, data_dim)
	return full_J

def diagonal_JJT(
		model,
		data_loader,
		batch_size,
		num_classes=10,
		device='cuda:0',
		data_dim=3*32*32):
	"""Compute the main diagonal of JJ^T, where J is the diagonal Jacobian.

	Input: Model, data_loader, batch_size
	Optional arguments: num_classes (default: 10), device (default: cuda:0), data_dim (default: 3072)
	Return: Array of len(data_loader)*batch_size with the main diagonal of JJ^T.
	"""
	Jdiag = []
	model = model.to(device)

	for batch, data in enumerate(data_loader):
		features, _ = data
		features = features.to(device)
		features = torch.autograd.Variable(features, requires_grad=True)
		out = model(features)
		J = compute_jacobian(features, out)# create_graph=True)
		J = J.reshape(batch_size,num_classes*data_dim)
		Jt = J.clone().transpose_(0,1)
		batch_diag = torch.mm(J,Jt).to('cpu') #
		del J, Jt
		torch.cuda.empty_cache()

	for ib in range(batch_size):
		Jdiag.append(batch_diag[ib, ib].to('cpu').numpy())

	del batch_diag
	torch.cuda.empty_cache()

	return np.array(Jdiag)

def sketch_jl_JJT(J, dim=5000, device="cuda:0"):
	"""Creates a Johnson-Lindenstrauss sketch of J of dimension dim, and computes M = J @ JT.

	Input: Jacobian, J
	Optional: dim (default: 5000)
	Return: M = PJ @ (PJ)^T, were P is a JL matrix.
	"""
	n, _ = J.shape
	P = 1/dim*torch.empty(dim, n, device=device).normal_(mean=0,std=1.)
	P_J = P @ J
	M = P_J @ P_J.t()
	del P_J

	return M

def power_method(M, iterations=100, device="cuda:0"):
	"""Computes the top eigenvalue of a matrix. This needs to be computed for kernel PM.

	Input: the Jacobian correlation matrix, M
	Optional: iterations (default: 100), device (default: cuda:0)
	Return: the largest eigenvalue of M.
	"""
	n, _ = M.shape
	vk = torch.empty(n, device=device).normal_(mean=0, std=1.)

	for i in range(iterations):
		vk1 = M @ vk
		vk1_norm = torch.norm(vk1)
		vk = vk1 / vk1_norm

	top_eig = vk @ M @ vk
	del vk
	del vk1

	return top_eig

def orthonormal(v_list):
	"""This generates a vector x, orthogonal to a set of vectors, v_list.
	Input: A list of vectors, v_list.
	Return: Orthogonalized vector, x, orthogonal to the vectors in the list, v_list.
	The solution to A^Tx = 0 will give this, where A has column-wise entries
	of v in v_list.
	"""
	n = len(v_list)
	m = len(v_list[0])
	A = torch.zeros((n, m))
	b = torch.zeros(n)
	for i in range(n):
		A[i, :] = v_list[i]
	x, _ = torch.solve(b, A.t())
	x = x/torch.norm(x)
	return x





def slq(M, n_vec=20, m=100, device="cuda:0"):
	"""An implemention of the Stochastic Lanczos Quadrature to compute the spectral density of M = JJ^T.

	Input: the correlation matrix of the Jacobian M.
	Optional: number of random vectors, n_vec (default: 20)
	number of iterations, m (default: 100)
	Return: List of arrays of eigenvalues and densities of len(n_vec) and array size (m).

	ToDo: Batch computation, orthogonalizing v in the else statement, Pearlmutter's trick.
	"""
	n, _ = M.shape
	eigs = []
	ws = []
	vs = []

	for k in range(n_vec):
		print("Iteration {} of n_vec".format(k))
		v = torch.randint(high=2, size=(n,), device=device, dtype=torch.float32)
		v[v == 0] = -1
		v = v/torch.norm(v)
		vs.append(v)
		T = torch.zeros(m,m, device=device)

		for i in range(m):
			if i == 0:
				w = M @ v
				a = w @ v
				w = w - a*v
				v_j1 = v
				T[i][i] = a
			else:
				b = torch.norm(w)
				if b != 0:
					v = w/b
				else:
					#This old bit of code assumes that with high probability, a new random vector will be orthogonal to others.
					# Leaving in for posterity.
					#v = torch.randint(high=2, size=(n,), device=device, dtype=torch.float32)
					#v[v==0] = -1 #make it rademacher
					#v = v/torch.norm(v)

					v = orthonormal(vs)
					vs.append(v)

				w = M @ v
				a = w @ v
				w = w - a*v - b*v_j1
				v_j1 = v

				T[i][i] = a
				T[i-1][i] = b #there is no beta 0
				T[i][i-1] = b

		T = T.to("cpu") #do eig on CPU, small enough anyway
		#eig, U = torch.symeig(T, eigenvectors = True)
		eig, w = torch.eig(T, eigenvectors = True)
		eigs.append(list(eig[:,0]))
		w = w[0,:]**2
		ws.append(list(w))

	return eigs, ws

def kernel_pm(M, m= 20, n_vec=100, device="cuda:0", power_it=100):
	"""An implementation of the Kernel Polynomial Method as outlined in Lin, Saad, Yang.

	Input: Jacobian correlation matrix M. Degree of Chebyshev expansion, m.
	Optional: Number of random vectors, n_vec (default:100), device (default: cuda:0), power_it (default: 100)
	Return: Coefficients for the chebyshev expansion, mu. They are the coefficients for the Chebyshev series
	1/sqrt(1-t^2)sum_k mu_k T_k(t).

	ToDo: Batch computatoin, Pearlmutter's trick.
	Note: A lot of tricks were used such that M is only made in memory once. If Pearlmutter's trick is implemented, these
	tricks could be removed.
	"""
	n, _ = M.shape
	a = 0 #smallest eigenvalue of M
	print("Computing top eigenvalue.")
	# We want to compute the power method on the GPU
	vk = torch.empty(n, device=device).normal_(mean=0, std=1.)
	for i in range(power_it):
		vk1 = M @ vk
		vk1_norm = torch.norm(vk1)
		vk = vk1 / vk1_norm
	b = vk @ M @ vk
	del vk
	del vk1
	
	
	# The following segment of code was written for M of size 50k x 50k (the training set of CIFAR10), which just fit on a P100. 
	# For the test set Jacobian (10k x 10k) it is not optimal.
	M = M.to("cpu")
	b = b.to("cpu")
	torch.cuda.empty_cache() #This line of code doesn't seem to work when nested inside a function.
	print("Finished top eigenvalue, computing mu")

	# We want to rescale M = (M - ((b + a)/2)*I)/((b-a)/2). M needs to be rescaled for Chebyshev basis
	# This is done in a for loop so it does not need to be made in memory.
	print("Rescaling M")
	for i in range(n):
	    M[i][i] = M[i][i] - (b+a)/2
	# This is done on the cpu, as you need a 2*size(M) to do this
	M = M/((b-a)/2)
	print("Done Rescaling M")
	M = M.to(device) #send M to gpu.

	zeta = torch.zeros(m, device = device)
	mu = torch.zeros(m, device = device)

	for l in range(n_vec): #number of vecs
		#print("Iteration {} of computing mu".format(l))
		v0 = torch.empty(n, device=device).normal_(mean=0, std=1.)
		for k in range(m): #cheby degree
			if k == 0:
				zeta[k] = zeta[k] + v0 @ v0
				vk = M @ v0
			elif k == 1:
				zeta[k] = zeta[k] + v0 @ vk
				# vk = 2* M @ vk - vk
				# Need to break up computation to fit on GPU
				tmp = M @ vk #- vk
				tmp = 2*tmp
				vk1 = vk
				vk = tmp - v0
				del tmp
			else:
				zeta[k] = zeta[k] + v0 @ vk
				tmp = M @ vk #- vk
				tmp = 2*tmp
				p = vk
				vk = tmp - vk1
				vk1 = p
				del tmp
				del p
		del v0
	zeta = zeta/n_vec
	for k in range(m):
		if k == 0:
			mu[k] = 1/(n*math.pi)*zeta[k]
		else:
			mu[k] = 2/(n*math.pi)*zeta[k]
	return mu.detach().cpu().numpy()

#
