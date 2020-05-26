"""
Helper Functions to compute the spectrum of J^T @ J and J @ J^T, 
were J is the Jacobian of the Neural Network J = df(x)/dx as a function of it's input (the data, not the weights).
We clarify some nomenclature. The diagonal Jacobian is constructed of terms only of the form dJ(x_i)/dx_i. 
It will be of dimension training_data_size*(output_dim*data_dim). The full J will contains terms of the form 
dJ(x_i)/dx_j, which will be of dimension training_data_size*(output_dim*training_data_size*data_dim)
The diagonal OF M = J @ J^T or M = J^T @ J is the main diagonal of M.
"""

import torch
from torch.autograd.gradcheck import zero_gradients

def batch_diagJ(inputs, output):
	"""
	input: input for the function for which the Jacobian will
	computed. It will be batch_size*data_dim. Make sure that the
	input is flagged as requires_grad=True with the torch.autograd.Variable
	wrapper. 

	output: output of the function for which the Jacobian will
	be computed. It will be batch_size*classes

	return: Jacobian of dimension batch_size*classes*data_dim
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

def construct_diagJ(model, data_loader, batch_size, device='cuda:0', num_classes=10, data_dim = 3*32*32):
	"""
	constructs the diagonal J matrix from batches.
	"""
	Js = []
	model.eval()
	model = model.to(device)

	for batch, data in enumerate(data_loader):
		features, _ = data

		inputs = features.to(device)
		inputs.requires_grad=True
		outputs = model(inputs)

		J = batch_diagJ(inputs, outputs)

		Js.append(J)

	full_J = torch.stack(Js, dim=0)
	full_J = full_J.reshape(len(train_loader)*batch_size, num_classes*data_dim)

	return full_J

def diagonal_JJT(model, data_loader, batch_size, num_classes=10, device='cuda:0', data_dim=3*32*32):
  '''compute J(J*v) diagonal elements , where J is the jacobian,'''

  # compute Jdiag
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

def sketch_JL_JJT(J, dim=5000):
	"""
	Creates a JL sketch of M = JJT of dimension dim.
	"""
	n, _ = J.shape

	P = 1/dim*torch.empty(dim, n).normal_(mean=0,std=1.)

	P_J = P @ J

	M = P_J @ P_J.t()

	return M


