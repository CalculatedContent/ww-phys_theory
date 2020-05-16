"""
Helper Functions to compute the spectrum of J^T @ J and J @ J^T, 
were J is the Jacobian of the Neural Network J = df(x)/dx.
the diagonal J will be of dimension training_data_size*(output_dim*data_dim)
while the Full J will be of dimension training_data_size*(output_dim*training_data_size*data_dim)
"""

from torch.autograd.gradcheck import zero_gradients

def compute_jacobian(inputs, output):
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


def jacobian_diagonal(model, data_loader, batch_size, num_classes=10, device='cuda:0', data_dim=3*32*32):
  '''compute J(J*v) diagnonal elements , where J is the jacobian,'''

  # compute Jdiag
  Jdiag = []
  model = model.to(device)

  for batch, data in enumerate(data_loader):
    features, _ = data
    features = features.to(device)

    features = torch.autograd.Variable(features, requires_grad=True)
    out = model(features)

    J = compute_jacobian(features, out)# create_graph=True)
    J = J.view(batch_size,num_classes*data_dim)
    Jt = J.clone().transpose_(0,1)
    batch_diag = torch.mm(J,Jt).to('cpu') #
    del J, Jt
    torch.cuda.empty_cache()


    Jdiag.append(batch_diag.to('cpu').numpy())

    del batch_diag
    torch.cuda.empty_cache()

  return np.array(Jdiag)

