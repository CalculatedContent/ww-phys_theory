"""
Helper Functions to compute the spectrum of J^T*J and J*J^T, 
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
