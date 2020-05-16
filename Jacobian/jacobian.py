"""
Helper Functions to compute the spectrum of J^T*J and J*J^T, 
were J is the Jacobian of the Neural Network J = df(x)/dx.
the diagonal J will be of dimension training_data_size*(output_dim*data_dim)
while the Full J will be of dimension training_data_size*(output_dim*training_data_size*data_dim)
"""