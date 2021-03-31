import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import gzip
import csv
import time
import argparse

from utils import *
from sharenet import *

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data_dir', dest='data_dir')
	parser.add_argument('-r', '--results_dir', dest='results_dir')
	parser.add_argument('-K', '--K', dest='K',type=int)
	parser.add_argument('-f', '--file_name', dest='file_name')
	parser.add_argument('-nc', '--n_components', dest='n_components',type=int,default=10)
	parser.add_argument('-tol','--tolerance',dest='tolerance',type=float,default=0.001)
	parser.add_argument('-sf','--se_file_name',dest='se_file_name',type=str)

	args = parser.parse_args()
	start = time.time()

	# load standard error
	V = []
	for cluster_no in range(1,args.K+1):
		file_name = 'V.cluster{}.{}'.format(cluster_no,args.se_file_name)
		if file_name.endswith('.gz'):
			network_std = pd.read_csv(os.path.join(args.data_dir,file_name),\
				delimiter='\t',header=None,compression='gzip').values
		else:
			network_std = pd.read_csv(os.path.join(args.data_dir,file_name),\
				delimiter='\t',header=None).values
		print('Loaded standard deviation estimates for cluster {}'.format(cluster_no))
		V.append(network_std)

	# load networks
	pred_network_dict = {}
	for cluster_no in range(1,args.K+1):
		file_name = 'cluster{}.{}'.format(cluster_no,args.file_name)
		if file_name.endswith('.gz'):
			pred_network = pd.read_csv(os.path.join(args.data_dir,file_name),\
				delimiter='\t',header=None,compression='gzip').values
		else:
			pred_network = pd.read_csv(os.path.join(args.data_dir,file_name),\
				delimiter='\t',header=None).values

		print('Loaded network estimates for cluster {}'.format(cluster_no))
		pred_network_dict[cluster_no] = pred_network

	X = [pred_network_dict[cluster_no] \
		for cluster_no in range(1,args.K+1)]

	# weak priors on mean, covariance
	covariance_prior = None
	degrees_of_freedom_prior = args.K
	mean_prior = None

	model = ShareNet(n_components=args.n_components,\
		covariance_prior=covariance_prior,\
		degrees_of_freedom_prior=degrees_of_freedom_prior,\
		mean_prior=mean_prior,\
		init_params='kmeans')

	use_block = args.K > 10

	# fit ShareNet model
	print('Fitting ShareNet model...')
	model.fit(X,V,tol=args.tolerance,use_block=use_block)

	# write revised networks
	revised_networks = model.get_revised_edge_scores()
	for i,cluster_no in enumerate(range(1,args.K+1)):
		n_genes = revised_networks[0].shape[0]
		prefix = 'sharenet.nc{}'.format(args.n_components)
		file_name = 'cluster{}.{}.{}'.format(cluster_no,prefix,args.file_name.split('.gz')[0])
		np.savetxt(os.path.join(args.results_dir,file_name),revised_networks[i],delimiter='\t')

	# save parameters
	file_name = 'sharenet.nc{}.{}'.format(args.n_components,\
		args.file_name.split('.gz')[0])
	np.savetxt(os.path.join(args.results_dir,'m_tilde.{}'.format(file_name)),\
		model.m_tilde.reshape(model.m_tilde.shape[0],-1),delimiter='\t')
	np.savetxt(os.path.join(args.results_dir,'phi.{}'.format(file_name)),\
		model.phi,delimiter='\t')
	np.savetxt(os.path.join(args.results_dir,'covariances.{}'.format(file_name)),\
		model.covariances_.reshape(model.covariances_.shape[0],-1),delimiter='\t')
	np.savetxt(os.path.join(args.results_dir,'means.{}'.format(file_name)),\
		model.means_,delimiter='\t')
