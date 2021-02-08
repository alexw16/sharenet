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
	parser.add_argument('-thr','--threshold',dest='threshold',type=float)
	parser.add_argument('-nr','--num_regs',dest='num_regs',type=int)
	parser.add_argument('-tol','--tolerance',dest='tolerance',type=float,default=0.001)
	parser.add_argument('-pr','--prior',dest='prior',type=str,default='identity.weak')
	parser.add_argument('-V','--V_noise',dest='V_noise',type=str,default='specific')
	parser.add_argument('-sf','--se_file_name',dest='se_file_name',type=str)
	# parser.add_argument('-pr','--predict',dest='predict',type=int,default=0)

	args = parser.parse_args()

	regtarget_data = np.loadtxt(os.path.join(args.data_dir,'regtarget.txt'),\
		delimiter='\t',dtype=str)
	regtarget_dict = {int(target): list(map(int,tfs.split(';'))) for target,tfs \
		in regtarget_data if len(tfs) > 0}
	target_inds = sorted(list(regtarget_dict.keys()))
	tf_inds = regtarget_dict[target_inds[0]]
	target_inds = sorted(list(set(target_inds + tf_inds))) # include TF-TF connections
	n_genes = int(max(set(tf_inds + target_inds))+1)
	print(n_genes)

	start = time.time()

	# load standard error
	V = []
	for cluster_no in range(1,args.K+1):
		file_name = 'V.cluster{}.{}.txt'.format(cluster_no,args.se_file_name)
		network_std = pd.read_csv(os.path.join(args.results_dir,file_name),delimiter='\t',header=None).values
		if 'pidc' not in args.file_name:
			network_std = network_std[target_inds][:,tf_inds]

		if 'specific' in args.V_noise:
			V.append(network_std)
		elif 'mean' in args.V_noise:
			network_std[:] = network_std.mean()
			V.append(network_std)

	network_dict = {}
	for trial_no in range(1,2):
		pred_network_dict = {}
		for cluster_no in range(1,args.K+1):
			file_name = 'trial{}.cluster{}.{}'.format(trial_no,cluster_no,args.file_name)
			pred_network= load_network(args.results_dir,file_name,n_genes)
			if 'pidc' not in args.file_name:
				pred_network = pred_network[target_inds][:,tf_inds]
			pred_network_dict[cluster_no] = pred_network

		network_dict[trial_no] = pred_network_dict

	print('Setup: {} seconds'.format(np.round(time.time()-start,3)))
	for trial_no in range(1,2):
		print('Trial {}'.format(trial_no),args.file_name)

		X = [network_dict[trial_no][cluster_no] \
			for cluster_no in range(1,args.K+1)]

		if args.threshold != None:
			# N = int(len(target_inds)*len(tf_inds)*args.threshold)

			# inds2keep = []
			# for i in range(X.shape[1]):
			# 	thresh = np.partition(abs(X[:,i]),-N)[-N]
			# 	inds2keep.extend(np.where(abs(X[:,i]) >= thresh)[0])
			# inds2keep = sorted(list(set(inds2keep)))
			# print('{} edges'.format(len(inds2keep)),'/',X.shape[0])

			# row_inds,col_inds = np.unravel_index(inds2keep,(len(target_inds),len(tf_inds)))
			# row_inds = np.array([target_inds[i] for i in row_inds])
			# col_inds = np.array([tf_inds[i] for i in col_inds])
			thresh_name = 'thresh{}'.format(args.threshold)

		elif args.num_regs != None:
			# N = args.num_regs
			# row_inds = []
			# col_inds = []
			
			# for i in range(len(target_inds)):
			# 	col_inds2keep = []
			# 	for cluster_no in range(1,args.K+1):
			# 		row_values = network_dict[trial_no][cluster_no][i]
			# 		thresh = np.partition(abs(row_values),-N)[-N]
			# 		col_inds2keep.extend(np.where(row_values > thresh)[0])
			# 	col_inds.extend(sorted(list(set(col_inds2keep))))
			# 	row_inds.extend([i]*len(set(col_inds2keep)))
			# inds2keep = np.ravel_multi_index(np.array([row_inds,col_inds]),\
			# 	(len(target_inds),len(tf_inds)))
			# print('{} edges'.format(len(inds2keep)),'/',X.shape[0])
			# col_inds = [tf_inds[i] for i in col_inds]
			# row_inds = [target_inds[i] for i in row_inds]

			thresh_name = 'nr{}'.format(args.num_regs)

		# save indices
		# file_name = 'gmm.nc{}.{}.{}.{}.trial{}.{}'.format(args.n_components,thresh_name,\
		# 	args.prior,args.V_noise,trial_no,args.file_name)
		# np.savetxt(os.path.join(args.results_dir,'indices.{}'.format(file_name)),\
		# 	np.array([row_inds,col_inds]).T,fmt='%i',delimiter='\t')

		if 'identity' in args.prior:
			covariance_prior = None
		elif 'knn' in args.prior:
			from sklearn.neighbors import NearestNeighbors

			n_neighbors = 5
			knn = NearestNeighbors(n_neighbors=n_neighbors)
			X_values = np.array([x.flatten() for x in X])
			knn.fit(X_values)
			knn_mat = knn.kneighbors_graph().toarray()
			mnn_mat = knn_mat * knn_mat.T # MNN

			if 'knn.prec' in args.prior:
				mnn_mat += np.eye(mnn_mat.shape[0])*n_neighbors
				# covariance_prior = mnn_mat/mnn_mat.max()

			elif 'knn.cov' in args.prior:
				mnn_mat += np.eye(mnn_mat.shape[0])*n_neighbors
				covariance_prior = mnn_mat/mnn_mat.max()
				X_std = X_values.std(1)
				covariance_prior = (covariance_prior*X_std).T*X_std

			X_values = None

		if 'weak' in args.prior:
			degrees_of_freedom_prior = args.K
		elif 'medium' in args.prior:
			degrees_of_freedom_prior = 0.5*np.size(X[0])
		elif 'strong' in args.prior:
			if 'strong10' in args.prior:
				degrees_of_freedom_prior = 10*np.size(X[0])
			else:
				degrees_of_freedom_prior = np.size(X[0])

		if 'normwish' in args.prior:
			mean_prior = None

		model = ShareNet(n_components=args.n_components,\
			covariance_prior=covariance_prior,\
			degrees_of_freedom_prior=degrees_of_freedom_prior,\
			mean_prior=mean_prior,\
			init_params='kmeans')

		if 'tabula_muris' in args.data_dir:
			max_it = 20
		else:
			max_it = 100

		print('max it:',max_it)

		use_block = args.K > 10

		model.fit(X,V,tol=args.tolerance,max_it=max_it,use_block=use_block)

		revised_networks = model.get_revised_edge_scores()
		for i,cluster_no in enumerate(range(1,args.K+1)):
			network = np.zeros((n_genes,n_genes))
			network[np.ix_(target_inds,tf_inds)] = revised_networks[i]
			prefix = 'gmm.nc{}.{}.{}.{}'.format(args.n_components,thresh_name,\
				args.prior,args.V_noise)
			file_name = 'trial{}.cluster{}.{}.{}'.format(trial_no,cluster_no,prefix,args.file_name)
			np.savetxt(os.path.join(args.results_dir,file_name),network,delimiter='\t')

		# save parameters
		file_name = 'gmm.nc{}.{}.{}.{}.trial{}.{}'.format(args.n_components,thresh_name,\
			args.prior,args.V_noise,trial_no,args.file_name)
		np.savetxt(os.path.join(args.results_dir,'m_tilde.{}'.format(file_name)),\
			model.m_tilde.reshape(model.m_tilde.shape[0],-1),delimiter='\t')
		np.savetxt(os.path.join(args.results_dir,'phi.{}'.format(file_name)),\
			model.phi,delimiter='\t')
		np.savetxt(os.path.join(args.results_dir,'covariances.{}'.format(file_name)),\
			model.covariances_.reshape(model.covariances_.shape[0],-1),delimiter='\t')
		np.savetxt(os.path.join(args.results_dir,'means.{}'.format(file_name)),\
			model.means_,delimiter='\t')
