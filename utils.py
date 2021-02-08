import numpy as np
import os
from scipy import sparse, io
import pandas as pd
import random

def posterior_predictiveLL(X,y,alpha,mu,s2,sigma2_eps,XX):

	N = X.shape[0]
	y_XB = y-X.dot(mu*alpha)
	LL = -N/2*np.log(2*np.pi*sigma2_eps) 
	LL -= 1./(2*sigma2_eps)*(y_XB.dot(y_XB))
	LL -= 1./(2*sigma2_eps)*((alpha*(s2+mu**2)-(alpha*mu)**2)*np.diag(XX)).sum()
	LL = LL/N

	return LL

def preprocess_data(data,log_transform,z_transform,\
				depthnorm,clip):

	if depthnorm:
		data = (data.T/data[:,-1]).T*10000

	if log_transform:
		data = np.log(1+data)

	data = data-data.mean(0)

	if z_transform:
		std = data.std(0)
		std[std==0] = 1
		data = data/std

	if clip:
		data = np.clip(data,-10,10)

	return data

def load_data_train_test(data_dir,cluster_no_list,num_cells_list,trial_no,\
										percent=80,delimiter='\t',results_dir=None,\
										return_train_inds=False):

	np.random.seed(trial_no)

	cluster_data_train_dict = {}
	cluster_data_test_dict = {}
	train_inds_dict = {}
	for i,cluster_no in enumerate(cluster_no_list):
		txt_file_path = os.path.join(data_dir,'{}.txt'.format(cluster_no))
		mtx_file_path = os.path.join(data_dir,'{}.mtx'.format(cluster_no))
		if os.path.exists(txt_file_path):
			data = np.loadtxt(txt_file_path,delimiter=delimiter)
		elif os.path.exists(mtx_file_path):
			data = io.mmread(mtx_file_path).toarray()

		if num_cells_list[i] < data.shape[0]:
			num_train_cells = int(percent/100*num_cells_list[i])
			num_test_cells = num_cells_list[i]-num_train_cells
			train_inds = np.random.choice(np.arange(data.shape[0]),num_train_cells)
			remaining_inds = sorted(list(set(range(data.shape[0]))-set(train_inds)))
			test_inds = np.random.choice(remaining_inds,num_test_cells)
		else: 
			all_inds = list(range(data.shape[0]))
			random.shuffle(all_inds)

			num_train_cells = int(percent/100*data.shape[0])

			train_inds = all_inds[0:num_train_cells]
			test_inds = all_inds[num_train_cells:]

		train_inds_dict[cluster_no] = train_inds

		cluster_data_train_dict[cluster_no] = data[train_inds]
		cluster_data_test_dict[cluster_no] = data[test_inds]

	return cluster_data_train_dict,cluster_data_test_dict

def bootstrap_sample(data_dir,cluster_no_list,trial_no,\
					percent=80,delimiter='\t'):

	np.random.seed(trial_no)

	cluster_data_dict = {}
	train_inds_dict = {}
	for i,cluster_no in enumerate(cluster_no_list):
		txt_file_path = os.path.join(data_dir,'{}.txt'.format(cluster_no))
		mtx_file_path = os.path.join(data_dir,'{}.mtx'.format(cluster_no))
		if os.path.exists(txt_file_path):
			data = np.loadtxt(txt_file_path,delimiter=delimiter)
		elif os.path.exists(mtx_file_path):
			data = io.mmread(mtx_file_path).toarray()

		np.random.seed(trial_no)
		inds = np.random.choice(range(data.shape[0]),size=data.shape[0],replace=True)

		cluster_data_dict[cluster_no] = data[inds]
		
	return cluster_data_dict

def load_network(results_dir,file_name,n_genes):
	
	pred_network = pd.read_csv(os.path.join(results_dir,file_name),\
		sep='\t',header=None).values
	if 'genie' in file_name:
		pred_network = pred_network.T
		if 'beeline' not in results_dir:
			inds_file_name = file_name.split('txt')[0] + 'nonzero_inds.txt'
			inds2keep = np.loadtxt(os.path.join(results_dir,inds_file_name),dtype=int)
			network = np.zeros((n_genes,n_genes))
			network[np.ix_(inds2keep,inds2keep)] = pred_network.copy()
			pred_network = network.copy()
	elif 'corr' in file_name:
		pred_network *= (1-np.eye(pred_network.shape[0]))

	return pred_network

