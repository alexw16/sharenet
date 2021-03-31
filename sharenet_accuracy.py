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

def calculate_accuracy(prefix,write_dir,pred_network_dict,ref_network_dict,ref_network_name):

	from sklearn.metrics import average_precision_score,roc_auc_score

	data_dict = {'cluster_no': [], 'auprc': [],'auroc': [], 'epr': []}
	cluster_no_set = set(pred_network_dict.keys()) & set(ref_network_dict.keys())
	for cluster_no in sorted(cluster_no_set):

		y_pred = abs(pred_network_dict[cluster_no].flatten())
		y_true = ref_network_dict[cluster_no].flatten()

		auprc = average_precision_score(y_true,y_pred)
		auroc = roc_auc_score(y_true,y_pred)

		k = int(y_true.sum())
		thresh = np.partition(y_pred,-k)[-k]
		edge_density = k/len(y_true)

		early_precision = (((y_pred >= thresh) + y_true) == 2).sum()/k
		epr = early_precision/edge_density
		data_dict['cluster_no'].append(cluster_no)
		data_dict['auprc'].append(auprc)
		data_dict['auroc'].append(auroc)
		data_dict['epr'].append(epr)

	df = pd.DataFrame(data_dict)

	results_file_name = '{}.{}.csv'.format(ref_network_name,prefix)
	df.to_csv(os.path.join(write_dir,results_file_name))

	return df
	
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data_dir', dest='data_dir')
	parser.add_argument('-r', '--results_dir', dest='results_dir')
	parser.add_argument('-K', '--K', dest='K',type=int)
	parser.add_argument('-f', '--file_name', dest='file_name')
	parser.add_argument('-rn', '--ref_network', dest='ref_network')

	args = parser.parse_args()

	write_dir = os.path.join(args.data_dir,'accuracy')
	if not os.path.exists(write_dir):
		os.mkdir(write_dir)

	ref_dir = os.path.join(args.data_dir,'accuracy')
	genes = np.loadtxt(os.path.join(args.data_dir,'genes.txt'),dtype=str)

	regtarget_data = np.loadtxt(os.path.join(args.data_dir,'regtarget.txt'),
		delimiter='\t',dtype=str)
	regtarget_dict = {int(target): list(map(int,tfs.split(';'))) for target,tfs 
		in regtarget_data if len(tfs) > 0}

	target_inds = sorted(list(regtarget_dict.keys()))
	tf_inds = regtarget_dict[target_inds[0]]
	target_inds = sorted(tf_inds + target_inds)

	n_genes = len(genes)

	pred_network_dict = {}
	for cluster_no in range(1,args.K+1):
		f = 'cluster{}.{}'.format(cluster_no,args.file_name)
		file_path = os.path.join(args.results_dir,f)
		if file_path.endswith('.gz'):
			pred_network = pd.read_csv(file_path,delimiter='\t',\
				header=None,compression='gzip').values
		else:
			pred_network = pd.read_csv(file_path,delimiter='\t',\
				header=None).values

		# if 'gmm' not in args.file_name and 'sharenet' not in args.file_name:
		if 'pidc' in args.file_name:
			network = np.zeros((n_genes,n_genes))
			network[np.ix_(target_inds,tf_inds)] = pred_network.copy()
			pred_network = network.copy()
		elif 'genie' in args.file_name:
			pred_network = pred_network.T
			inds_file_name = 'cluster{}.{}.nonzero_inds.txt'.format(\
				cluster_no,args.file_name.split('.txt')[0])
			inds2keep = np.loadtxt(os.path.join(args.results_dir,inds_file_name),dtype=int)
			network = np.zeros((n_genes,n_genes))
			network[np.ix_(inds2keep,inds2keep)] = pred_network.copy()
			pred_network = network.copy()
		elif 'corr' in args.file_name:
			pred_network *= (1-np.eye(pred_network.shape[0]))

		pred_network_dict[cluster_no] = abs(pred_network)

	# load reference network
	if args.ref_network == 'STRING':
		ref_network = np.zeros((n_genes,n_genes))
		inds = np.loadtxt(os.path.join(args.data_dir,'reference','{}.txt'.format(args.ref_network)),delimiter='\t',dtype=int)
		ref_network[(inds[:,0],inds[:,1])] = 1
		ref_network += ref_network.T
		ref_network = ref_network.astype(bool).astype(int)
		ref_network_dict = {cluster_no: ref_network for cluster_no in range(1,args.K+1)}

		pred_network_dict = {cluster_no: network[target_inds][:,tf_inds] \
			for cluster_no,network in pred_network_dict.items()}
		ref_network_dict = {cluster_no: network[target_inds][:,tf_inds] \
			for cluster_no,network in ref_network_dict.items()}

	elif args.ref_network == 'nonspecific_chip':
		ref_network = np.zeros((n_genes,n_genes))
		inds = np.loadtxt(os.path.join(args.data_dir,'reference','{}.txt'.format(args.ref_network)),delimiter='\t',dtype=int)
		tf_inds2keep = sorted(list(set(inds[:,0])))
		ref_network[(inds[:,1],inds[:,0])] = 1

		ref_network = ref_network.astype(bool).astype(int)
		ref_network_dict = {cluster_no: ref_network for cluster_no in range(1,args.K+1)}

		pred_network_dict = {cluster_no: network[target_inds][:,tf_inds2keep] \
			for cluster_no,network in pred_network_dict.items()}
		ref_network_dict = {cluster_no: network[target_inds][:,tf_inds2keep] \
			for cluster_no,network in ref_network_dict.items()}

	elif 'specific_chip' in args.ref_network:
		ref_network_dict = {}
		for cluster_no in range(1,args.K+1):
			path = os.path.join(args.data_dir,'reference','{}.specific_chip.txt'.format(cluster_no))
			if os.path.exists(path):
				ref_network = pd.read_csv(os.path.join(path),sep='\t',header=None).values
				tf_inds2keep = np.where(ref_network.sum(0))[0]
				if ref_network.sum() >= 50: # include if at least 50 edges
					ref_network_dict[cluster_no] = ref_network[target_inds][:,tf_inds2keep]
					pred_network_dict[cluster_no] = pred_network_dict[cluster_no][target_inds][:,tf_inds2keep]

		
	prefix = '{}'.format(args.file_name.split('.txt')[0])
	calculate_accuracy(prefix,write_dir,pred_network_dict,ref_network_dict,args.ref_network)

