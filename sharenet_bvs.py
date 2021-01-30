import numpy as np
import os
import time
import csv
import argparse
import multiprocessing as mp
from scipy.stats import norm

from utils import *

def sigmoid(x):

	x_clip = np.clip(x,-30,30)
	value = np.where(x_clip >= 0, 1 / (1 + np.exp(-x_clip)), \
		np.exp(x_clip) / (1 + np.exp(x_clip)))
	return value

class TargetGeneCAVI(object):
	def __init__(self,input_dict):

		self.target_ind = input_dict['target_ind']
		self.method = input_dict['method']
		self.y_dict = input_dict['y_dict']
		self.X_dict = input_dict['X_dict']
		self.XX_dict = input_dict['XX_dict']
		self.Xy_dict = input_dict['Xy_dict']

		self.tolerance = input_dict['tolerance']

		self.cluster_no_list = sorted(list(self.XX_dict.keys()))	

		self.num_regs = self.X_dict[self.cluster_no_list[0]].shape[1]	
		
		self.cluster_size_dict = {cluster_no: data.shape[0] \
			for cluster_no,data in self.y_dict.items()}

		# initialize parameters
		init_params_dict = input_dict['init_params'] if 'init_params' \
			in input_dict else None
		self.initialize_parameters(init_params_dict)

	def initialize_parameters(self,init_params_dict=None):

		self.pi = 0.05

		# use given initial parameters
		if init_params_dict:
			self.s2 = init_params_dict['s2']
			self.sigma2_beta = init_params_dict['sigma2_beta']
			self.sigma2_eps = init_params_dict['sigma2_eps']
			self.alpha = init_params_dict['alpha']
			self.mu = init_params_dict['mu']

			if 'mixture' in self.method:
				self.m_tilde = init_params_dict['m_tilde']
				self.S_tilde = init_params_dict['S_tilde']
				self.mvg_mean = init_params_dict['mvg_mean']
				self.mvg_precision = init_params_dict['mvg_precision']

		# initial default parameters
		else:
			self.s2 = {cluster_no: np.ones(self.num_regs) \
				for cluster_no in self.cluster_no_list}
			self.sigma2_beta = {cluster_no: 1 \
						for cluster_no in self.cluster_no_list}
			self.sigma2_eps = {cluster_no: 1 \
					for cluster_no in self.cluster_no_list}
			self.alpha = {cluster_no: np.ones(self.num_regs)*self.pi \
				for cluster_no in self.cluster_no_list}
			self.mu = {cluster_no: np.zeros(self.num_regs) \
				for cluster_no in self.cluster_no_list}

			if 'mixture' in self.method:
				self.m_tilde = {cluster_no: np.ones(self.num_regs)*-2 \
					for cluster_no in self.cluster_no_list}
				self.S_tilde = np.array([np.eye(len(self.cluster_no_list)) \
					for i in range(self.num_regs)])

		if 'mixture' in self.method:
			eta = 0.5
			K = 6
			self.w = np.array([norm.cdf(eta*(t+0.5),0,1)-norm.cdf(eta*(t-0.5),0,1) \
				for t in np.arange(-K,K+1)])

		# dictionary to track changes in alpha parameter
		self.param_change = {param_name: {cluster_no: np.ones(self.num_regs) \
			for cluster_no in self.cluster_no_list} for param_name in ['alpha']}

	def update_tilde(self):

		self.update_m_tilde()
		self.update_S_tilde()

		results_params_dict = {}
		results_params_dict['m_tilde'] = self.m_tilde
		results_params_dict['S_tilde'] = self.S_tilde

		return results_params_dict

	def check_convergence(self,max_it):
		relative_change_list = np.array([self.param_change['alpha'][cluster_no][i] \
			for i in range(self.num_regs) for cluster_no in self.cluster_no_list])

		if relative_change_list.max() < self.tolerance or self.it >= max_it:
			print('GENE {} CONVERGED ({} regulators): {} iterations'.format(\
				self.target_ind,self.num_regs,self.it))
			return True
		else:
			return False

	def create_results_dict(self):
		results_dict = {'alpha': self.alpha,'mu': self.mu, 's2': self.s2, \
			'sigma2_beta': self.sigma2_beta,\
			'sigma2_eps': self.sigma2_eps}

		if 'mixture' in self.method:
			results_dict['m_tilde'] = self.m_tilde
			results_dict['S_tilde'] = self.S_tilde

		return results_dict
				
	def run_cavi(self,max_it=200):

		global_start = time.time()

		self.it = 0
		converged = False

		# approximate expectation of log sigmoid term
		self.approx_exp_log_sigmoid()

		elbo_list = []
		while not converged:

			if self.it < 1:
				self.update_s2()

			self.update_alpha()
			self.update_mu()

			converged = self.check_convergence(max_it)
			self.it += 1

		results_params_dict = self.create_results_dict()

		return results_params_dict

	def update_s2(self):
		for j,cluster_no in enumerate(self.cluster_no_list):
			sigma2_eps = self.sigma2_eps[cluster_no]
			sigma2_beta = self.sigma2_beta[cluster_no]
			XX_ii = np.diag(self.XX_dict[cluster_no])

			denom = XX_ii + sigma2_eps/sigma2_beta

			self.s2[cluster_no] = sigma2_eps/denom

	def update_mu(self):

		r_vec = np.array([self.alpha[c] * self.mu[c] for c \
					in self.cluster_no_list]).T
		mu_params = np.array([self.mu[c] for c \
					in self.cluster_no_list]).T
		alpha_params = np.array([self.alpha[c] for c \
					in self.cluster_no_list]).T
		s2_params = np.array([self.s2[c] for c \
					in self.cluster_no_list]).T
		sigma2_eps_params = np.array([self.sigma2_eps[c] for c \
					in self.cluster_no_list])
		XX = np.array([self.XX_dict[c] for c \
					in self.cluster_no_list])
		Xy = np.array([self.Xy_dict[c] for c \
					in self.cluster_no_list]).T
		s2_div_sigma2_eps = s2_params/sigma2_eps_params
		sigma2_beta = np.array([self.sigma2_beta[c] for c \
					in self.cluster_no_list])

		for i in range(self.num_regs):
			r_vec[i,:] = 0
			sum_term = (r_vec * XX[:,i,:].T).sum(0)
			mu_params[i,:] = s2_div_sigma2_eps[i,:] * \
				(Xy[i,:]-sum_term)

			r_vec[i,:] = mu_params[i,:]*alpha_params[i,:]

		for j,cluster_no in enumerate(self.cluster_no_list):
			self.mu[cluster_no] = mu_params[:,j]

	def update_alpha(self):

		s2 = np.array([self.s2[cluster_no] \
			for cluster_no in self.cluster_no_list]).T
		sigma2_beta = np.array([self.sigma2_beta[cluster_no] \
			for cluster_no in self.cluster_no_list])
		sigma2_eps = np.array([self.sigma2_eps[cluster_no] \
			for cluster_no in self.cluster_no_list])
		mu = np.array([self.mu[cluster_no] \
			for cluster_no in self.cluster_no_list]).T
		alpha = np.array([self.alpha[cluster_no] \
			for cluster_no in self.cluster_no_list]).T

		u = 0.5*(np.log(s2/sigma2_beta) + mu**2/s2)

		if 'mixture' not in self.method:
			u += np.log(self.pi/(1-self.pi))
		else:
			u += (self.exp_log_sigmoid1-self.exp_log_sigmoid2)

		for i,cluster_no in enumerate(self.cluster_no_list):

			old_param = self.alpha[cluster_no]
			new_param = 1/(1+np.exp(-u[:,i]))

			self.param_change['alpha'][cluster_no] = abs(new_param-old_param)
			self.alpha[cluster_no] = new_param

	def approx_exp_log_sigmoid(self):

		eta = 0.5
		K = 6
		w = np.array([norm.cdf(eta*(t+0.5),0,1)-norm.cdf(eta*(t-0.5),0,1) \
			for t in np.arange(-K,K+1)])

		m_tilde = np.array([self.m_tilde[cluster_no] \
			for cluster_no in self.cluster_no_list]).T

		S_tilde_diag = np.array([np.diagonal(self.S_tilde[i]) \
			for i in range(self.S_tilde.shape[0])])

		values_mat1 = np.array([np.log(sigmoid(m_tilde + t*eta*np.sqrt(S_tilde_diag))) \
			for t in np.arange(-K,K+1)])
		values_mat1 = np.moveaxis(values_mat1,0,1)
		self.exp_log_sigmoid1 = w.dot(values_mat1)

		values_mat2 = np.array([np.log(1-sigmoid(m_tilde + t*eta*np.sqrt(S_tilde_diag))) \
			for t in np.arange(-K,K+1)])
		values_mat2 = np.moveaxis(values_mat2,0,1)
		self.exp_log_sigmoid2 = w.dot(values_mat2)

	def update_sigma2_beta(self):

		for cluster_no in self.cluster_no_list:
			alpha = self.alpha[cluster_no]
			s2 = self.s2[cluster_no]
			mu = self.mu[cluster_no]

			sigma2_beta = (alpha*(s2 + mu**2)).sum()/alpha.sum()
			self.sigma2_beta[cluster_no] = np.clip(sigma2_beta,10**-5,10**5)

	def update_sigma2_epsilon(self):

		for cluster_no in self.cluster_no_list:
			N = self.cluster_size_dict[cluster_no]
			y = self.y_dict[cluster_no]
			X = self.X_dict[cluster_no]
			r = self.mu[cluster_no] * \
				self.alpha[cluster_no]
			XX = self.XX_dict[cluster_no]

			alpha = self.alpha[cluster_no].copy()
			mu = self.mu[cluster_no]
			s2 = self.s2[cluster_no]

			y_XB = y-X.dot(r)

			value = y_XB.dot(y_XB) + ((alpha*(s2+mu**2)-(alpha*mu)**2)*np.diag(XX)).sum()
			value *= 1/N

			self.sigma2_eps[cluster_no] = np.clip(value,10**-5,10**5)

	def grad_m_tilde(self,x,alpha,S_tilde_diag,eta=0.5,K=6):

		t_values = np.arange(-K,K+1)
		g = np.array([self.w[i]*(alpha-sigmoid(x + eta*t*np.sqrt(S_tilde_diag))) \
			for i,t in enumerate(t_values)]).sum(0)

		if 'mixture' in self.method:
			g += np.einsum('ij,ijk->ik',-(x - self.mvg_mean),self.mvg_precision)
		else:
			g += -(x - self.mvg_mean).dot(self.mvg_precision)

		return g

	def update_m_tilde(self,step_size=1,max_it=50,tol=0.01):

		start = time.time()

		alpha = np.array([self.alpha[cluster_no] for cluster_no \
			in self.cluster_no_list]).T
		m_tilde = np.array([self.m_tilde[cluster_no] for cluster_no \
				in self.cluster_no_list]).T
		S_tilde_diag = np.array([np.diagonal(self.S_tilde[i]) \
			for i in range(self.S_tilde.shape[0])])

		it = 0
		delta_max_list = [1]
		m_tilde_old = m_tilde.copy()

		# coordinate ascent
		while it < max_it and delta_max_list[-1] > tol:
			g = self.grad_m_tilde(m_tilde,alpha,S_tilde_diag)
			m_tilde_old = m_tilde.copy()
			m_tilde += step_size*g
			delta_max_list.append(abs(m_tilde_old-m_tilde).max())
			it += 1

		for j,cluster_no in enumerate(self.cluster_no_list):
			self.m_tilde[cluster_no] = m_tilde[:,j]

	def grad_S_tilde(self,x,alpha,m_tilde,eta=0.5,K=6):

		S_tilde_diag_sqrt = np.sqrt(np.array([np.diagonal(x[i]) \
			for i in range(x.shape[0])]))

		t_values = np.arange(-K,K+1)

		# set up diagonal matrix
		D = np.zeros(x.shape)
		d = np.array([self.w[i]*eta*t*(alpha-sigmoid(m_tilde \
			+ eta*t*S_tilde_diag_sqrt)) for i,t in \
			enumerate(t_values)]).sum(0)/(2*S_tilde_diag_sqrt)
		for i in range(x.shape[0]):
			D[i][np.diag_indices(x.shape[1])] = d[i]
		
		g = -0.5*self.mvg_precision + D

		if np.count_nonzero(x) == x.shape[0]*x.shape[1]:
			x_inv = np.zeros(x.shape)
			x_inv[x != 0] = 1./x[x != 0]
			g += 0.5*x_inv
		else:
			g += 0.5*np.linalg.inv(x)

		return g

	def update_S_tilde(self,step_size=1,max_it=50,tol=0.01):

		start = time.time()
		alpha = np.array([self.alpha[cluster_no] for cluster_no \
			in self.cluster_no_list]).T
		m_tilde = np.array([self.m_tilde[cluster_no] for cluster_no \
				in self.cluster_no_list]).T

		n = len(self.cluster_no_list)

		S_tilde = self.S_tilde.copy()
		S_tilde_old = S_tilde.copy()

		it = 0
		delta_max_list = [1]

		# coordinate ascent
		while it < max_it and delta_max_list[-1] > tol:
			S_tilde_old = S_tilde.copy()
			g = self.grad_S_tilde(S_tilde,alpha,m_tilde)
			S_tilde += step_size*g
			delta_max_list.append(abs(S_tilde_old-S_tilde).max())
			it += 1

		self.S_tilde = S_tilde.copy()

# updates alpha, mu, s2 for a specific target gene
def cavi_update(input_dict):
	cavi = TargetGeneCAVI(input_dict)
	return cavi.run_cavi()

# updates m_tilde, S_tilde for a specific target gene
def cavi_update_tilde(input_dict):
	cavi = TargetGeneCAVI(input_dict)
	return cavi.update_tilde()

class GeneNetworkModel(object):
	def __init__(self,cluster_data_dict,
				 method,results_dir,outfile,
				 test_cluster_data_dict=None,
				 regtarget_dict=None,
				 tolerance=0.01,log_transform=False,
				 z_transform=False,q_transform=False,\
				 num_components=10):

		self.method = method
		self.cluster_data_dict = cluster_data_dict
		self.test_cluster_data_dict = test_cluster_data_dict

		self.log_transform = log_transform
		self.z_transform = z_transform
		self.q_transform = q_transform

		self.results_dir = results_dir
		self.outfile = outfile

		self.cluster_no_list = sorted(list(cluster_data_dict.keys()))
		self.n_genes = self.cluster_data_dict[1].shape[1]

		# set regulator-target pairings
		self.set_regtarget_pairings(regtarget_dict)

		depthnorm = True if 'depthnorm' in self.method else False
		clip = True if 'clip' in self.method else False

		# data pre-processing on training set
		for cluster_no,data in self.cluster_data_dict.items():
			data = preprocess_data(data,self.log_transform,\
				self.z_transform,self.q_transform,depthnorm,clip)
			self.cluster_data_dict[cluster_no] = data

		# data transformations on test set
		if self.test_cluster_data_dict != None:
			for cluster_no,data in self.test_cluster_data_dict.items():
				data = preprocess_data(data,self.log_transform,self.z_transform,\
					self.q_transform,depthnorm,clip)
				self.test_cluster_data_dict[cluster_no] = data
		
		self.C = len(self.cluster_no_list)
		self.K = num_components #10 # number of mixture of components
		self.beta_0 = 1
		self.mu_0 = np.zeros(self.C)

		# set up dictionaries to store variational parameters
		self.params_dict = {}
		if 'cavi' in self.method:
			self.params_dict = {key: {} for key in ['alpha','mu','s2',\
				'sigma2_eps','sigma2_beta']}
			if 'mixture' in self.method:
				self.params_dict['m_tilde'] = {}
				self.params_dict['S_tilde'] = {}
				self.params_dict['mvg_mean'] = {}
				self.params_dict['mvg_precision'] = {}
				self.params_dict['phi'] = {}
		else:
			self.params_dict = {'beta': {}}

		self.tolerance = tolerance

		self.XX_dict = {cluster_no: data.T.dot(data) for cluster_no,data \
			in self.cluster_data_dict.items()}

	def set_regtarget_pairings(self,regtarget_dict):
		if regtarget_dict == None:
			self.regtarget_dict = {i: sorted(list(set(range(self.n_genes)) - {i})) for \
				i in range(self.n_genes)}
			self.target_inds_list = list(range(self.n_genes))
		else:
			self.regtarget_dict = regtarget_dict
			# remove tf-tf pairings
			self.regtarget_dict = {target_ind: sorted(list(set(tf_inds_list) - {target_ind})) for \
				target_ind,tf_inds_list in self.regtarget_dict.items()}
			self.target_inds_list = sorted(list(regtarget_dict.keys()))

	def prepare_input_dict(self,target_ind):

		input_dict = {}
		reg_inds = self.regtarget_dict[target_ind]
		y_dict = {cluster_no: data[:,target_ind] for cluster_no,data \
			in self.cluster_data_dict.items()}
		X_dict = {cluster_no: data[:,reg_inds] for cluster_no,data \
			in self.cluster_data_dict.items()}
		XX_dict = {cluster_no: self.XX_dict[cluster_no][reg_inds,:][:,reg_inds] \
			for cluster_no in self.cluster_no_list}
		Xy_dict = {cluster_no: self.XX_dict[cluster_no][reg_inds,target_ind] \
			for cluster_no in self.cluster_no_list}

		input_dict = {}
		input_dict['method'] = self.method
		input_dict['target_ind'] = target_ind
		input_dict['y_dict'] = y_dict
		input_dict['X_dict'] = X_dict
		input_dict['XX_dict'] = XX_dict.copy()
		input_dict['Xy_dict'] = Xy_dict.copy()
		input_dict['tolerance'] = self.tolerance

		return input_dict

	def prepare_init_params(self,target_ind):

		init_params_dict = {}
		for param in self.params_list:
			init_params_dict[param] = self.params_dict[param][target_ind]

		if 'mixture' in self.method:
			
			m_tilde = np.array([self.params_dict['m_tilde'][target_ind][c] \
				for c in self.cluster_no_list]).T
			phi = self.params_dict['phi'][target_ind]

			init_params_dict['mvg_precision'] = np.einsum('ij,jkl->ikl',phi,self.precisions_)
			init_params_dict['mvg_mean'] = phi.dot(self.means_)

		return init_params_dict

	def prepare_init_mixture_params(self):

		self.means_ = np.ones((self.K,self.C))*-2
		self.precisions_ = np.array([np.eye(self.C) \
			for i in range(self.K)])*0.001

		num_edges = sum([len(v) for k,v in self.regtarget_dict.items()])
		self.phi = np.ones((num_edges,self.K))/self.K
		self.N_k = self.phi.sum(0)

		self.dof = self.K
		self.dof_tilde = self.dof + self.phi.sum(0)
		self.B_tilde = (self.precisions_.T/self.dof_tilde).T
		self.precisions_ = (self.B_tilde.T*self.dof_tilde).T
		self.covariances_ = np.linalg.inv(self.precisions_)
		self.Psi_inv = np.eye(self.C)*self.dof

		for target_ind in self.target_inds_list:
			n_regs = len(self.regtarget_dict[target_ind])
			self.params_dict['phi'][target_ind] = np.ones((n_regs,self.K))/self.K

	def update_regression_parameters(self,n_processes,initialize):
		'''update alpha, mu, s2 parameters'''

		for i in range(0,len(self.target_inds_list),n_processes):
			start = i
			end = min(i+n_processes,len(self.target_inds_list))

			# parallelize 
			if n_processes > 1:
				with mp.Pool(processes=n_processes) as p:
					input_dict_list = []
					for j in range(start,end):
						target_ind = self.target_inds_list[j]
						input_dict = self.prepare_input_dict(target_ind)

						# set up initial parameters
						if not initialize:
							init_params_dict = None
						else:
							init_params_dict = self.prepare_init_params(target_ind)											
						input_dict['init_params'] = init_params_dict
						input_dict_list.append(input_dict)
					results = p.map(cavi_update,input_dict_list)

			# serial
			else:
				target_ind = self.target_inds_list[i]
				input_dict = self.prepare_input_dict(target_ind)
				# set up initial parameters
				if not initialize:
					init_params_dict = None
				else:
					init_params_dict = self.prepare_init_params(target_ind)		
				input_dict['init_params'] = init_params_dict
				results_dict = cavi_update(input_dict)
				results = [results_dict]								

			# update local parameters
			for j,target_ind in enumerate(self.target_inds_list[start:end]):
				results_params_dict = results[j]
				for param in self.params_list:
					self.params_dict[param][target_ind] = results_params_dict[param]

	def update_tilde_parameters(self,n_processes):
		'''update m_tilde, S_tilde parameters'''

		for i in range(0,len(self.target_inds_list),n_processes):
			start = i
			end = min(i+n_processes,len(self.target_inds_list))

			if n_processes > 1:
				with mp.Pool(processes=n_processes) as p:
					input_dict_list = []
					for j in range(start,end):
						target_ind = self.target_inds_list[j]
						input_dict = self.prepare_input_dict(target_ind)
						input_dict['init_params'] = self.prepare_init_params(target_ind)
						input_dict_list.append(input_dict)
					results = p.map(cavi_update_tilde,input_dict_list)

			for j,target_ind in enumerate(self.target_inds_list[start:end]):
				results_params_dict = results[j]
				for param in ['m_tilde','S_tilde']:
					self.params_dict[param][target_ind] = results_params_dict[param]

	def update_means(self,m_tilde):
		for k in range(self.K):
			weighted_sum = (self.phi[:,k]*m_tilde.T).T.sum(0)
			self.means_[k,:] = (weighted_sum + self.beta_0*self.mu_0)/(self.beta_0 + self.N_k[k])

	def update_precisions(self,m_tilde,S_tilde):

		self.dof_tilde = self.dof + self.phi.sum(0)

		for k in range(self.K):
			diff = m_tilde-self.means_[k]
			scatter_matrix = (self.phi[:,k]*diff.T).dot(diff) \
				+ (self.phi[:,k]*S_tilde.T).T.sum(0)

			mean_diff = self.mu_0-(self.phi[:,k]*m_tilde.T).T.sum(0)/self.N_k[k]
			mean_prior_term = mean_diff.T.dot(mean_diff)
			mean_prior_term *= self.beta_0*self.N_k[k]/(self.beta_0+self.N_k[k])
			B_tilde_inv = self.Psi_inv + scatter_matrix + mean_prior_term
			self.B_tilde[k] = np.linalg.inv(B_tilde_inv)
			self.covariances_[k] = B_tilde_inv/self.dof_tilde[k]

		self.precisions_ = np.linalg.inv(self.covariances_)
	
	def update_phi(self,m_tilde,S_tilde):
		phi_unnormalized = np.zeros(self.phi.shape)
		
		trace = np.einsum('lij,kji->kl', self.precisions_, S_tilde)

		for k in range(self.K):
			diff = m_tilde-self.means_[k]
			quad = (diff.dot(self.precisions_[k])*diff).sum(1)
			s,logdet = np.linalg.slogdet(self.covariances_[k])
			logdet *= s
			ll = -0.5*quad - 0.5*logdet - 0.5*self.C*np.log(2*np.pi)
			phi_unnormalized[:,k] = ll - 0.5*trace[:,k]
		self.phi = np.exp(phi_unnormalized)
		self.phi = (self.phi.T/self.phi.sum(1)).T
		if np.isnan(self.phi).sum():
			self.phi[np.isnan(self.phi)] = 1./self.K

		self.N_k = self.phi.sum(0)

	def update_mixture(self):

		m_tilde_list = []
		for target_ind in self.target_inds_list:
			m_tilde = np.array([self.params_dict['m_tilde'][target_ind][c] \
				for c in self.cluster_no_list]).T
			m_tilde_list.append(m_tilde)
		m_tilde = np.concatenate(m_tilde_list)

		S_tilde_mat_list = []
		for target_ind in self.target_inds_list:
			S_tilde_mat_list.append(self.params_dict['S_tilde'][target_ind])
		S_tilde = np.concatenate(S_tilde_mat_list)

		self.update_means(m_tilde)
		self.update_precisions(m_tilde,S_tilde)
		self.update_phi(m_tilde,S_tilde)

		# update phi per target ind
		ind = 0
		for target_ind in self.target_inds_list:
			n_regs = len(self.regtarget_dict[target_ind])
			self.params_dict['phi'][target_ind] = self.phi[ind:ind+n_regs]
			ind += n_regs

	def update_hyperprior_parameters(self,n_processes):

		inner_it = 0
		max_change = 1
		m_tilde_old = 0
		self.max_change_list = []
		while inner_it < 100 and max_change > 0.01:

			# update m_tilde, S_tilde
			self.update_tilde_parameters(n_processes)

			# update mixture parameters
			if 'mixture' in self.method:
				self.update_mixture()

			# relative change of m_tilde
			m_tilde_list = []
			for target_ind in self.target_inds_list:
				m_tilde = np.array([self.params_dict['m_tilde'][target_ind][cluster_no] \
					for cluster_no in self.cluster_no_list]).T
				m_tilde_list.append(m_tilde)
			m_tilde = np.concatenate(m_tilde_list)
			max_change = abs(m_tilde - m_tilde_old).max()
			m_tilde_old = m_tilde.copy()

			inner_it += 1

			print('Inner it {}'.format(inner_it),max_change,'-----')

			self.max_change_list.append(max_change)

		print('TOTAL INNER IT: {}'.format(inner_it))

	def estimate_network(self,approach,n_processes=16):

		self.params_list = ['alpha','mu','s2','sigma2_beta','sigma2_eps']
		if 'mixture' in self.method:
			self.params_list.extend(['m_tilde','S_tilde'])

		if approach == 'cavi':

			# initialize mvg parameters
			if 'mixture' in self.method:
				self.prepare_init_mixture_params()

			self.update_regression_parameters(n_processes,initialize=False)
			for j,target_ind in enumerate(self.target_inds_list):
				self.write_params(target_ind,first_pass=True)

			if 'mixture' in self.method:
				self.update_hyperprior_parameters(n_processes)
				self.update_regression_parameters(n_processes,initialize=True)

			# write parameters
			for j,target_ind in enumerate(self.target_inds_list):
				self.write_params(target_ind,first_pass=False)

			if 'mixture' in self.method:
				outfile = 'mvg_mean.' + self.outfile

	def write_params(self,target_ind,first_pass=True):

		for param_name,param_dict in self.params_dict.items():
			if 'mvg_' not in param_name and param_name not in ['S_tilde','mixture_weights','phi']:
				for cluster_no in self.cluster_no_list:
					if first_pass:
						outfile = self.outfile.split('.txt')[0] + '.first_pass.txt'
						outfile = param_name + '.cluster{}.'.format(cluster_no) + outfile
					else:
						outfile = param_name + '.cluster{}.'.format(cluster_no) + self.outfile
					with open(os.path.join(self.results_dir,outfile),'a') as f:
						writer = csv.writer(f,delimiter='\t')						
						if param_name in ['b0','r']:
							writer.writerow([target_ind,param_dict[target_ind][cluster_no]])
						elif 'sigma' in param_name:
							continue
						else:
							for i,neigh_ind in enumerate(self.regtarget_dict[target_ind]):
								neigh_param = param_dict[target_ind][cluster_no][i]
								if neigh_param != 0:
									writer.writerow([target_ind,neigh_ind,neigh_param])

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data_dir', dest='data_dir')
	parser.add_argument('-r', '--results_dir', dest='results_dir')
	parser.add_argument('-K', '--K', dest='K',type=int)
	parser.add_argument('-m', '--method', dest='method')
	parser.add_argument('-tol','--tolerance',dest='tolerance',type=float,default=0.01)
	parser.add_argument('-rt','--regtarget',dest='regtarget',type=int,default=0)
	parser.add_argument('-np','--n_processes',dest='n_processes',type=int,default=8)
	parser.add_argument('-nc','--num_components',dest='num_components',type=int,default=10)

	args = parser.parse_args()

	log_transform = True if 'gaussian' in args.method else False
	z_transform = True if ('gaussian' in args.method and 'no_z' not in args.method) else False

	# load regulator-target gene pairings
	if args.regtarget != 0:
		print('USING TARGET-REGULATOR PAIRING')
		if 'gtrd' in args.method:
			regtarget_data = np.loadtxt(os.path.join(args.data_dir,'regtarget_gtrd.txt'),\
				delimiter='\t',dtype=str)			
		else:
			regtarget_data = np.loadtxt(os.path.join(args.data_dir,'regtarget.txt'),\
				delimiter='\t',dtype=str)	

		regtarget_dict = {int(target): list(map(int,tfs.split(';'))) for target,tfs \
			in regtarget_data if len(tfs) > 0}

		target_inds = list(regtarget_dict.keys())
		for tf_ind in regtarget_dict[target_inds[0]]:
			regtarget_dict.update({tf_ind: regtarget_dict[target_inds[0]]})

	else:
		regtarget_dict = None
		
	if not os.path.exists(args.results_dir):
		os.makedirs(args.results_dir)

	num_trials = 4 if 'testLL' in args.method else 2
	for trial_no in range(1,num_trials):
		outfile = 'trial{}.{}.tol{}.txt'.format(trial_no,args.method,\
			round(args.tolerance,5))

		# load data
		if 'testLL' not in args.method:
			cluster_data_dict = read_all_cluster_data(args.data_dir,list(range(1,args.K+1)))
			test_cluster_data_dict = None
		elif 'downsample' in args.method:
			num_cells_list = [int(args.method.split('downsample')[1].split('.')[0])] + [10000]*(args.K-1)
			cluster_data_dict,test_cluster_data_dict = load_data_train_test(args.data_dir,\
				list(range(1,args.K+1)),num_cells_list,trial_no)
			cluster_data_dict = {c: np.concatenate([data,test_cluster_data_dict[c]]) \
				for c,data in cluster_data_dict.items()}

		else:
			print('USING TEST LOG LIKELIHOOD')
			cluster_data_dict,_ = load_data_train_test(args.data_dir,\
				list(range(1,args.K+1)),[10000]*args.K,trial_no,results_dir=args.results_dir)

		n_genes = cluster_data_dict[1].shape[1]

		start = time.time()
		print('METHOD: {}'.format(args.method))
		if not os.path.exists(os.path.join(args.results_dir,'alpha.cluster1.' + outfile)):

			name = 'trial{}'.format(trial_no)
			model = GeneNetworkModel(cluster_data_dict,args.method,args.results_dir,outfile,\
									 regtarget_dict=regtarget_dict,\
									 tolerance=args.tolerance,\
									 log_transform=log_transform,\
									 q_transform=q_transform,\
									 z_transform=z_transform,\
									 num_components=args.num_components)

			if 'cavi' in args.method:
				model.estimate_network(approach='cavi',n_processes=args.n_processes)

			end = time.time()
			print('Total Run Time: {}'.format(end-start))


