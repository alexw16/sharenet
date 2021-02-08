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
		self.use_sharenet = input_dict['use_sharenet']
		self.verbose = input_dict['verbose']

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

			if self.use_sharenet:
				self.m_tilde = init_params_dict['m_tilde']
				self.S_tilde = init_params_dict['S_tilde']
				self.weighted_mean = init_params_dict['weighted_mean']
				self.weighted_precision = init_params_dict['weighted_precision']

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

			if self.use_sharenet:
				self.m_tilde = {cluster_no: np.ones(self.num_regs)*-2 \
					for cluster_no in self.cluster_no_list}
				self.S_tilde = np.array([np.eye(len(self.cluster_no_list)) \
					for i in range(self.num_regs)])

		if self.use_sharenet:
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
			if self.verbose:
				print('GENE {} CONVERGED ({} regulators): {} iterations'.format(\
					self.target_ind,self.num_regs,self.it))
			return True
		else:
			return False

	def create_results_dict(self):
		results_dict = {'alpha': self.alpha,'mu': self.mu, 's2': self.s2, \
			'sigma2_beta': self.sigma2_beta,\
			'sigma2_eps': self.sigma2_eps}

		if self.use_sharenet:
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

		if self.use_sharenet:
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
			self.sigma2_beta[cluster_no] = sigma2_beta

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

			self.sigma2_eps[cluster_no] = value

	def grad_m_tilde(self,x,alpha,S_tilde_diag,eta=0.5,K=6):

		t_values = np.arange(-K,K+1)
		g = np.array([self.w[i]*(alpha-sigmoid(x + eta*t*np.sqrt(S_tilde_diag))) \
			for i,t in enumerate(t_values)]).sum(0)
		g += np.einsum('ij,ijk->ik',-(x - self.weighted_mean),self.weighted_precision)
		
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
		
		g = -0.5*self.weighted_precision + D

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

class ShareNetGeneModel(object):
	def __init__(self,cluster_data_dict,
				 use_sharenet,
				 regtarget_dict=None,
				 tolerance=0.01,
				 num_components=10,\
				 covariance_prior=None,\
				 mean_prior=None,\
				 degrees_of_freedom_prior=None,\
				 beta_0=1,
				 verbose=True):

		self.use_sharenet = use_sharenet
		self.verbose = verbose
		self.cluster_data_dict = cluster_data_dict

		self.cluster_no_list = sorted(list(cluster_data_dict.keys()))
		self.n_genes = self.cluster_data_dict[1].shape[1]

		# set regulator-target pairings
		self.set_regtarget_pairings(regtarget_dict)

		self.C = len(self.cluster_no_list)
		self.K = num_components

		# mixture hyperparameters
		self.beta_0 = beta_0
		if covariance_prior is not None:
			self.covariance_prior = covariance_prior
		else:
			self.use_covariance_prior = False

		if degrees_of_freedom_prior is not None:
			self.dof = degrees_of_freedom_prior
		else:
			self.dof = None

		if mean_prior is not None:
			self.mu_0 = mean_prior
		else:
			self.mu_0 = None

		# set up dictionaries to store variational parameters
		self.params_dict = {key: {} for key in ['alpha','mu','s2',\
			'sigma2_eps','sigma2_beta']}
		if self.use_sharenet:
			self.params_dict.update({key: {} for key in \
				['m_tilde','S_tilde','weighted_mean',\
				'weighted_precision','phi']})

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
			# remove auto-reg gene pairs
			self.regtarget_dict = {target_ind: sorted(list(set(tf_inds_list) \
				- {target_ind})) for target_ind,tf_inds_list \
				in self.regtarget_dict.items()}
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
		input_dict['use_sharenet'] = self.use_sharenet
		input_dict['target_ind'] = target_ind
		input_dict['y_dict'] = y_dict
		input_dict['X_dict'] = X_dict
		input_dict['XX_dict'] = XX_dict.copy()
		input_dict['Xy_dict'] = Xy_dict.copy()
		input_dict['tolerance'] = self.tolerance
		input_dict['verbose'] = self.verbose

		return input_dict

	def prepare_init_params(self,target_ind):

		init_params_dict = {}
		for param in self.params_list:
			init_params_dict[param] = self.params_dict[param][target_ind]

		if self.use_sharenet:
			m_tilde = np.array([self.params_dict['m_tilde'][target_ind][c] \
				for c in self.cluster_no_list]).T
			phi = self.params_dict['phi'][target_ind]

			init_params_dict['weighted_precision'] = np.einsum('ij,jkl->ikl',phi,self.precisions_)
			init_params_dict['weighted_mean'] = phi.dot(self.means_)

		return init_params_dict

	def prepare_init_mixture_params(self):

		if self.mu_0 is None:
			self.mu_0 = np.zeros(self.C)

		if self.dof is None:
			self.dof = self.C

		if self.covariance_prior is None:
			self.Psi_inv = np.eye(self.C)*self.dof
		else:
			self.Psi_inv = self.covariance_prior*self.dof

		self.means_ = np.ones((self.K,self.C))*-2
		self.precisions_ = np.array([np.eye(self.C) \
			for i in range(self.K)])

		num_edges = sum([len(v) for k,v in self.regtarget_dict.items()])
		self.phi = np.ones((num_edges,self.K))/self.K
		self.N_k = self.phi.sum(0)

		self.dof_tilde = self.dof + self.phi.sum(0)
		self.B_tilde = (self.precisions_.T/self.dof_tilde).T
		self.precisions_ = (self.B_tilde.T*self.dof_tilde).T
		self.covariances_ = np.linalg.inv(self.precisions_)

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
				target_ind = self.target_inds_list[j]
				input_dict = self.prepare_input_dict(target_ind)
				input_dict['init_params'] = self.prepare_init_params(target_ind)

				results_params_dict = cavi_update_tilde(input_dict)
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

		# update phi for each target ind
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

			if self.verbose:
				print(inner_it,max_change)

			self.max_change_list.append(max_change)

	def fit(self,n_processes=1):

		self.params_list = ['alpha','mu','s2','sigma2_beta','sigma2_eps']
		if self.use_sharenet:
			self.params_list.extend(['m_tilde','S_tilde'])

		# initialize mixture parameters
		if self.use_sharenet:
			self.prepare_init_mixture_params()

		# update regression parameters
		self.update_regression_parameters(n_processes,initialize=False)

		# update mixture prior parameters + regression parameters
		if self.use_sharenet:
			self.update_hyperprior_parameters(n_processes)
			self.update_regression_parameters(n_processes,initialize=True)

	def get_model_params(self,param_name,cluster_no,target_ind):

		return self.params_dict[param_name][target_ind][cluster_no]

	def write_params(self,param_name,results_dir,outfile):
		for cluster_no in self.cluster_no_list:
			with open(os.path.join(results_dir,outfile),'w') as f:
				writer = csv.writer(f,delimiter='\t')						
				for target_ind in self.target_inds_list:
					for i,neigh_ind in enumerate(self.regtarget_dict[target_ind]):
						neigh_param = param_dict[target_ind][cluster_no][i]
						if neigh_param != 0:
							writer.writerow([target_ind,neigh_ind,neigh_param])


