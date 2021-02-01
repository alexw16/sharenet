import numpy as np
import time

class ShareNet(object):
	def __init__(self,n_components,covariance_prior=None,mean_prior=None,degrees_of_freedom_prior=None,\
				 init_params='kmeans',random_state=1,beta_0=1):
		np.random.seed(random_state)
		
		self.K = n_components
		self.init_params = init_params
		self.beta_0 = beta_0

		if covariance_prior is not None:
			self.covariance_prior = covariance_prior
			self.use_covariance_prior = True
		else:
			self.use_covariance_prior = False
			self.covariance_prior = None

		if degrees_of_freedom_prior is not None:
			self.dof = degrees_of_freedom_prior
		else:
			self.dof = None

		if mean_prior is not None:
			self.mu_0 = mean_prior
		else:
			self.mu_0 = None

	def update_m_tilde(self):
		first_term = self.phi.dot(np.array([self.means_[k].dot(self.precisions_[k]) \
								for k in range(self.K)]))
		first_term += self.XdotV
		self.m_tilde = (first_term[:,:,np.newaxis]*self.S_tilde).sum(1)
	
	def update_S_tilde(self):

		diag_indices = (np.repeat(np.arange(self.N),self.C),\
			np.tile(np.arange(self.C),self.N),np.tile(np.arange(self.C),self.N))
		S_tilde_inv = np.einsum('ij,jkl->ikl',self.phi,self.precisions_)
		S_tilde_inv[diag_indices] += self.V.flatten()
		self.S_tilde = np.linalg.inv(S_tilde_inv)
		
	def update_phi(self):
		phi_unnormalized = np.zeros(self.phi.shape)
		
		trace = np.einsum('lij,kji->kl', self.precisions_, self.S_tilde)

		for k in range(self.K):
			diff = self.m_tilde-self.means_[k]
			quad = (diff.dot(self.precisions_[k])*diff).sum(1)
			s,logdet = np.linalg.slogdet(self.covariances_[k])
			logdet *= s
			ll = -0.5*quad - 0.5*logdet - 0.5*self.C*np.log(2*np.pi)
			phi_unnormalized[:,k] = ll - 0.5*trace[:,k]
		self.phi = np.exp(phi_unnormalized)
		self.phi = (self.phi.T/self.phi.sum(1)).T
		self.phi[np.isnan(self.phi)] = 1./self.K

		self.N_k = self.phi.sum(0)
	
	def update_precisions(self):

		self.dof_tilde = self.dof + self.phi.sum(0)

		for k in range(self.K):
			diff = self.m_tilde-self.means_[k]
			scatter_matrix = (self.phi[:,k]*diff.T).dot(diff) \
				+ (self.phi[:,k]*self.S_tilde.T).T.sum(0)

			if self.use_covariance_prior:
				mean_diff = self.mu_0-(self.phi[:,k]*self.m_tilde.T).T.sum(0)/self.N_k[k]
				mean_prior_term = mean_diff.T.dot(mean_diff)
				mean_prior_term *= self.beta_0*self.N_k[k]/(self.beta_0+self.N_k[k])
				B_tilde_inv = self.Psi_inv + scatter_matrix + mean_prior_term
				self.B_tilde[k] = np.linalg.inv(B_tilde_inv)
				self.covariances_[k] = B_tilde_inv/self.dof_tilde[k]
			else:
				self.covariances_[k] *= 1/self.phi[:,k].sum()
		self.precisions_ = np.linalg.inv(self.covariances_)
	
	def update_means(self):
		for k in range(self.K):
			weighted_sum = (self.phi[:,k]*self.m_tilde.T).T.sum(0)
			self.means_[k,:] = (weighted_sum + self.beta_0*self.mu_0)/(self.beta_0 + self.N_k[k])
		
	def initialize_parameters(self,X,V=None):
		self.X = np.array(X).T
		self.C = X.shape[1]
		self.N = X.shape[0]
		
		if V is not None:
			self.V = np.array(V).T
			self.XdotV = self.X*self.V
		else:
			self.V = np.eye(self.C)/X.std()
			self.XdotV = self.X*self.V
			self.V = np.array([self.V for i in range(self.X.shape[0])])

		if self.mu_0 is None:
			self.mu_0 = np.zeros(self.C)

		if self.dof is None:
			self.dof = self.C

		if self.covariance_prior is None:
			self.Psi_inv = np.eye(self.C)*self.dof
		else:
			self.Psi_inv = self.covariance_prior*self.dof
		
		self.m_tilde = X.copy()
		self.S_tilde = self.V.copy()	

		self.X = None # remove X to reduce memory footprint

	def initialize_mixture_parameters(self,X,V=None):

		if self.init_params == 'kmeans':
			from sklearn.cluster import KMeans
			km = KMeans(n_clusters=self.K,max_iter=20).fit(X)
			self.means_ = km.cluster_centers_
			self.phi = np.zeros((X.shape[0],self.K))
			self.phi[np.arange(self.phi.shape[0]), km.labels_] = 1
		else:
			self.means_ = np.random.random((self.K,self.C))
			self.phi = np.ones((X.shape[0],self.K))/self.K	

		self.precisions_ = np.array([np.eye(self.C) for k in range(self.K)])
		self.covariances_ = np.linalg.inv(self.precisions_)

		self.dof_tilde = self.dof + self.phi.sum(0)
		self.B_tilde = (self.precisions_.T/self.dof_tilde).T
		self.precisions_ = (self.B_tilde.T*self.dof_tilde).T
		self.covariances_ = np.linalg.inv(self.precisions_)

	def fit(self,X,V=None,max_it=100,tol=0.01,verbose=True):

		start = time.time()
		self.initialize_parameters(X,V)
		self.initialize_mixture_parameters(X,V)

		for it in range(max_it):
			old_m_tilde = self.m_tilde.copy()
			self.update_S_tilde()
			self.update_m_tilde()
			self.update_phi()
			self.update_means()
			self.update_precisions()
			
			relative_change = abs(self.m_tilde-old_m_tilde).max()/abs(old_m_tilde+1e-10).mean()
			if relative_change < tol and it > 4:
				break
			else:
				if verbose:
					print(it,relative_change)
		end = time.time()
		print('Time: {} seconds'.format(np.round(end-start,3)))

	def get_revised_edge_scores(self):
		
		return self.m_tilde

	def predict(self,X,V=None,max_it=100,tol=0.01,verbose=True):
		self.initialize_parameters(X,V)
		for it in range(max_it):
			old_m_tilde = self.m_tilde.copy()
			self.update_S_tilde()
			self.update_m_tilde()
			self.update_phi()
			relative_change = abs(self.m_tilde-old_m_tilde).max()/abs(old_m_tilde+1e-10).mean()
			if relative_change < tol:
				break
			else:
				if verbose:
					print(it,relative_change)

		return self.m_tilde,self.S_tilde,self.phi


