# ShareNet
ShareNet is a Bayesian framework for boosting the accuracy of cell type-specific gene regulatory associations that can be used with a wide range of general network inference algorithms. It takes as input a set of initial network estimates for each cell type in a dataset, computed using a user's network inference algorithm of choice, and then outputs a corresponding set of revised networks.

## Usage
Here is an example workflow for using ShareNet with the outputs of a chosen network inference algorithm.
```python
# list of network edge scores 
# (each element in the list corresponds to the matrix of network edge scores for a unique cell 
# type - e.g., T x R matrix for a network considering T target genes, R regulator genes) 
# Note: networks must be of the same size
X = [list of numpy.ndarray]

# list of network edge score standard deviations
# (each element in the list corresponds to the standard deviations of 
# the network edge scores for a unique cell type)
V = [list of numpy.ndarray]

import sharenet

# n_components = number of mixture components to use 
model = sharenet.ShareNet(n_components = n_components)

# update model parameters using CAVI
model.fit(X,V)

# get revised network edge scores
model.get_revised_edge_scores()
```

Here is an example using the Bayesian variable selection generative model.
```python
# dictionary of cell type-specific gene expression profiles, where keys are the 
# names/numbers associated with a cell type and the values are arrays of the corresponding
# gene expression data (the columns in each array should correspond to the same set of 
# genes across the various cell types)
celltype_data_dict = {1: numpy.ndarray ,..., C: numpy.ndarray}

import sharenet_bvs as snb

model = snb.ShareNetGeneModel(celltype_data_dict,use_sharenet=True)

# update model parameters (n_processes specifies number of processes for parallelization)
model.fit(n_processes = n_processes)

# get model parameters
model.get_model_params(param_name, celltype_no, target_ind)
```

## Questions
For questions about the code, contact [alexwu@mit.edu](mailto:alexwu@mit.edu).
