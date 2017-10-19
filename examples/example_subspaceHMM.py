
# Import
import numpy as np
import sys
from sklearn.decomposition import PCA
sys.path.append('..')
import hmmlearn.hmm as hmm

SEED = 103125

# Load the data
TRAINING_DATA_FILE = 'facescrambling_eeg_erp_sub1_cond1.csv'
Xtrain = np.genfromtxt(TRAINING_DATA_FILE, delimiter=',')
print(Xtrain.shape)


# Do PCA
N_COMP = 5
pca = PCA(n_components=N_COMP)
pca.fit(Xtrain)
Xtrain_sub = pca.transform(Xtrain)



# Run HMM on subspace
N_STATES_TRUE = 3
hmm_true_sub = hmm.GaussianHMM(n_components=N_STATES_TRUE, random_state=SEED)
hmm_true_sub.fit(Xtrain_sub)


# Simulate new data
T = 500
Xsub, _ = hmm_true_sub.sample(n_samples=T, random_state=SEED)

# Map to orignal space
X = pca.inverse_transform(Xsub)



# Run Subspace HMM
subhmm = hmm.SubspaceGaussianHMM(n_components=N_STATES_TRUE,
                                 forward_model=pca.components_,
                                 covariance_type='full')

subhmm.fit(X)

