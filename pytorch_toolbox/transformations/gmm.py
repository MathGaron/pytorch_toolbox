import numpy as np
from sklearn import mixture

class GMM(object):
    """
    Representation of a Gaussian mixture model probability distribution.
    """
    def __init__(self, n_components, covariance_type):
        """
        :param n_components: number of Gaussians to be used
        :param covariance_type: 'full', 'tied', 'diag', 'spherical'
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.covariances = []
        self.means = []

    def fit(self, samples):
        """
        Estimate model parameters.
        :param samples: samples to be used for the estimation, given as a 
        numpy array (row=sample, column=feature)
        """
        gmix = mixture.GaussianMixture(self.n_components, self.covariance_type)
        gmix.fit(samples)
        self.covariances = gmix.covariances_
        self.means = gmix.means_

    def draw(self, n_samples, gaussian_index=0):
        """
        Returns a vector of random samples drawn from the GMM, shape (row=sample, column=feature)
        :param n_samples: how many samples to draw
        :param gaussian_index: which gaussian from the mixture to draw from.
        """
        print(self.means.shape[1])
        if self.covariance_type == 'full': # Gaussians with full covariance.
            covariance_matrix = self.covariances[gaussian_index]
        elif self.covariance_type == 'diag': # Gaussians with diagonal covariance matrices.
            covariance_matrix = np.diag(self.covariances[gaussian_index])
        elif self.covariance_type == 'tied': # Gaussians with a tied covariance matrix; the same covariance matrix is shared by all the gaussians.
            covariance_matrix = self.covariances
        elif self.covariance_type == 'spherical': # Spherical Gaussians: variance is the same along all axes and zero across-axes.
            covariance_matrix = self.covariances[gaussian_index] * np.eye(self.means.shape[1])

        random_samples = np.random.multivariate_normal(self.means[gaussian_index],
                                                       covariance_matrix,
                                                       n_samples)

        return random_samples

def main():
    # as input we have 20 samples, half has the value of [1,2] and half [3,4]. Therefore, when fitting two gaussians we
    # would like to see one gaussian with mean [1,2] and the other [3,4].
    samples = np.repeat(np.array([[1, 2], [3, 4]]), 10, axis=0)
    gmm = GMM(2, 'full') #gmm with 2 gaussians and one covariance matrix for each gaussian (full)
    gmm.fit(samples)
    new_sample = gmm.draw(1)
    print(new_sample) # the result must be something close to [1,2] or [3,4]

if __name__ == "__main__": main()