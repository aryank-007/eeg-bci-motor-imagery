"""
Feature extraction for EEG motor imagery.
Implements Common Spatial Patterns (CSP) and log-band-power features.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.signal import welch


class LogBandPower(BaseEstimator, TransformerMixin):
    """
    Compute log-band-power per channel over the trial window.
    Simple but effective baseline feature for mu/beta oscillations.
    """

    def __init__(self, sfreq: float = 160.0, band=(8.0, 30.0)):
        self.sfreq = sfreq
        self.band = band

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X: (n_trials, n_channels, n_times)
        n_trials, n_channels, n_times = X.shape
        features = np.zeros((n_trials, n_channels))
        for i in range(n_trials):
            freqs, psd = welch(X[i], fs=self.sfreq, nperseg=min(256, n_times))
            band_mask = (freqs >= self.band[0]) & (freqs <= self.band[1])
            features[i] = np.log(psd[:, band_mask].mean(axis=1) + 1e-10)
        return features


class CommonSpatialPatterns(BaseEstimator, TransformerMixin):
    """
    CSP for two-class motor imagery.
    Finds spatial filters that maximize variance for one class
    while minimizing it for the other — directly capturing
    event-related (de)synchronization in mu/beta bands.
    """

    def __init__(self, n_components: int = 6):
        self.n_components = n_components
        self.filters_ = None

    def fit(self, X, y):
        classes = np.unique(y)
        assert len(classes) == 2, "CSP requires exactly 2 classes"
        X0 = X[y == classes[0]]
        X1 = X[y == classes[1]]

        cov0 = self._mean_cov(X0)
        cov1 = self._mean_cov(X1)

        # Solve generalized eigenvalue problem: cov0 * W = cov1 * W * D
        composite = cov0 + cov1
        eigenvalues, eigenvectors = np.linalg.eigh(composite)
        # Whitening matrix
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + 1e-8))
        W = D_inv_sqrt @ eigenvectors.T

        # Project and solve standard eigenvalue problem
        S0 = W @ cov0 @ W.T
        _, U = np.linalg.eigh(S0)

        # Combined spatial filters
        filters = (W.T @ U).T

        # Take n_components/2 from each end (most discriminative)
        n = self.n_components // 2
        idx = np.concatenate([np.arange(n), np.arange(-n, 0)])
        self.filters_ = filters[idx]
        return self

    def transform(self, X):
        # X: (n_trials, n_channels, n_times)
        projected = np.einsum("fc,nct->nft", self.filters_, X)
        # Log-variance of each spatial filter output
        var = np.var(projected, axis=2)
        log_var = np.log(var + 1e-10)
        # Normalize by total log-variance for robustness
        log_var /= np.sum(np.abs(log_var), axis=1, keepdims=True) + 1e-10
        return log_var

    @staticmethod
    def _mean_cov(X):
        """Normalized mean covariance matrix across trials."""
        n_trials = X.shape[0]
        covs = np.array(
            [x @ x.T / (np.trace(x @ x.T) + 1e-10) for x in X]
        )
        return covs.mean(axis=0)


def build_csp_lda_pipeline(n_components: int = 6) -> Pipeline:
    """Standard CSP + LDA pipeline used as baseline in BCI research."""
    return Pipeline([
        ("csp", CommonSpatialPatterns(n_components=n_components)),
        ("lda", LinearDiscriminantAnalysis(solver="svd")),
    ])


def build_logpower_lda_pipeline() -> Pipeline:
    """Log-band-power + LDA — simple but interpretable baseline."""
    return Pipeline([
        ("logpower", LogBandPower()),
        ("lda", LinearDiscriminantAnalysis(solver="svd")),
    ])
