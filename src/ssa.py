import numpy as np


class SingularSpectrumAnalysis:
    """Análise de Espectro Singular (SSA) para filtragem de ruído em séries temporais."""

    def __init__(self, window_size: int):
        self.L = window_size

    def fit_transform(self, series: np.ndarray, keep_components: int = 3) -> np.ndarray:
        N = len(series)
        K = N - self.L + 1
        if K <= 0:
            return series.copy()

        # Matriz de trajetória de Hankel
        X = np.column_stack([series[i : i + self.L] for i in range(K)])
        U, Sigma, VT = np.linalg.svd(X, full_matrices=False)

        keep = min(keep_components, len(Sigma))
        X_rec = np.zeros_like(X)
        for i in range(keep):
            X_rec += Sigma[i] * np.outer(U[:, i], VT[i, :])

        # Média diagonal para reconstrução da série
        series_rec = np.zeros(N)
        counts = np.zeros(N)
        for i in range(self.L):
            for j in range(K):
                series_rec[i + j] += X_rec[i, j]
                counts[i + j] += 1

        return series_rec / counts