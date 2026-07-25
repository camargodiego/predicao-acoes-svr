from typing import Dict, Any
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.ssa import SingularSpectrumAnalysis


def walk_forward_validation(
    close_prices: np.ndarray,
    log_returns: np.ndarray,
    split_ratio: float = 0.8,
    train_window: int = 500,
    window_ssa: int = 30,
    n_lags: int = 5
) -> Dict[str, np.ndarray]:
    """Executa a validação Walk-Forward causal garantindo ausência de Data Leakage."""
    split_idx = int(len(close_prices) * split_ratio)
    start_idx = max(split_idx, window_ssa + n_lags)

    pred_returns, real_returns = [], []
    pred_prices, real_prices = [], []

    for idx in range(start_idx, len(close_prices)):
        win_start = max(0, idx - train_window)
        past_returns = log_returns[win_start:idx]

        if len(past_returns) < window_ssa + n_lags:
            continue

        # Filtragem SSA causal aplicada somente na janela histórica
        ssa = SingularSpectrumAnalysis(window_size=window_ssa)
        filtered_returns = ssa.fit_transform(past_returns, keep_components=3)

        X_test = filtered_returns[-n_lags:].reshape(1, -1)

        X_train, y_train = [], []
        for j in range(n_lags, len(filtered_returns)):
            X_train.append(filtered_returns[j - n_lags : j])
            y_train.append(past_returns[j])  # Alvo = Retorno real não filtrado

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Padronização ajustada apenas nos dados de treino históricos
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        # Treinamento do modelo SVR
        model = SVR(kernel="rbf", C=10.0, gamma=0.1, epsilon=0.01)
        model.fit(X_train_scaled, y_train_scaled)

        pred_scaled = model.predict(X_test_scaled)
        pred_log_ret = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]

        # Reconstrução do preço por recomposição dos retornos
        price_prev = close_prices[idx - 1]
        price_pred = price_prev * np.exp(pred_log_ret)

        pred_returns.append(pred_log_ret)
        real_returns.append(log_returns[idx])
        pred_prices.append(price_pred)
        real_prices.append(close_prices[idx])

    return {
        "pred_returns": np.array(pred_returns),
        "real_returns": np.array(real_returns),
        "pred_prices": np.array(pred_prices),
        "real_prices": np.array(real_prices),
    }