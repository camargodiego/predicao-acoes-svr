from typing import Dict, Any
import numpy as np
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import binomtest


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula métricas estatísticas de regressão sobre os log-retornos."""
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred))
    }


def compute_directional_metrics(y_true_ret: np.ndarray, y_pred_ret: np.ndarray) -> Dict[str, Any]:
    """Avalia a acurácia direcional e executa o teste de hipótese binomial."""
    dir_true = np.where(y_true_ret >= 0, 1, 0)
    dir_pred = np.where(y_pred_ret >= 0, 1, 0)

    da = np.mean(dir_true == dir_pred)
    tn, fp, fn, tp = confusion_matrix(dir_true, dir_pred).ravel()

    n_success = int(np.sum(dir_true == dir_pred))
    n_trials = len(dir_true)
    p_val = binomtest(n_success, n_trials, p=0.5, alternative="greater").pvalue

    return {
        "acuracia_direcional": da,
        "matriz_confusao": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "p_valor": p_val,
        "n_sucessos": n_success,
        "n_tentativas": n_trials
    }


def run_backtest(
    prices: np.ndarray,
    pred_returns: np.ndarray,
    cost_per_trade: float = 0.0005
) -> Dict[str, Any]:
    """Executa o backtest da estratégia baseada em sinais com custo operacional."""
    sinais = np.where(pred_returns >= 0, 1, -1)
    retornos_estrategia = []
    posicao_anterior = 0

    for i in range(len(sinais)):
        ret_real = (prices[i] / prices[i - 1] - 1) if i > 0 else 0.0

        if i == 0:
            custo = cost_per_trade
        else:
            custo = (2 * cost_per_trade) if sinais[i] != posicao_anterior else 0.0

        ret_bruto = posicao_anterior * ret_real if i > 0 else 0.0
        retornos_estrategia.append(ret_bruto - custo)
        posicao_anterior = sinais[i]

    strat_ret = np.array(retornos_estrategia)
    cum_ret_strat = (1 + strat_ret).prod() - 1
    cum_ret_bh = (prices[-1] / prices[0]) - 1

    sharpe = (
        (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(252)
        if np.std(strat_ret) > 0
        else 0.0
    )

    return {
        "retornos_estrategia": strat_ret,
        "retorno_acumulado_estrategia": cum_ret_strat,
        "retorno_acumulado_buy_hold": cum_ret_bh,
        "indice_sharpe": sharpe
    }