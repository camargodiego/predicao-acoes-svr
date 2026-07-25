"""
Script de execução da pipeline quantitativa de previsão e backtest SUZB3 (SSA-SVR).
"""
from src.data import fetch_and_process_returns
from src.model import walk_forward_validation
from src.metrics import (
    compute_regression_metrics,
    compute_directional_metrics,
    run_backtest
)


def run_pipeline():
    print("=" * 60)
    print("1. Coletando dados históricos da SUZB3.SA...")
    df, prices, returns = fetch_and_process_returns()

    print("2. Executando Validação Walk-Forward Causal (SSA + SVR)...")
    results = walk_forward_validation(prices, returns)

    print("\n" + "=" * 60)
    print("3. RESULTADOS EMPÍRICOS E MÉTRICAS ESTATÍSTICAS")
    print("=" * 60)

    # Métricas de Regressão
    reg_metrics = compute_regression_metrics(
        results["real_returns"], 
        results["pred_returns"]
    )
    print(f"R² Score (Log-Retornos) : {reg_metrics['r2']:.4f}")
    print(f"MAE (Log-Retornos)      : {reg_metrics['mae']:.4f}")
    print(f"RMSE (Log-Retornos)     : {reg_metrics['rmse']:.4f}")

    # Métricas Direcionais
    dir_metrics = compute_directional_metrics(
        results["real_returns"], 
        results["pred_returns"]
    )
    print(f"\nAcurácia Direcional (DA): {dir_metrics['acuracia_direcional']:.2%}")
    print(f"p-valor (Teste Binomial) : {dir_metrics['p_valor']:.4f}")
    print(f"Acertos / Total de Operações: {dir_metrics['n_sucessos']} / {dir_metrics['n_tentativas']}")

    # Simulação do Backtest
    bt_results = run_backtest(results["real_prices"], results["pred_returns"])
    print("\n" + "=" * 60)
    print("4. BACKTEST DA ESTRATÉGIA (COM CUSTO DE 0,05% POR TRADE)")
    print("=" * 60)
    print(f"Retorno Acumulado da Estratégia : {bt_results['retorno_acumulado_estrategia']:.2%}")
    print(f"Retorno Buy & Hold (Benchmark)  : {bt_results['retorno_acumulado_buy_hold']:.2%}")
    print(f"Índice de Sharpe Anualizado     : {bt_results['indice_sharpe']:.2f}")


if __name__ == "__main__":
    run_pipeline()