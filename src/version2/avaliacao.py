"""
avaliacao.py — Version 2 Refatorada
--------------------------------------
Correções aplicadas:
  1. F1 macro e F1 weighted adicionados ao relatório de cada limiar.
  2. IC 95 % via bootstrap para AUC no conjunto de teste — permite afirmações
     estatísticas mais rigorosas no artigo ("AUC = X ± Y, IC 95% [a, b]").
  3. `avaliarModelo` simples mantido para compatibilidade, mas deprecado.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _bootstrap_auc(
    y_true: pd.Series,
    prob: np.ndarray,
    n_iter: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    IC 95 % para o AUC via bootstrap percentílico.

    Retorna (ic_baixo, ic_alto).
    """
    rng = np.random.default_rng(seed)
    aucs = []
    y_arr = np.asarray(y_true)
    for _ in range(n_iter):
        idx = rng.integers(0, len(y_arr), size=len(y_arr))
        if y_arr[idx].nunique() if hasattr(y_arr[idx], "nunique") else len(np.unique(y_arr[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_arr[idx], prob[idx]))
    aucs = np.array(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def avaliarModeloCompleto(
    nome: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    limiares: tuple = (0.5,),
    bootstrap_auc: bool = True,
) -> dict:
    """
    Avaliação completa no conjunto de teste.

    Métricas reportadas
    -------------------
    - Baseline (classe majoritária)
    - ROC-AUC com IC 95 % bootstrap (1 000 re-amostras)
    - Por limiar: Acurácia, Precisão, Recall, F1 binário,
                  F1 macro, F1 weighted, Matriz de Confusão

    Retorna
    -------
    dict com todas as métricas calculadas (útil para exportar para CSV/LaTeX).
    """
    prob     = model.predict_proba(X_test)[:, 1]
    baseline = max(y_test.mean(), 1 - y_test.mean())
    auc      = roc_auc_score(y_test, prob)

    print(f"\n{'=' * 60}")
    print(f"  {nome}")
    print(f"{'=' * 60}")
    print(f"  Baseline (sempre prever classe majoritária): {baseline:.4f}")
    print(f"  ROC-AUC: {auc:.4f}", end="")

    if bootstrap_auc:
        ic_b, ic_a = _bootstrap_auc(y_test, prob)
        print(f"  |  IC 95%: [{ic_b:.4f}, {ic_a:.4f}]")
    else:
        print()

    resultados: dict = {"modelo": nome, "auc": auc, "baseline": baseline}

    for limiar in limiares:
        pred = (prob >= limiar).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

        acc      = accuracy_score(y_test, pred)
        prec     = precision_score(y_test, pred, zero_division=0)
        rec      = recall_score(y_test, pred, zero_division=0)
        f1_bin   = f1_score(y_test, pred, zero_division=0)
        f1_mac   = f1_score(y_test, pred, average="macro",    zero_division=0)
        f1_wei   = f1_score(y_test, pred, average="weighted", zero_division=0)

        print(f"\n  Limiar {limiar}:")
        print(f"    Matriz de Confusão → TN={tn}  FP={fp}  FN={fn}  TP={tp}")
        print(f"    Acurácia  = {acc:.4f}")
        print(f"    Precisão  = {prec:.4f}  |  Recall = {rec:.4f}  |  F1 binário = {f1_bin:.4f}")
        print(f"    F1 macro  = {f1_mac:.4f}  |  F1 weighted = {f1_wei:.4f}")

        resultados[f"limiar_{limiar}"] = {
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "acuracia": acc, "precisao": prec, "recall": rec,
            "f1": f1_bin, "f1_macro": f1_mac, "f1_weighted": f1_wei,
        }

    return resultados


# Mantido para compatibilidade com código legado; use avaliarModeloCompleto.
def avaliarModelo(tipo: str, model, X_test, y_test) -> float:
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do {tipo}: {accuracy:.4f}")
    return accuracy