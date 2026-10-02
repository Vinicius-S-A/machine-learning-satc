"""
regressao.py — Version 2 Refatorada
-------------------------------------
Correções aplicadas:
  1. `colunasNumericas` substituída por `obterColunasNumericas(X_train)` — lista
     estática passada ao ColumnTransformer (elimina o callable dinâmico).
  2. `class_weight="balanced"` adicionado ao LogisticRegression para compensar
     o desequilíbrio de ~14 % de positivos no CDC.
  3. `criarModelo` agora recebe `colunas_num` como argumento explícito, tornando
     o pipeline completamente determinístico e testável de forma isolada.
  4. IC 95 % via t-Student adicionado ao `validarRegressao` para relatório
     científico mais rigoroso.
"""

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def criarModelo(c: float, colunas_num: list[str]) -> Pipeline:
    """
    Cria o pipeline de Regressão Logística.

    Parâmetros
    ----------
    c : float
        Inverso da regularização L2 (C = 1/λ).
    colunas_num : list[str]
        Lista ESTÁTICA de colunas não-binárias a normalizar. Deve ser
        obtida com `obterColunasNumericas(X_train)` ANTES de chamar esta
        função, nunca passada como callable.

    Nota sobre class_weight
    -----------------------
    Com ~14 % de positivos, o solver minimizaria uma função de custo que
    trata FN e FP como igualmente baratos, tendendo a classificar quase
    tudo como negativo. `class_weight="balanced"` repondera automaticamente
    cada classe inversamente à sua frequência:
        peso_neg = n / (2 * n_neg)
        peso_pos = n / (2 * n_pos)
    O efeito prático é que o modelo aprende a "se importar mais" com os
    casos positivos, aumentando o recall sem necessidade de SMOTE.
    """
    preparo = ColumnTransformer(
        transformers=[("num", StandardScaler(), colunas_num)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    return Pipeline(steps=[
        ("prep", preparo),
        ("logit", LogisticRegression(
            C=c,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",   # ← CORREÇÃO: trata o desequilíbrio de classes
            random_state=42,
        )),
    ])


def treinarRegressao(X_train: pd.DataFrame, y_train: pd.Series, c: float) -> Pipeline:
    from preprocessamento import obterColunasNumericas
    colunas_num = obterColunasNumericas(X_train)
    model = criarModelo(c, colunas_num)
    model.fit(X_train, y_train)
    return model


# ─── Validação (usa somente o treino — o teste nunca é tocado) ─────────────────

def validarRegressao(X_train: pd.DataFrame, y_train: pd.Series, c: float) -> np.ndarray:
    """Validação cruzada com IC 95 % via t-Student."""
    from preprocessamento import obterColunasNumericas
    colunas_num = obterColunasNumericas(X_train)

    auc = cross_val_score(criarModelo(c, colunas_num), X_train, y_train,
                          cv=cv, scoring="roc_auc")

    # IC 95 % via distribuição t (robusto mesmo com n=5 folds)
    n = len(auc)
    media, dp = auc.mean(), auc.std(ddof=1)
    ic_baixo, ic_alto = st.t.interval(0.95, df=n - 1, loc=media, scale=dp / np.sqrt(n))

    print("\n--- Validação cruzada — Regressão Logística (5 folds, somente treino) ---")
    print("ROC-AUC por fold:", auc.round(4))
    print(f"Média = {media:.4f} | DP = {dp:.4f} | IC 95%: [{ic_baixo:.4f}, {ic_alto:.4f}]")
    return auc


def compararC(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    valores: tuple = (0.01, 0.1, 1, 10),
) -> float:
    from preprocessamento import obterColunasNumericas
    colunas_num = obterColunasNumericas(X_train)

    print("\n--- Comparando C por validação cruzada (somente treino) ---")
    resultados = {}
    for c in valores:
        auc = cross_val_score(criarModelo(c, colunas_num), X_train, y_train,
                              cv=cv, scoring="roc_auc")
        resultados[c] = auc.mean()
        print(f"C = {c:<6} | AUC média = {auc.mean():.4f} | DP = {auc.std():.4f}")

    melhor = max(resultados, key=resultados.get)
    print(f"Melhor C na validação: {melhor}")
    return melhor


def analisarLimiares(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    c: float,
    limiares: tuple = (0.1, 0.2, 0.3, 0.5),
) -> None:
    """
    Analisa o impacto do limiar de decisão usando cross_val_predict no treino.
    Reporta TP, FP, FN, Precisão, Recall e F1 para cada limiar.
    """
    from preprocessamento import obterColunasNumericas
    colunas_num = obterColunasNumericas(X_train)

    prob = cross_val_predict(
        criarModelo(c, colunas_num), X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]

    print("\n--- Efeito do limiar (previsões de validação no treino) ---")
    print(f"{'Limiar':<8}{'TP':>8}{'FP':>8}{'FN':>8}{'Precisão':>11}{'Recall':>9}{'F1':>9}")
    for tau in limiares:
        pred = (prob >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, pred, labels=[0, 1]).ravel()
        prec = precision_score(y_train, pred, zero_division=0)
        rec  = recall_score(y_train, pred, zero_division=0)
        f1   = f1_score(y_train, pred, zero_division=0)
        print(f"{tau:<8}{tp:>8}{fp:>8}{fn:>8}{prec:>10.2%}{rec:>9.2%}{f1:>9.4f}")

    print("\n[NOTA] Com ~14% de positivos, limiares menores aumentam o recall "
          "(menos diabéticos não detectados) à custa da precisão. A escolha "
          "depende do custo relativo de cada tipo de erro — documente no artigo.")


# ─── Interpretação ─────────────────────────────────────────────────────────────

def mostrarCoeficientes(model: Pipeline) -> pd.Series:
    nomes = model.named_steps["prep"].get_feature_names_out()
    pesos = model.named_steps["logit"].coef_[0]
    coef  = pd.Series(pesos, index=nomes).sort_values(ascending=False)

    print("\n--- Coeficientes (classe 1 = diabetes ou pré-diabetes) ---")
    print(coef.round(4))
    print("Intercepto:", model.named_steps["logit"].intercept_.round(4))
    print("[NOTA] Colunas não-binárias estão padronizadas (1 unidade = 1 DP). "
          "É associação aprendida, não prova de causalidade.")
    return coef