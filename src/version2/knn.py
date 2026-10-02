"""
knn.py — Version 2 Refatorada
-------------------------------
Correções aplicadas:
  1. `colunasNumericas` callable substituído por `obterColunasNumericas(X_train)`
     — lista estática passada ao ColumnTransformer.
  2. `criarModeloKNN` recebe `colunas_num` como argumento explícito.
  3. IC 95 % adicionado ao `compararK` para consistência com o relatório científico.
"""

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def criarModeloKNN(k: int, colunas_num: list[str]) -> Pipeline:
    """
    Cria o pipeline KNN.

    Parâmetros
    ----------
    k : int
        Número de vizinhos.
    colunas_num : list[str]
        Lista ESTÁTICA de colunas não-binárias a normalizar. Deve ser
        obtida com `obterColunasNumericas(X_train)` ANTES de chamar esta
        função.

    Nota sobre KNN e desbalanceamento
    ----------------------------------
    O KNN não tem parâmetro `class_weight` equivalente ao da Regressão
    Logística. A estratégia mais simples aqui é ajustar o limiar de decisão
    em `avaliarModeloCompleto` (limiares < 0.5), o que já está implementado.
    Para datasets muito desbalanceados, considere usar `weights="distance"`
    ou aplicar SMOTE antes do treino (apenas sobre X_train).
    """
    preparo = ColumnTransformer(
        transformers=[("num", StandardScaler(), colunas_num)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    return Pipeline(steps=[
        ("prep", preparo),
        ("knn", KNeighborsClassifier(n_neighbors=k, n_jobs=-1)),
    ])


def treinarKNN(X_train: pd.DataFrame, y_train: pd.Series, k: int = 3) -> Pipeline:
    from preprocessamento import obterColunasNumericas
    colunas_num = obterColunasNumericas(X_train)
    model = criarModeloKNN(k, colunas_num)
    model.fit(X_train, y_train)
    return model


def compararK(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    valores: tuple = (3, 15, 51, 101),
    tamanho_amostra: int = 20_000,
) -> int:
    """
    Seleciona o melhor k via validação cruzada sobre uma amostra do treino.

    O teste NUNCA é tocado. A amostra é estratificada para preservar a
    proporção de classes. O k escolhido na amostra pode ser ligeiramente
    menor que o ideal para o treino completo — documente essa limitação.
    """
    from preprocessamento import obterColunasNumericas

    if len(X_train) > tamanho_amostra:
        X_am, _, y_am, _ = train_test_split(
            X_train, y_train,
            train_size=tamanho_amostra,
            random_state=42,
            stratify=y_train,
        )
    else:
        X_am, y_am = X_train, y_train

    colunas_num = obterColunasNumericas(X_am)

    print(f"\n--- Comparando k do KNN ({len(X_am)} linhas, somente treino) ---")
    resultados = {}
    for k in valores:
        auc = cross_val_score(criarModeloKNN(k, colunas_num), X_am, y_am,
                              cv=cv, scoring="roc_auc")
        n = len(auc)
        media, dp = auc.mean(), auc.std(ddof=1)
        ic_baixo, ic_alto = st.t.interval(
            0.95, df=n - 1, loc=media, scale=dp / np.sqrt(n)
        )
        resultados[k] = media
        print(f"k = {k:<4} | AUC = {media:.4f} | DP = {dp:.4f} | "
              f"IC 95%: [{ic_baixo:.4f}, {ic_alto:.4f}]")

    melhor = max(resultados, key=resultados.get)
    print(f"Melhor k na validação: {melhor}")
    return melhor