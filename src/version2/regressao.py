import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessamento import colunasNumericas

# Divide mantendo as proporções das classes
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def criarModelo(c):
    preparo = ColumnTransformer(
        transformers=[("num", StandardScaler(), colunasNumericas)],  # só as colunas não binárias são padronizadas
        remainder="passthrough",  # as binárias passam direto
        verbose_feature_names_out=False  # mantém os nomes originais
    )

    return Pipeline(steps=[
        ("prep", preparo),
        ("logit", LogisticRegression(C=c, solver="lbfgs", max_iter=1000))
    ])


def treinarRegressao(X_train, y_train, c):
    model = criarModelo(c)
    model.fit(X_train, y_train)

    return model


# ----------------------------------------------------------------------------------------------------
# VALIDAÇÃO (usa somente o treino, o teste não é tocado)
# ----------------------------------------------------------------------------------------------------

def validarRegressao(X_train, y_train, c):
    auc = cross_val_score(criarModelo(c), X_train, y_train, cv=cv, scoring="roc_auc")

    print("\n--- Validacao cruzada da Regressao Logistica (5 rodadas, somente treino) ---")
    print("ROC-AUC por rodada:", auc.round(4))
    print(f"Media = {auc.mean():.4f} | Desvio = {auc.std():.4f}")

    return auc


def compararC(X_train, y_train, valores=(0.1, 1, 10)):
    print("\n------------- Comparando C por validacao cruzada: -------------")
    resultados = {}
    for c in valores:
        auc = cross_val_score(criarModelo(c), X_train, y_train, cv=cv, scoring="roc_auc")
        resultados[c] = auc.mean()
        print(f"C = {c:<5} | AUC media = {auc.mean():.4f} | desvio = {auc.std():.4f}")

    melhor = max(resultados, key=resultados.get)
    print(f"melhor c na validacao e {melhor}")

    return melhor


def analisarLimiares(X_train, y_train, c, limiares=(0.1, 0.2, 0.3, 0.5)):
    # cross_val_predict: cada pessoa do treino é prevista por um modelo que não a viu no ajuste
    prob = cross_val_predict(criarModelo(c), X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    print("\n--- Efeito do limiar (previsoes de validacao no treino) ---")
    print(f"{'limiar':<8}{'TP':>8}{'FP':>8}{'FN':>8}{'Precisao':>11}{'Recall':>9}")
    for tau in limiares:
        pred = (prob >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, pred, labels=[0, 1]).ravel()
        prec = precision_score(y_train, pred, zero_division=0)
        rec = recall_score(y_train, pred, zero_division=0)
        print(f"{tau:<8}{tp:>8}{fp:>8}{fn:>8}{prec:>10.2%}{rec:>9.2%}")

# ----------------------------------------------------------------------------------------------------
# INTERPRETAÇÃO
# ----------------------------------------------------------------------------------------------------

def mostrarCoeficientes(model):
    nomes = model.named_steps["prep"].get_feature_names_out()
    pesos = model.named_steps["logit"].coef_[0]
    coef = pd.Series(pesos, index=nomes).sort_values(ascending=False)

    print("\n--- Coeficientes (classe 1 = diabetes ou pre-diabetes) ---")
    print(coef.round(4))
    print("Intercepto:", model.named_steps["logit"].intercept_.round(4))

    return coef