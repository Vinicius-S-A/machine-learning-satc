import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Divide mantendo as proporcoes das classes
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def criarModelo(c):
    preparo = ColumnTransformer(
        transformers=[("num", StandardScaler(), ["idade"])], #apenas idade precisa de padronizacao
        remainder="passthrough", # o resto passa
        verbose_feature_names_out=False # mantem os nomes originais
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
# VALIDAcaO
# ----------------------------------------------------------------------------------------------------

def validarRegressao(X_train, y_train, c):
    auc = cross_val_score(criarModelo(c), X_train, y_train, cv=cv, scoring="roc_auc")

    print("\n--- Validacao cruzada (5 rodadas, somente treino) ---")
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
    print(f"melhor c na validacao e {melhor} ")

    return melhor


def analisarLimiares(X_train, y_train, c, limiares=(0.3, 0.5, 0.7)):
    # cross_val_predict: cada passageiro do treino e previsto por um modelo que nao o viu no ajuste
    prob = cross_val_predict(criarModelo(c), X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    print("\n--- Efeito do limiar (previsoes de validacao no treino) ---")
    print(f"{'limiar':<8}{'TP':>5}{'TN':>5}{'FP':>5}{'FN':>5}{'Precisao':>11}{'Recall':>9}")
    for tau in limiares:
        pred = (prob >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, pred, labels=[0, 1]).ravel()
        print(f"{tau:<8}{tp:>5}{tn:>5}{fp:>5}{fn:>5}{precision_score(y_train, pred):>10.2%}{recall_score(y_train, pred):>9.2%}")

    # Em triagem de diabetes, um falso negativo (doente que passa despercebido) costuma
    # ser mais grave que um falso positivo, entao limiares menores podem fazer sentido


# ----------------------------------------------------------------------------------------------------
# INTERPRETAcaO
# ----------------------------------------------------------------------------------------------------

def mostrarCoeficientes(model):
    nomes = model.named_steps["prep"].get_feature_names_out()
    pesos = model.named_steps["logit"].coef_[0]
    coef = pd.Series(pesos, index=nomes).sort_values(ascending=False)

    print("\n--- Coeficientes (classe 1 = diabetes positivo) ---")
    print(coef.round(4))
    print("Intercepto:", model.named_steps["logit"].intercept_.round(4))
    # Positivo aumenta o escore de diabetes; negativo diminui. "idade" está padronizada
    # (1 unidade = 1 desvio-padrao). e associacao aprendida, nao prova causalidade.

    return coef