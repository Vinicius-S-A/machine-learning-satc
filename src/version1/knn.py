from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 
def criarModeloKNN(k):
    # O KNN usa distância, e a idade (16 a 90) dominaria as colunas 0/1 se ficasse sem escala.
    # Apenas idade precisa de padronização, o resto passa direto.
    preparo = ColumnTransformer(
        transformers=[("num", StandardScaler(), ["idade"])],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
 
    return Pipeline(steps=[
        ("prep", preparo),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])
 
def treinarKNN(X_train, y_train, k):
    model = criarModeloKNN(k)
    model.fit(X_train, y_train)
 
    return model

def compararK(X_train, y_train, valores=(3, 15, 51, 101), tamanho_amostra=20000):
    # Com k=3 a probabilidade do KNN só assume 4 valores (0, 1/3, 2/3, 1), o que prejudica AUC e limiares.
    # Com muitos dados, k maior costuma funcionar melhor.
    # A validação cruzada do KNN é cara, então usa uma AMOSTRA do treino (o teste não é tocado).
    # Atenção: o melhor k tende a crescer com a quantidade de dados, então o k achado na amostra
    # pode ser um pouco menor que o ideal para o treino completo.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if len(X_train) > tamanho_amostra:
        X_amostra, _, y_amostra, _ = train_test_split(
            X_train, y_train, train_size=tamanho_amostra, random_state=42, stratify=y_train
        )
    else:
        X_amostra, y_amostra = X_train, y_train

    print(f"\n------------- Comparando k do KNN por validacao cruzada ({len(X_amostra)} linhas) -------------")
    resultados = {}
    for k in valores:
        auc = cross_val_score(criarModeloKNN(k), X_amostra, y_amostra, cv=cv, scoring="roc_auc")
        resultados[k] = auc.mean()
        print(f"k = {k:<4} | AUC media = {auc.mean():.4f} | desvio = {auc.std():.4f}")

    melhor = max(resultados, key=resultados.get)
    print(f"melhor k na validacao e {melhor}")

    return melhor

# from sklearn.neighbors import KNeighborsClassifier

# def treinarKNN(X_train, y_train):
#     model = KNeighborsClassifier(n_neighbors=3)
#     model.fit(X_train, y_train)
    
#     return model