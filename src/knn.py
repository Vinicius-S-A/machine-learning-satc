from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 
def criarModeloKNN(k=3):
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
 
def treinarKNN(X_train, y_train):
    model = criarModeloKNN()
    model.fit(X_train, y_train)
 
    return model

# from sklearn.neighbors import KNeighborsClassifier

# def treinarKNN(X_train, y_train):
#     model = KNeighborsClassifier(n_neighbors=3)
#     model.fit(X_train, y_train)
    
#     return model