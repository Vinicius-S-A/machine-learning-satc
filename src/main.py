from preprocessamento import carregarDados
from knn import treinarKNN
from regressao import treinarRegressao, validarRegressao, compararC, analisarLimiares
from avaliacao import avaliarModelo

# ______________________
# CONFIGURAÇÃO:
avaliar = True
analisarRegressao = False
# ______________________

X_train, X_test, y_train, y_test = carregarDados()

knn = treinarKNN(X_train, y_train)
logistic = treinarRegressao(X_train, y_train, 10) # c = 10, pois na análise foi o melhor valor encontrado

if avaliar:
    avaliarModelo("KNN", knn, X_test, y_test)
    avaliarModelo("Regressão Logística", logistic, X_test, y_test)

if analisarRegressao:
    validarRegressao(X_train, y_train, 1)
    c = compararC(X_train, y_train, (0.1, 1, 10))
    analisarLimiares(X_train, y_train, c, (0.3, 0.5, 0.7))