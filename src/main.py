from preprocessamento import carregarDados
from knn import treinarKNN
from regressao import treinarRegressao
from avaliacao import avaliarModelo

X_train, X_test, y_train, y_test = carregarDados()

knn = treinarKNN(X_train, y_train)
logistic = treinarRegressao(X_train, y_train)

avaliarModelo("KNN", knn, X_test, y_test)


# avaliarModelo("Regressão Logística", logistic, X_test, y_test)