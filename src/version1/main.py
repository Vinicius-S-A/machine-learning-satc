from preprocessamento import carregarDados
from knn import compararK, treinarKNN
from regressao import treinarRegressao, validarRegressao, compararC, analisarLimiares
from avaliacao import avaliarModelo
from previsao import preverNovoPaciente
import pandas as pd

# ______________________
# CONFIGURAÇÃO:
avaliar = False
analisarRegressao = True
analisarKNN = False
prever = True
# ______________________

X_train, X_test, y_train, y_test = carregarDados()

knn = treinarKNN(X_train, y_train, 9)
logistic = treinarRegressao(X_train, y_train, 0.1)

if analisarKNN:
    k = compararK(X_train, y_train, (3, 15, 51, 101))

if avaliar:
    avaliarModelo("KNN", knn, X_test, y_test)
    avaliarModelo("Regressão Logística", logistic, X_test, y_test)

if analisarRegressao:
    validarRegressao(X_train, y_train, 1)
    c = compararC(X_train, y_train, (0.1, 1, 10))
    analisarLimiares(X_train, y_train, 0.1, (0.3, 0.5, 0.7))

if prever:
    dados_paciente = pd.DataFrame([{
        "idade": 20,
        "genero": 0, # 1 = masculino, 0 = feminino
        "poliuria": 1,
        "polidipsia": 1,
        "perda_peso_subita": 0,
        "fraqueza": 0,
        "polifagia": 0,
        "candidiase_genital": 0,
        "visao_embacada": 0,
        "coceira": 0,
        "irritabilidade": 0,
        "cicatrizacao_atrasada": 1,
        "paresia_parcial": 0,
        "rigidez_muscular": 0,
        "alopecia": 0,
        "obesidade": 1
    }])

    limiar_clinico = 0.5 # 10%

    preverNovoPaciente(knn, limiar_clinico, dados_paciente)
    preverNovoPaciente(logistic, 0.5, dados_paciente)