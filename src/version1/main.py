from preprocessamento import carregarDados
from knn import treinarKNN
from regressao import treinarRegressao, validarRegressao, compararC, analisarLimiares
from avaliacao import avaliarModelo
from previsao import preverNovoPaciente
import pandas as pd

# ______________________
# CONFIGURAÇÃO:
avaliar = False
analisarRegressao = True
prever = True
# ______________________

X_train, X_test, y_train, y_test = carregarDados()

knn = treinarKNN(X_train, y_train)
logistic = treinarRegressao(X_train, y_train, 0.1)

if avaliar:
    avaliarModelo("KNN", knn, X_test, y_test)
    avaliarModelo("Regressão Logística", logistic, X_test, y_test)

if analisarRegressao:
    validarRegressao(X_train, y_train, 1)
    c = compararC(X_train, y_train, (0.1, 1, 10))
    analisarLimiares(X_train, y_train, 0.1, (0.3, 0.5, 0.7))

if prever:
    dados_paciente = pd.DataFrame([{
        "idade": 30,
        "genero": 1,
        "poliuria": 0,
        "polidipsia": 0,
        "perda_peso_subita": 0,
        "fraqueza": 1,
        "polifagia": 0,
        "candidiase_genital": 1,
        "visao_embacada": 1,
        "coceira": 0,
        "irritabilidade": 0,
        "cicatrizacao_atrasada": 1,
        "paresia_parcial": 0,
        "rigidez_muscular": 0,
        "alopecia": 0,
        "obesidade": 1
    }])

    limiar_clinico = 0.3 # 30%

    preverNovoPaciente(knn, limiar_clinico, dados_paciente)
    preverNovoPaciente(logistic, 0.5, dados_paciente)