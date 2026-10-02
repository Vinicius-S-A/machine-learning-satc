from preprocessamento import carregarDados
from knn import treinarKNN, compararK
from regressao import (treinarRegressao, validarRegressao, compararC, analisarLimiares, mostrarCoeficientes)
from avaliacao import avaliarModeloCompleto

from previsao import preverNovoPaciente
from pacientes import PACIENTES

# ______________________
# CONFIGURAÇÃO:
tamanho = None              # None = dataset inteiro, ex.: 50000 = amostra (o KNN fica bem mais rápido)
avaliar = True
analisarRegressao = True
analisarKNN = True
prever = True              # o previsao.py ainda usa as colunas do dataset antigo (Sylhet)
limiares = (0.7, 0.5, 0.2)       # limiares usados na avaliação final
# ______________________

X_train, X_test, y_train, y_test = carregarDados("data/cdc_diabetes.csv", tamanho)
print(f"Treino: {len(X_train)} linhas | Teste: {len(X_test)} linhas | Colunas: {X_train.shape[1]}")

# Valores padrão; se a análise estiver ligada, são substituídos pelos escolhidos na validação (só no treino)
c = 0.1
k = 101 # 3

if analisarRegressao:
    validarRegressao(X_train, y_train, c)
    c = compararC(X_train, y_train, (0.1, 1, 10))
    analisarLimiares(X_train, y_train, c, (0.1, 0.2, 0.3, 0.5))

if analisarKNN:
    k = compararK(X_train, y_train, (3, 15, 51, 101))

logistic = treinarRegressao(X_train, y_train, c)
knn = treinarKNN(X_train, y_train, k)

if avaliar:
    avaliarModeloCompleto(f"KNN (k={k})", knn, X_test, y_test, limiares)
    avaliarModeloCompleto(f"Regressao Logistica (C={c})", logistic, X_test, y_test, limiares)

if analisarRegressao:
    mostrarCoeficientes(logistic)

if prever:
    limiar_clinico = 0.1
    for nome, paciente in PACIENTES.items():
        print(f"\n########## Paciente: {nome} ##########")
        print(f"--- Modelo: KNN (k={k}) ---")
        preverNovoPaciente(knn, paciente, limiar_clinico)
        print(f"--- Modelo: Regressao Logistica (C={c}) ---")
        preverNovoPaciente(logistic, paciente, limiar_clinico)