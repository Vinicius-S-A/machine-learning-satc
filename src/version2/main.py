"""
main.py — Version 2 Refatorada
---------------------------------
Orquestra o pipeline completo: carga → validação → treino → avaliação → persistência.

Correções aplicadas em relação ao original
------------------------------------------
1. Semente global já fixada em preprocessamento.py (importada aqui também para
   garantir a ordem de importação correta).
2. `colunasNumericas` callable eliminado — cada módulo usa `obterColunasNumericas`.
3. `class_weight="balanced"` na Regressão Logística (ver regressao.py).
4. Persistência dos modelos treinados com joblib (reprodutibilidade do artigo).
5. Métricas completas: F1 macro/weighted + IC 95 % bootstrap no AUC.
"""

import os
import joblib

# A importação de preprocessamento fixa as sementes global e numpy antes de
# qualquer outra importação que possa usar o estado aleatório.
from preprocessamento import carregarDados

from knn      import treinarKNN, compararK
from regressao import (
    treinarRegressao,
    validarRegressao,
    compararC,
    analisarLimiares,
    mostrarCoeficientes,
)
from avaliacao import avaliarModeloCompleto
from previsao  import preverNovoPaciente

<<<<<<< HEAD
# ─── Configuração ──────────────────────────────────────────────────────────────
TAMANHO           = None    # None = dataset completo | ex.: 50_000 = amostra
REMOVER_DUPLICATAS = False  # ver explicação em preprocessamento.py
AVALIAR           = True
ANALISAR_REGRESSAO = True
ANALISAR_KNN      = True
PREVER            = True
LIMIARES          = (0.5, 0.2)   # limiares avaliados no conjunto de teste
SALVAR_MODELOS    = True         # persiste os modelos em models/
# ───────────────────────────────────────────────────────────────────────────────

# ─── 1. Carga e divisão ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = carregarDados(
    tamanho=TAMANHO,
    remover_duplicatas=REMOVER_DUPLICATAS,
)
print(f"Treino: {len(X_train)} linhas | Teste: {len(X_test)} linhas | "
      f"Features: {X_train.shape[1]}")

# ─── 2. Seleção de hiperparâmetros (somente sobre treino) ──────────────────────
c_escolhido = 0.1
k_escolhido = 3

if ANALISAR_REGRESSAO:
    validarRegressao(X_train, y_train, c_escolhido)
    c_escolhido = compararC(X_train, y_train, (0.01, 0.1, 1, 10))
    analisarLimiares(X_train, y_train, c_escolhido, (0.1, 0.2, 0.3, 0.5))
=======
from previsao import preverNovoPaciente
from pacientes import PACIENTES
from selecao import compararSelecao

# ______________________
# CONFIGURAÇÃO:
tamanho = None              # None = dataset inteiro | ex.: 50000 = amostra (o KNN fica bem mais rápido)

analisarSelecao = False     # compara a AUC removendo grupos de colunas
avaliar = False
analisarRegressao = False
analisarKNN = False

prever = True              # testa os pacientes de exemplo do pacientes.py
limiar_clinico = 0.3        # limiar de decisão usado na previsão (ver analisarLimiares)
limiares = (0.5, 0.2)       # limiares usados na avaliação final
# ______________________

X_train, X_test, y_train, y_test = carregarDados(tamanho=tamanho)
print(f"Treino: {len(X_train)} linhas | Teste: {len(X_test)} linhas | Colunas: {X_train.shape[1]}")

# Valores padrão, MAS se a análise (analisarRegressao ou analisarKNN) estiver ligada, são substituídos pelos escolhidos na validação
c = 0.1
k = 51

if analisarSelecao:
    compararSelecao(X_train, y_train, c=c, k=max(k, 15), uma_por_vez=True)
>>>>>>> 7ed5cd0fd073c886638f73e3f66fa72d99557860

if ANALISAR_KNN:
    k_escolhido = compararK(X_train, y_train, (3, 15, 51, 101))

# ─── 3. Treino final (sobre todo X_train) ──────────────────────────────────────
logistic = treinarRegressao(X_train, y_train, c_escolhido)
knn      = treinarKNN(X_train, y_train, k_escolhido)

# ─── 4. Avaliação no conjunto de teste (única vez) ─────────────────────────────
if AVALIAR:
    avaliarModeloCompleto(
        f"Regressão Logística (C={c_escolhido}, class_weight=balanced)",
        logistic, X_test, y_test, LIMIARES,
    )
    avaliarModeloCompleto(
        f"KNN (k={k_escolhido})",
        knn, X_test, y_test, LIMIARES,
    )

if ANALISAR_REGRESSAO:
    mostrarCoeficientes(logistic)

<<<<<<< HEAD
# ─── 5. Persistência dos modelos (reprodutibilidade) ──────────────────────────
if SALVAR_MODELOS:
    os.makedirs("models", exist_ok=True)
    joblib.dump(logistic, f"models/logistic_C{c_escolhido}_balanced.pkl")
    joblib.dump(knn,      f"models/knn_k{k_escolhido}.pkl")
    print(f"\n[INFO] Modelos salvos em models/")
    print(f"  logistic_C{c_escolhido}_balanced.pkl")
    print(f"  knn_k{k_escolhido}.pkl")
    print("  Para carregar: model = joblib.load('models/<arquivo>.pkl')")

# ─── 6. Previsão de novo paciente ──────────────────────────────────────────────
if PREVER:
    paciente_exemplo = {
        "pressao_alta": 1,
        "colesterol_alto": 1,
        "checou_colesterol": 1,
        "imc": 34,
        "fumante": 1,
        "avc": 0,
        "doenca_cardiaca_ou_infarto": 1,
        "atividade_fisica": 0,
        "consome_frutas": 0,
        "consome_vegetais": 0,
        "alcool_excessivo": 1,
        "tem_plano_saude": 1,
        "sem_medico_por_custo": 0,
        "saude_geral": 4,
        "dias_saude_mental_ruim": 12,
        "dias_saude_fisica_ruim": 10,
        "dificuldade_caminhar": 1,
        "sexo": 1,
        "faixa_etaria": 12,
        "escolaridade": 2,
        "renda": 3,
    }

    preverNovoPaciente(logistic, paciente_exemplo, limiar_clinico=0.3)
    preverNovoPaciente(knn,      paciente_exemplo, limiar_clinico=0.3)
=======
if prever:
    for nome, paciente in PACIENTES.items():
        print(f"\n########## Paciente: {nome} ##########")
        print(f"--- Modelo: KNN (k={k}) ---")
        preverNovoPaciente(knn, paciente, limiar_clinico=limiar_clinico)
        print(f"--- Modelo: Regressao Logistica (C={c}) ---")
        preverNovoPaciente(logistic, paciente, limiar_clinico=limiar_clinico)
>>>>>>> 7ed5cd0fd073c886638f73e3f66fa72d99557860
