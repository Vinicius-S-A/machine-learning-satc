import pandas as pd


def preverNovoPaciente(modelo_treinado, limiar_clinico=0.3):
    print("\n" + "=" * 50)
    print("SISTEMA DE TRIAGEM: AVALIAÇÃO DE NOVO PACIENTE")
    print("=" * 50)

    # 1. Criação do perfil do paciente
    dados_paciente = pd.DataFrame([{
        "idade": 48,
        "genero": 0,
        "poliuria": 1,
        "polidipsia": 1,
        "perda_peso_subita": 0,
        "fraqueza": 1,
        "polifagia": 0,
        "candidiase_genital": 0,
        "visao_embacada": 1,
        "coceira": 0,
        "irritabilidade": 0,
        "cicatrizacao_atrasada": 1,
        "paresia_parcial": 1,
        "rigidez_muscular": 0,
        "alopecia": 0,
        "obesidade": 1
    }])

    # 2. Cálculo da probabilidade com o modelo treinado
    probabilidades = modelo_treinado.predict_proba(dados_paciente)[0]
    prob_risco = probabilidades[1]

    print(f"Probabilidade clínica calculada: {prob_risco:.2%}")

    # 3. Regra de decisão com limiar personalizado
    if prob_risco >= limiar_clinico:
        print("ALERTA: Paciente classificado como POSITIVO para risco de Diabetes.")
        print(f"(Limiar de segurança de {limiar_clinico * 100:.0f}% ultrapassado)")
    else:
        print("DIAGNÓSTICO: Paciente classificado como NEGATIVO para risco inicial.")

    print("=" * 50 + "\n")