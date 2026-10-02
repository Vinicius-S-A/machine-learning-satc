import pandas as pd

COLUNAS_ESPERADAS = [
    "pressao_alta",
    "colesterol_alto",
    # "checou_colesterol",
    "imc",
    # "fumante",
    "avc",
    "doenca_cardiaca_ou_infarto",
    "atividade_fisica",
    "consome_frutas",
    "consome_vegetais",
    # "alcool_excessivo",
    # "tem_plano_saude",
    # "sem_medico_por_custo",
    "saude_geral",
    # "dias_saude_mental_ruim",
    "dias_saude_fisica_ruim",
    "dificuldade_caminhar",
    "sexo",
    "faixa_etaria",
    # "escolaridade",
    # "renda",
]


def preverNovoPaciente(modelo_treinado, dados_paciente, limiar_clinico):
    if not isinstance(dados_paciente, dict):
        raise TypeError("dados_paciente deve ser um dicionário com as chaves esperadas.")

    chaves_entrada = set(dados_paciente.keys())
    chaves_esperadas = set(COLUNAS_ESPERADAS)

    if chaves_entrada != chaves_esperadas:
        faltantes = [coluna for coluna in COLUNAS_ESPERADAS if coluna not in dados_paciente]
        extras = [coluna for coluna in dados_paciente.keys() if coluna not in COLUNAS_ESPERADAS]
        mensagem = "dados_paciente deve conter exatamente as 21 chaves esperadas."
        if faltantes:
            mensagem += f" Faltantes: {faltantes}."
        if extras:
            mensagem += f" Extras: {extras}."
        raise ValueError(mensagem)

    df_paciente = pd.DataFrame([dados_paciente], columns=COLUNAS_ESPERADAS)

    if hasattr(modelo_treinado, "predict_proba"):
        probabilidades = modelo_treinado.predict_proba(df_paciente)[0]
    else:
        raise AttributeError("O modelo treinado não possui o método predict_proba().")

    if len(probabilidades) < 2:
        raise ValueError("O modelo deve produzir probabilidades para duas classes (0 e 1).")

    prob_risco = float(probabilidades[1])

    print("\n" + "=" * 60)
    print("SISTEMA DE TRIAGEM: AVALIAÇÃO DE NOVO PACIENTE")
    print("=" * 60)
    print(f"Probabilidade clínica calculada: {prob_risco:.2%}")

    if prob_risco >= limiar_clinico:
        print("ALERTA: Paciente classificado como POSITIVO para risco de Diabetes.")
        print(f"(Limiar de segurança de {limiar_clinico * 100:.0f}% ultrapassado)")
        decisao = "positivo"
    else:
        print("DIAGNÓSTICO: Paciente classificado como NEGATIVO para risco inicial.")
        decisao = "negativo"

    print("=" * 60 + "\n")

    return {
        "predicao": decisao,
        "prob_risco": prob_risco,
        "prob_risco_percentual": prob_risco * 100,
        "limiar_clinico": limiar_clinico,
    }