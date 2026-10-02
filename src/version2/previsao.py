"""
previsao.py — Version 2 Refatorada
-------------------------------------
Mantido funcionalmente idêntico ao original (já estava bem implementado).
Pequenas melhorias:
  - Documentação alinhada com o restante do módulo refatorado.
  - Sem alterações de lógica.
"""

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


def preverNovoPaciente(
    modelo_treinado,
    dados_paciente: dict,
    limiar_clinico: float = 0.3,
) -> dict:
    """
    Prediz o risco de diabetes para um único paciente.

    Parâmetros
    ----------
    modelo_treinado : Pipeline scikit-learn
        Modelo com método predict_proba.
    dados_paciente : dict
        Dicionário com exatamente as 21 chaves de COLUNAS_ESPERADAS.
    limiar_clinico : float
        Limiar de decisão (padrão 0.30 — mais conservador que 0.5
        para minimizar falsos negativos em triagem clínica).

    Retorna
    -------
    dict com probabilidade, decisão e limiar usado.
    """
    if not isinstance(dados_paciente, dict):
        raise TypeError("dados_paciente deve ser um dicionário com as 21 chaves esperadas.")

    chaves_entrada   = set(dados_paciente.keys())
    chaves_esperadas = set(COLUNAS_ESPERADAS)

    if chaves_entrada != chaves_esperadas:
        faltantes = [c for c in COLUNAS_ESPERADAS if c not in dados_paciente]
        extras    = [c for c in dados_paciente   if c not in chaves_esperadas]
        msg = "dados_paciente deve conter exatamente as 21 chaves esperadas."
        if faltantes:
            msg += f" Faltantes: {faltantes}."
        if extras:
            msg += f" Extras: {extras}."
        raise ValueError(msg)

    df_paciente = pd.DataFrame([dados_paciente], columns=COLUNAS_ESPERADAS)

    if not hasattr(modelo_treinado, "predict_proba"):
        raise AttributeError("O modelo treinado não possui o método predict_proba().")

    probabilidades = modelo_treinado.predict_proba(df_paciente)[0]

    if len(probabilidades) < 2:
        raise ValueError("O modelo deve produzir probabilidades para duas classes (0 e 1).")

    prob_risco = float(probabilidades[1])
    decisao    = "positivo" if prob_risco >= limiar_clinico else "negativo"

    print("\n" + "=" * 60)
    print("SISTEMA DE TRIAGEM: AVALIAÇÃO DE NOVO PACIENTE")
    print("=" * 60)
    print(f"Probabilidade clínica calculada: {prob_risco:.2%}")

    if decisao == "positivo":
        print("ALERTA: Paciente classificado como POSITIVO para risco de Diabetes.")
        print(f"(Limiar de {limiar_clinico * 100:.0f}% ultrapassado)")
    else:
        print("DIAGNÓSTICO: Paciente classificado como NEGATIVO para risco inicial.")

    print("=" * 60 + "\n")

    return {
        "predicao": decisao,
        "prob_risco": prob_risco,
        "prob_risco_percentual": prob_risco * 100,
        "limiar_clinico": limiar_clinico,
    }