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
    "checou_colesterol",
    "imc",
    "fumante",
    "avc",
    "doenca_cardiaca_ou_infarto",
    "atividade_fisica",
    "consome_frutas",
    "consome_vegetais",
    "alcool_excessivo",
    "tem_plano_saude",
    "sem_medico_por_custo",
    "saude_geral",
    "dias_saude_mental_ruim",
    "dias_saude_fisica_ruim",
    "dificuldade_caminhar",
    "sexo",
    "faixa_etaria",
    "escolaridade",
    "renda",
]


def preverNovoPaciente(modelo_treinado, dados_paciente: dict, limiar_clinico=0.3):
    """
    Realiza a predição de risco de diabetes para um único paciente.

    Parâmetros:
        modelo_treinado: modelo treinado com método predict_proba.
        dados_paciente: dicionário contendo exatamente as 21 colunas esperadas.
        limiar_clinico: limiar de decisão para classe positiva (default: 0.30).

    Retorna:
        dict com a probabilidade de risco e a decisão final.
    """
    if not isinstance(dados_paciente, dict):
        raise TypeError("dados_paciente deve ser um dicionário com as 21 chaves esperadas.")

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
    print("=" * 50)

    # Cálculo da probabilidade com o modelo treinado
    probabilidades = modelo_treinado.predict_proba(dados_paciente)[0]
    prob_risco = probabilidades[1]

    print(f"Probabilidade clínica calculada: {prob_risco:.2%}")

    if prob_risco >= limiar_clinico:
        print("ALERTA: Paciente classificado como POSITIVO para risco de Diabetes.")
        print(f"(Limiar de segurança de {limiar_clinico * 100:.0f}% ultrapassado)")
        decisao = "positivo"
    else:
        print("DIAGNÓSTICO: Paciente classificado como NEGATIVO para risco inicial.")

    print("=" * 60 + "\n")

    return {
        "predicao": decisao,
        "prob_risco": prob_risco,
        "prob_risco_percentual": prob_risco * 100,
        "limiar_clinico": limiar_clinico,
    }


if __name__ == "__main__":
    # Exemplo de uso em produção:
    # class ModeloMock:
    #     def predict_proba(self, df):
    #         return [[0.15, 0.85]]
    #
    # modelo_mock = ModeloMock()
    # paciente_alto_risco = {
    #     "pressao_alta": 1,
    #     "colesterol_alto": 1,
    #     "checou_colesterol": 1,
    #     "imc": 34,
    #     "fumante": 1,
    #     "avc": 0,
    #     "doenca_cardiaca_ou_infarto": 1,
    #     "atividade_fisica": 0,
    #     "consome_frutas": 0,
    #     "consome_vegetais": 0,
    #     "alcool_excessivo": 1,
    #     "tem_plano_saude": 1,
    #     "sem_medico_por_custo": 0,
    #     "saude_geral": 4,
    #     "dias_saude_mental_ruim": 12,
    #     "dias_saude_fisica_ruim": 10,
    #     "dificuldade_caminhar": 1,
    #     "sexo": 1,
    #     "faixa_etaria": 12,
    #     "escolaridade": 2,
    #     "renda": 3,
    # }
    # preverNovoPaciente(modelo_mock, paciente_alto_risco, limiar_clinico=0.3)

    print("Exemplo de dicionário para um paciente de alto risco:")
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
    print(paciente_exemplo)