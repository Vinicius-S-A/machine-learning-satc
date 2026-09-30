import pandas as pd


def preverNovoPaciente(modelo_treinado, limiar_clinico, dados_paciente):
    print("\n" + "=" * 50)
    print("SISTEMA DE TRIAGEM: AVALIAÇÃO DE NOVO PACIENTE")
    print("=" * 50)

    # Cálculo da probabilidade com o modelo treinado
    probabilidades = modelo_treinado.predict_proba(dados_paciente)[0]
    prob_risco = probabilidades[1]

    print(f"Probabilidade clínica calculada: {prob_risco:.2%}")

    # Regra de decisão com limiar personalizado
    if prob_risco >= limiar_clinico:
        print("ALERTA: Paciente classificado como POSITIVO para risco de Diabetes.")
        print(f"(Limiar de segurança de {limiar_clinico * 100:.0f}% ultrapassado)")
    else:
        print("DIAGNÓSTICO: Paciente classificado como NEGATIVO para risco inicial.")

    print("=" * 50 + "\n")