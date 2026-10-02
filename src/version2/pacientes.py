
#   colunas 0/1 : 0 = não | 1 = sim
#   sexo        : 0 = feminino | 1 = masculino
#   saude_geral : 1 = excelente, 2 = muito boa, 3 = boa, 4 = razoável, 5 = ruim
#   dias_saude_* : dias ruins nos últimos 30 dias (0 a 30)
#   faixa_etaria : 1 = 18-24 | 2 = 25-29 | 3 = 30-34 | 4 = 35-39 | 5 = 40-44 | 6 = 45-49 | 7 = 50-54
#                  8 = 55-59 | 9 = 60-64 | 10 = 65-69 | 11 = 70-74 | 12 = 75-79 | 13 = 80 ou mais

PACIENTES = {
    # Jovem, peso normal, ativo, sem condições de saúde
    "Baixo risco": {
        "pressao_alta": 1,
        "colesterol_alto": 1,
        "imc": 40,
        "avc":1,
        "doenca_cardiaca_ou_infarto": 1,
        "atividade_fisica": 1,
        "consome_frutas": 1,
        "consome_vegetais": 1,
        "saude_geral": 5,
        "dias_saude_fisica_ruim": 30,
        "dificuldade_caminhar": 1,
        "sexo": 0,
        "faixa_etaria": 12,
    },

    # Meia-idade, sobrepeso, pressão alta, sedentário
    # "Risco intermediario": {
    #     "pressao_alta": 1,
    #     "colesterol_alto": 0,
    #     "checou_colesterol": 1,
    #     "imc": 29,
    #     "fumante": 0,
    #     "avc": 0,
    #     "doenca_cardiaca_ou_infarto": 0,
    #     "atividade_fisica": 0,
    #     "consome_frutas": 1,
    #     "consome_vegetais": 0,
    #     "alcool_excessivo": 0,
    #     "tem_plano_saude": 1,
    #     "sem_medico_por_custo": 0,
    #     "saude_geral": 3,
    #     "dias_saude_mental_ruim": 3,
    #     "dias_saude_fisica_ruim": 5,
    #     "dificuldade_caminhar": 0,
    #     "sexo": 1,
    #     "faixa_etaria": 8,
    #     "escolaridade": 4,
    #     "renda": 5,
    # },

    # # Idoso, obesidade, pressão e colesterol altos, doença cardíaca, saúde ruim
    # "Alto risco": {
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
    # },

    # # Igual ao "Baixo risco", mas só com IMC muito alto: mostra o peso de UM fator isolado
    # "Somente IMC alto": {
    #     "pressao_alta": 0,
    #     "colesterol_alto": 0,
    #     "checou_colesterol": 1,
    #     "imc": 38,
    #     "fumante": 0,
    #     "avc": 0,
    #     "doenca_cardiaca_ou_infarto": 0,
    #     "atividade_fisica": 1,
    #     "consome_frutas": 1,
    #     "consome_vegetais": 1,
    #     "alcool_excessivo": 0,
    #     "tem_plano_saude": 1,
    #     "sem_medico_por_custo": 0,
    #     "saude_geral": 1,
    #     "dias_saude_mental_ruim": 0,
    #     "dias_saude_fisica_ruim": 0,
    #     "dificuldade_caminhar": 0,
    #     "sexo": 0,
    #     "faixa_etaria": 3,
    #     "escolaridade": 6,
    #     "renda": 7,
    # },
}