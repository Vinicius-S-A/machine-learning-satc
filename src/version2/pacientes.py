
#   colunas 0/1 : 0 = não | 1 = sim
#   sexo        : 0 = feminino | 1 = masculino
#   saude_geral : 1 = excelente, 2 = muito boa, 3 = boa, 4 = razoável, 5 = ruim
#   dias_saude_* : dias ruins nos últimos 30 dias (0 a 30)
#   faixa_etaria : 1 = 18-24 | 2 = 25-29 | 3 = 30-34 | 4 = 35-39 | 5 = 40-44 | 6 = 45-49 | 7 = 50-54
#                  8 = 55-59 | 9 = 60-64 | 10 = 65-69 | 11 = 70-74 | 12 = 75-79 | 13 = 80 ou mais

PACIENTES = {
    # Meia-idade, sobrepeso, pressão alta, sedentário
    "Risco intermediario": {
        "pressao_alta": 1,
        "colesterol_alto": 0,
        # "checou_colesterol": 1,
        "imc": 29,
        # "fumante": 0,
        "avc": 0,
        "doenca_cardiaca_ou_infarto": 0,
        "atividade_fisica": 1,
        "consome_frutas": 1,
        "consome_vegetais": 0,
        "saude_geral": 5,
        "dias_saude_fisica_ruim": 5,
        "dificuldade_caminhar": 0,
        "sexo": 1,
        "faixa_etaria": 6,
    },
}