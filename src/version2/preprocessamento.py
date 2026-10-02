import pandas as pd
from sklearn.model_selection import train_test_split

def colunasNumericas(X):
    # Colunas binárias (0/1) passam direto pelo modelo.
    # As demais (imc, faixa_etaria, saude_geral, renda...) têm escalas diferentes e são padronizadas.
    # Usada pelos pipelines do KNN e da regressão logística.
    return [coluna for coluna in X.columns if X[coluna].nunique() > 2]

def carregarDados(caminho, tamanho):
    df = pd.read_csv(caminho)

    # ----------------------------------------------------------------------------------------------------
    # LIMPEZA DE DADOS
    # ----------------------------------------------------------------------------------------------------

    # print(df.head())
    # print(df.info())
    # print(df.describe())
    # print(df["Diabetes_binary"].value_counts(normalize=True))
    # o dataset tem cerca de 14% de positivos (diabetes ou pré-diabetes): é DESBALANCEADO...
    # Um modelo que sempre responde "não" acerta 86%, então a acurácia sozinha engana
    # Por isso a avaliação usa AUC, precisão, recall e análise de limiar

    # remover colunas indesejadas/ruído
    list = [
        "Education", 
        "Income", 
        "CholCheck", 
        "MentHlth", 
        "AnyHealthcare", 
        "NoDocbcCost", 
        "HvyAlcoholConsump", 
        "Smoker",
        # "PhysActivity",
        # "Fruits",
        # "Veggies",
        # "GenHlth",
        # "PhysHlth",
        # "DiffWalk",
        # "Stroke"
        ]
    for col in list:
        if col in df.columns:
            df = df.drop(columns=[col])

    print("Duplicados:", df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # -----------------------------------------------------------------------------------------------------
    # TRATAMENTO DE DADOS
    # -----------------------------------------------------------------------------------------------------

    # tradução dos nomes, apenas para ficar melhor de visualizar
    # (nomes que não existirem no arquivo são simplesmente ignorados)
    translate = {
        "Diabetes_binary": "classe",
        "HighBP": "pressao_alta",
        "HighChol": "colesterol_alto",
        # "CholCheck": "checou_colesterol",
        "BMI": "imc",
        # "Smoker": "fumante",
        "Stroke": "avc",
        "HeartDiseaseorAttack": "doenca_cardiaca_ou_infarto",
        "PhysActivity": "atividade_fisica",
        "Fruits": "consome_frutas",
        "Veggies": "consome_vegetais",
        # "HvyAlcoholConsump": "alcool_excessivo",
        # "AnyHealthcare": "tem_plano_saude",
        # "NoDocbcCost": "sem_medico_por_custo",
        "GenHlth": "saude_geral",
        # "MentHlth": "dias_saude_mental_ruim",
        "PhysHlth": "dias_saude_fisica_ruim",
        "DiffWalk": "dificuldade_caminhar",
        "Sex": "sexo",
        "Age": "faixa_etaria",
        # "Education": "escolaridade",
        # "Income": "renda"
    }
    df = df.rename(columns=translate)

    if "classe" not in df.columns:
        raise ValueError(f"Coluna alvo 'Diabetes_binary' não encontrada. Colunas do arquivo: {list(df.columns)}")

    # (faixa_etaria 1-13, saude_geral 1-5, escolaridade 1-6, renda 1-8) ficam como números
    df = df.astype(int)

    # para rodar o KNN mais rápido, se tiver o tamanho
    if tamanho is not None and tamanho < len(df):
        _, df = train_test_split(df, test_size=tamanho, random_state=42, stratify=df["classe"])
        df = df.reset_index(drop=True)

    # -----------------------------------------------------------------------------------------------------
    # Divisão de X e Y
    # -----------------------------------------------------------------------------------------------------

    x = df.drop("classe", axis=1)
    y = df["classe"]

    # 30% de dados de teste e 70% de dados de treino
    # stratify = y preserva aproximadamente a proporção das classes.

    return train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)