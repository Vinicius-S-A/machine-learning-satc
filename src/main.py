import pandas as pd

df = pd.read_csv("data/diabetes.csv")

# ----------------------------------------------------------------------------------------------------

#mostra apenas as primeiras linhas, para ver se está tudo ok
print(df.head())

# ----------------------------------------------------------------------------------------------------
# LIMPEZA DE DADOS
# ----------------------------------------------------------------------------------------------------

# mostra as infos
print(df.info())

# mostra se existem valores nulos
print(df.isnull().sum())

# média, mnimo, máximo etc
print(df.describe())

# esta linha mostra se existem duplicados, e quantos existem,
print("Duplicados:", df.duplicated().sum())
# no dataset original foram encontrados 269 de 520 registros...
# porém foi mantido pois não se sabe se são realmente duplicados ou se são registros de pessoas diferentes com os mesmos dados

# comentário extra referente a aula:
# Em outros projetos seria importante, pois um modelo que retornasse positive para tudo, 
# teria uma accuracy de alta e não seria um bom modelo, como foi visto em aula.

print(df["class"].value_counts())
# dataset original retorna:
# Positive    320
# Negative    200 
# 
# OU SEJA "61,5%" para "38,5%", poderia ser considerado desbalanceado, mas não é um 
# desbalanceamento tão grande, então não será feito nenhum tratamento para balancear os dados

# comentário extra referente a aula:
# Em outros projetos seria importante, pois um modelo que retornasse positive para tudo, 
# teria uma accuracy de 61,5% e não seria um bom modelo, como foi visto em aula.

# -----------------------------------------------------------------------------------------------------
# TRATAMENTO DE DADOS
# -----------------------------------------------------------------------------------------------------

# uma simples tradução, apenas para ficar melhor de visualizar
translate = {
    "Age": "idade",
    "Gender": "genero",
    "Polyuria": "poliuria",
    "Polydipsia": "polidipsia",
    "sudden weight loss": "perda_peso_subita",
    "weakness": "fraqueza",
    "Polyphagia": "polifagia",
    "Genital thrush": "candidiase_genital",
    "visual blurring": "visao_embacada",
    "Itching": "coceira",
    "Irritability": "irritabilidade",
    "delayed healing": "cicatrizacao_atrasada",
    "partial paresis": "paresia_parcial",
    "muscle stiffness": "rigidez_muscular",
    "Alopecia": "alopecia",
    "Obesity": "obesidade",
    "class": "classe"
}

df = df.rename(columns=translate)

binaryColumns = {
    "poliuria",
    "polidipsia",
    "perda_peso_subita",
    "fraqueza",
    "polifagia",
    "candidiase_genital",
    "visao_embacada",
    "coceira",
    "irritabilidade",
    "cicatrizacao_atrasada",
    "paresia_parcial",
    "rigidez_muscular",
    "alopecia",
    "obesidade"
}

for column in binaryColumns:
    df[column] = df[column].map({
        "Yes": 1,
        "No": 0
    })

df["genero"] = df["genero"].map({
    "Male": 1,
    "Female": 0
})

df["classe"] = df["classe"].map({
    "Positive": 1,
    "Negative": 0
})

print(df.head())

# -----------------------------------------------------------------------------------------------------
# Divisão de X e Y
# -----------------------------------------------------------------------------------------------------

x = df.drop("classe", axis=1)
y = df["classe"]

print("x: ", x.head())
print("y: ", y.head())