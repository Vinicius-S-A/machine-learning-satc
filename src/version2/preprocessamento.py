"""
preprocessamento.py — Version 2 Refatorada
-------------------------------------------
Correções aplicadas:
  1. `colunasNumericas` agora é uma FUNÇÃO que retorna uma lista estática a partir
     de X_train. O callable nunca é passado ao ColumnTransformer; em vez disso,
     cada módulo chama `obterColunasNumericas(X_train)` e passa a lista resultante.
     Isso elimina o risco de o scikit-learn invocar o callable com X_test e obter
     um conjunto de colunas diferente do usado no fit.
  2. Semente global fixada aqui para garantir reprodutibilidade em qualquer módulo
     que importe este arquivo.
  3. Documentação dos riscos do flag remover_duplicatas.
"""

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ─── Semente global ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# ───────────────────────────────────────────────────────────────────────────────

CAMINHO_PADRAO = "data/cdc_diabetes.csv"

TRADUCAO = {
    "Diabetes_binary": "classe",
    "HighBP": "pressao_alta",
    "HighChol": "colesterol_alto",
    "CholCheck": "checou_colesterol",
    "BMI": "imc",
    "Smoker": "fumante",
    "Stroke": "avc",
    "HeartDiseaseorAttack": "doenca_cardiaca_ou_infarto",
    "PhysActivity": "atividade_fisica",
    "Fruits": "consome_frutas",
    "Veggies": "consome_vegetais",
    "HvyAlcoholConsump": "alcool_excessivo",
    "AnyHealthcare": "tem_plano_saude",
    "NoDocbcCost": "sem_medico_por_custo",
    "GenHlth": "saude_geral",
    "MentHlth": "dias_saude_mental_ruim",
    "PhysHlth": "dias_saude_fisica_ruim",
    "DiffWalk": "dificuldade_caminhar",
    "Sex": "sexo",
    "Age": "faixa_etaria",
    "Education": "escolaridade",
    "Income": "renda",
}


def obterColunasNumericas(X: pd.DataFrame) -> list[str]:
    """
    Retorna a lista ESTÁTICA de colunas não-binárias de X.

    Diferença crítica em relação ao código original
    ------------------------------------------------
    O original definia `colunasNumericas` como um callable e o passava
    diretamente ao ColumnTransformer:

        ColumnTransformer([("num", StandardScaler(), colunasNumericas)])

    O scikit-learn chama esse callable DUAS vezes internamente:
      - com X_train durante o fit()
      - com X_test durante o transform()

    Se X_test (por acaso, em amostras pequenas ou dados reais novos) tiver
    uma coluna que assume um terceiro valor único enquanto em X_train ela era
    binária, as duas chamadas retornam listas de comprimentos diferentes. O
    pipeline quebra silenciosamente ou — pior — aplica o scaler em colunas
    erradas sem nenhum aviso.

    A solução correta é chamar esta função UMA VEZ com X_train e passar a
    lista resultante (objeto estático) ao ColumnTransformer.
    """
    return [col for col in X.columns if X[col].nunique() > 2]


def carregarDados(
    caminho: str = CAMINHO_PADRAO,
    tamanho: int | None = None,
    remover_duplicatas: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Carrega, limpa e divide o dataset CDC em treino (70 %) e teste (30 %).

    Parâmetros
    ----------
    caminho : str
        Caminho para o arquivo CSV.
    tamanho : int | None
        Amostra estratificada desse tamanho (mantém a proporção de diabéticos).
        Útil para acelerar o KNN em experimentos exploratórios.
    remover_duplicatas : bool
        Padrão False. Com ~250 mil pessoas e poucas opções por coluna,
        registros duplicados provavelmente representam indivíduos distintos.
        Remover descarta informação real e altera a proporção de diabéticos.
        ATENÇÃO: quando True, a remoção ocorre ANTES do split — se um registro
        duplicado existir, ele não aparecerá em treino e teste ao mesmo tempo,
        o que é correto. Mas a perda de dados deve ser documentada no artigo.

    Retorna
    -------
    X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(caminho)
    df = df.drop(columns=["ID"], errors="ignore")

    nulos = df.isnull().sum().sum()
    if nulos > 0:
        print(f"[AVISO] {nulos} valores nulos encontrados — linhas removidas.")
        df = df.dropna()

    if remover_duplicatas:
        antes = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"[INFO] Duplicatas removidas: {antes - len(df)} linhas ({antes} → {len(df)}).")

    df = df.rename(columns=TRADUCAO)

    if "classe" not in df.columns:
        raise ValueError(
            f"Coluna alvo 'Diabetes_binary' não encontrada. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    df = df.astype(int)

    proporcao = df["classe"].mean()
    print(f"[INFO] Proporção de positivos (diabéticos/pré): {proporcao:.2%} "
          f"({df['classe'].sum()} de {len(df)})")

    if tamanho is not None and tamanho < len(df):
        _, df = train_test_split(
            df, test_size=tamanho, random_state=SEED, stratify=df["classe"]
        )
        df = df.reset_index(drop=True)

    X = df.drop("classe", axis=1)
    y = df["classe"]

    return train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)