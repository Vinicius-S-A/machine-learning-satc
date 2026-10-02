# é NECESSÁRIO rodar esse script apenas uma vez para baixar o dataset do CDC e salvar em um arquivo CSV do /data

import pandas as pd
from ucimlrepo import fetch_ucirepo

cdc = fetch_ucirepo(id=891)

df = pd.concat([cdc.data.features, cdc.data.targets], axis=1)

print(df.shape)              # esperado: (253680, 22)
print(df.columns.tolist())
print(df["Diabetes_binary"].value_counts(normalize=True))

df.to_csv("data/cdc_diabetes.csv", index=False)