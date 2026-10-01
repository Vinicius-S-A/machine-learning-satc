from sklearn.model_selection import cross_val_score, train_test_split

from regressao import criarModelo, cv
from knn import criarModeloKNN

# Grupos de colunas testados. Cada grupo é REMOVIDO do treino e comparado com o modelo completo.
GRUPOS_PADRAO = {
    "todas (referencia)": [],
    "sem socioeconomicas": ["renda", "escolaridade"],
    "sem acesso a saude": ["tem_plano_saude", "sem_medico_por_custo", "checou_colesterol"],
    "sem as 5": ["renda", "escolaridade", "tem_plano_saude",
                 "sem_medico_por_custo", "checou_colesterol"],
}

# Se a AUC cair menos que isso ao remover o grupo, ele é considerado descartável.
TOLERANCIA = 0.002


def _amostrar(X_train, y_train, tamanho_amostra):
    # Mesma amostra para todos os grupos, para a comparação ser justa (o teste não é tocado).
    if len(X_train) > tamanho_amostra:
        X_amostra, _, y_amostra, _ = train_test_split(
            X_train, y_train, train_size=tamanho_amostra, random_state=42, stratify=y_train
        )
        return X_amostra, y_amostra
    return X_train, y_train


def _auc(modelo, X, y):
    # Os mesmos folds (cv) são usados em todas as chamadas: a diferença entre grupos
    # vem da coluna removida, e não de uma divisão diferente dos dados.
    auc = cross_val_score(modelo, X, y, cv=cv, scoring="roc_auc")
    return auc.mean(), auc.std()


def compararSelecao(X_train, y_train, c=0.1, k=15, grupos=None, tamanho_amostra=20000, incluir_knn=True, uma_por_vez=False):
    if grupos is None:
        grupos = dict(GRUPOS_PADRAO)
    else:
        grupos = dict(grupos)

    if uma_por_vez:
        todas = sorted({col for cols in grupos.values() for col in cols})
        for col in todas:
            grupos[f"sem {col}"] = [col]

    X_knn, y_knn = _amostrar(X_train, y_train, tamanho_amostra) if incluir_knn else (None, None)

    print("\n------------- Selecao de atributos por validacao cruzada (somente treino) -------------")
    print(f"Regressao Logistica: {len(X_train)} linhas (C={c})")
    if incluir_knn:
        print(f"KNN: {len(X_knn)} linhas (k={k})")
    print(f"Descartavel se a AUC cair menos que {TOLERANCIA}\n")

    cabecalho = f"{'grupo removido':<26}{'LR AUC':>9}{'dif':>9}"
    if incluir_knn:
        cabecalho += f"{'KNN AUC':>10}{'dif':>9}"
    print(cabecalho)

    resultados = {}
    ref_lr = None
    ref_knn = None

    for nome, cols in grupos.items():
        cols_validas = [col for col in cols if col in X_train.columns]
        Xt = X_train.drop(columns=cols_validas)
        auc_lr, _ = _auc(criarModelo(c), Xt, y_train)

        auc_knn = None
        if incluir_knn:
            Xk = X_knn.drop(columns=cols_validas)
            auc_knn, _ = _auc(criarModeloKNN(k), Xk, y_knn)

        # o primeiro grupo (lista vazia) é a referência
        if ref_lr is None:
            ref_lr, ref_knn = auc_lr, auc_knn

        dif_lr = auc_lr - ref_lr
        linha = f"{nome:<26}{auc_lr:>9.4f}{dif_lr:>+9.4f}"
        if incluir_knn:
            dif_knn = auc_knn - ref_knn
            linha += f"{auc_knn:>10.4f}{dif_knn:>+9.4f}"
        print(linha)

        resultados[nome] = {"lr": auc_lr, "knn": auc_knn, "colunas": cols_validas}

    return resultados