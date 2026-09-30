from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)


def avaliarModelo(type, model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do {type}: {accuracy}")
    return accuracy


def avaliarModeloCompleto(type, model, X_test, y_test, limiares=(0.5,)):
    # Com ~14% de positivos a acurácia sozinha engana, então mostramos o baseline junto
    # e as demais métricas. A probabilidade é calculada uma única vez (o KNN é lento).
    prob = model.predict_proba(X_test)[:, 1]
    baseline = max(y_test.mean(), 1 - y_test.mean())

    print(f"\n=== {type} ===")
    print(f"Baseline (sempre prever a classe majoritaria): {baseline:.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, prob):.4f}")

    for limiar in limiares:
        pred = (prob >= limiar).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        print(f"\nLimiar {limiar}: TN={tn} FP={fp} FN={fn} TP={tp}")
        print(f"  Acuracia={accuracy_score(y_test, pred):.4f} | "
              f"Precisao={precision_score(y_test, pred, zero_division=0):.4f} | "
              f"Recall={recall_score(y_test, pred, zero_division=0):.4f} | "
              f"F1={f1_score(y_test, pred, zero_division=0):.4f}")