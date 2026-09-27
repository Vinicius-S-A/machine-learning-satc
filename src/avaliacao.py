from sklearn.metrics import accuracy_score

def avaliarModelo(type, model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do {type}: {accuracy}")
    return accuracy