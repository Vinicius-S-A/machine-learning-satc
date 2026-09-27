from sklearn.neighbors import KNeighborsClassifier

def treinarKNN(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    
    return model