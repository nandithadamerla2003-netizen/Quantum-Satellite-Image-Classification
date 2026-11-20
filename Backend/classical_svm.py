from sklearn.svm import SVC


def create_classical_svm():

    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    )

    return model


def train_classical_svm(X_train, y_train):

    model = create_classical_svm()

    print("Training Classical SVM Model...")

    model.fit(X_train, y_train)

    print("Classical SVM Training Completed!")

    return model