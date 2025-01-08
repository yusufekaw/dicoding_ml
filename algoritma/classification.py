#import library
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#membangun model klasifikasi
def model(X_train, y_train, X_test):
    model = LogisticRegression()
    # Melatih model pada data latih
    model.fit(X_train, y_train)
    # Evaluasi pada data training
    y_train_pred = model.predict(X_train)
    # Memprediksi data uji
    y_test_pred = model.predict(X_test)
    return y_train_pred, y_test_pred

#hasil modeling dalam bentu data
def train_test_set(X_train, X_test, y_train, y_train_pred, y_test, y_test_pred):
    #train set / hasil modeling
    train_set = X_train.copy()
    train_set['actual'] = y_train
    train_set['predict'] = y_train_pred
    #test set / hasil klasifikasi
    test_set = X_test.copy()
    test_set['actual'] = y_test
    test_set['predict'] = y_test_pred
    return train_set, test_set

#evaluasi model
def evaluation(actual, prediction):
    cm = confusion_matrix(actual, prediction) #confusion matrix
    accuracy = accuracy_score(actual, prediction) #accuracy
    report = classification_report(actual, prediction) #precision, recall, f1 score
    return cm, accuracy, report

def cm_visualization(cm):
    # Membuat heatmap untuk confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.title("Confusion Matrix")
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.show()