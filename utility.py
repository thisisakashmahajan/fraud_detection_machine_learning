# Utility code to support classifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def transform_data(data):
    numeric_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    scaler = StandardScaler()
    encoder = LabelEncoder()

    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    encoded_type = pd.DataFrame(encoder.fit_transform(data['type']))
    encoded_fraud = pd.DataFrame(encoder.fit_transform(data['isFraud']))

    encoded_type.columns = ['type']
    encoded_fraud.columns = ['isFraud']

    data.drop(columns=['type', 'isFraud', 'nameOrig', 'nameDest'], axis=1, inplace=True)
    data = pd.concat([data, encoded_type, encoded_fraud], axis=1)

    x = data.drop(columns=['isFraud'])
    y = data.isFraud

    return x, y


def print_metrics(original, predictions):
    score = accuracy_score(original, predictions)
    print('Accuracy of model:', round(score * 100, 2))
    print('Classification report -----------------------')
    print(classification_report(original, predictions))


def show_confusion_matrix(original, predictions, save=False, title='Confusion Matrix'):
    matrix = confusion_matrix(original, predictions)
    plt.figure(figsize=(8, 8)).set_dpi(256)
    plt.title(title)
    sns.heatmap(matrix, annot=True, fmt='')
    if save:
        plt.savefig('confusion_matrix.png', dpi=256)
    plt.show()
