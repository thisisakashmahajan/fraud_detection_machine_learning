import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/transactions.csv')
data = df.copy()

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

X = data.drop(columns=['isFraud'])
Y = data.isFraud

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
model = DecisionTreeClassifier()

model.fit(X_train, y_train)
predictions = model.predict(X_test)
testing_score = accuracy_score(y_test, predictions)
print(testing_score)

matrix = confusion_matrix(y_test, predictions)
print(matrix)
map = sns.heatmap(matrix, annot=True, fmt='')
plt.show()
