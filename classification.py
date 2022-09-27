import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utility import transform_data, print_metrics, show_confusion_matrix

df = pd.read_csv('data/transactions.csv')
data = df.copy()

X, Y = transform_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print_metrics(y_test, predictions)

show_confusion_matrix(y_test, predictions)
