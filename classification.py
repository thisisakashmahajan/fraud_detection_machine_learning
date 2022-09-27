import pandas as pd
from sklearn.model_selection import train_test_split
from utility import transform_data, print_metrics, show_confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


models = dict({'decision_tree': DecisionTreeClassifier(),
               'random_forest': RandomForestClassifier(),
               'gradient_boost': GradientBoostingClassifier(),
               'svm': SVC(),
               'xgboost': XGBClassifier()})

df = pd.read_csv('data/transactions.csv')
data = df.copy()

X, Y = transform_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = models.get('xgboost')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print_metrics(y_test, predictions)
show_confusion_matrix(y_test, predictions, save=True)
