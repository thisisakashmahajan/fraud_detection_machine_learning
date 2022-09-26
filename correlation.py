import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/transactions.csv')
print(df.columns)

correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()
