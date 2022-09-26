"""
This code balances dataset using newly found transactions that are fraud but labelled as non-fraud
"""
import pandas as pd

df = pd.read_csv('data/transactions.csv')

df['origin_fraud'] = ((df.oldbalanceOrg - df.newbalanceOrig) - (df.newbalanceDest - df.oldbalanceDest)) == 0
fraud = df[df['origin_fraud'] == True].index

for i in fraud:
    df.at[i, 'isFraud'] = 1

df.drop(columns=['origin_fraud'], inplace=True)
df.to_csv('data/transactions.csv', index=False)
