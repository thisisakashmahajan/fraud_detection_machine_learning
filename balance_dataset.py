"""
This code balances dataset using newly found transactions that are fraud but labelled as non-fraud
"""
import pandas as pd

df = pd.read_csv('data/transactions.csv')
# fraud_rule = ((df.oldbalanceOrg - df.newbalanceOrig) - (df.newbalanceDest - df.oldbalanceDest)) == 0
print(df.head())
