"""
This code is used to detect fraud transactions that are actually marked as non-fraud using standard banking rules
Based on these findings, dataset will be balanced

This is called as 'questioning the data'
"""
import pandas as pd

df = pd.read_csv('data/transactions.csv')

fraud = df[df['isFraud'] == 1]
non_fraud = df[df['isFraud'] == 0]

print('Fraud transactions:', len(fraud))
print('Non-fraud transactions:', len(non_fraud))

print('Minimum fraud amount:', min(fraud.amount))
print('Maximum fraud amount:', max(fraud.amount))

print('Minimum non-fraud amount:', min(non_fraud.amount))
print('Maximum non-fraud amount:', max(non_fraud.amount))
print('--------------')
print('The amount was credited to the destination')
print((non_fraud.newbalanceDest == (non_fraud.amount + non_fraud.oldbalanceDest)).value_counts())
print('--------------')
print('The amount was debited from origin')
print((non_fraud.newbalanceOrig == (non_fraud.oldbalanceOrg - non_fraud.amount)).value_counts())
print('--------------')
print('The case that amount debited and amount credited are not equal (pure fraud)')
new_proof = ((non_fraud.oldbalanceOrg - non_fraud.newbalanceOrig) - (non_fraud.newbalanceDest - non_fraud.oldbalanceDest)) \
            == 0
print(new_proof.value_counts())
print('--------------')
print('The amount same as mentioned is debited')
print(((non_fraud.oldbalanceOrg - non_fraud.newbalanceOrig) == non_fraud.amount).value_counts())
print('--------------')
print('The amount same as mentioned is credited')
print(((non_fraud.newbalanceDest - non_fraud.oldbalanceDest) == non_fraud.amount).value_counts())
