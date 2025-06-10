
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

credit_card_data = pd.read_csv('creditcard.csv')

print("Initial Dataset Info:")
print(credit_card_data.info())

print("\nMissing values in each column:")
print(credit_card_data.isnull().sum())

print("\nDistribution of Legit vs Fraud transactions:")
print(credit_card_data['Class'].value_counts())

legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

legit_sample = legit.sample(n=len(fraud), random_state=1)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression()
model.fit(X_train, Y_train)


X_train_prediction = model.predict(X_train)
training_accuracy = accuracy_score(Y_train, X_train_prediction)
print("\nAccuracy on Training data: ", training_accuracy)

X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, X_test_prediction)
print("Accuracy on Test data: ", test_accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, X_test_prediction))

print("\nClassification Report:")
print(classification_report(Y_test, X_test_prediction))
