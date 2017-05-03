# Will Conlin, Basic Machine Learning

import numpy as np
import pandas as pd
import sklearn
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')

#Format Data

train_df = train_df.fillna(0)
test_df = test_df.fillna(0)

train_df.default_oct = train_df.default_oct.map( {"yes": 1, "no": 0} ).astype(int)

train_df['pay_1'] = train_df['pay_1']+2
train_df['pay_2'] = train_df['pay_2']+2
train_df['pay_3'] = train_df['pay_3']+2
train_df['pay_4'] = train_df['pay_4']+2
train_df['pay_5'] = train_df['pay_5']+2
train_df['pay_6'] = train_df['pay_6']+2

test_df['pay_1'] = test_df['pay_1']+2
test_df['pay_2'] = test_df['pay_2']+2
test_df['pay_3'] = test_df['pay_3']+2
test_df['pay_4'] = test_df['pay_4']+2
test_df['pay_5'] = test_df['pay_5']+2
test_df['pay_6'] = train_df['pay_6']+2

# Convert to averages to reduce overfitting

train_df['avgPayStatus'] = train_df[['pay_1', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']].mean(axis=1)
test_df['avgPayStatus'] = test_df[['pay_1', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']].mean(axis=1)

train_df['avgPayAmt'] = train_df[['pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6']].mean(axis=1)
test_df['avgPayAmt'] = test_df[['pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6']].mean(axis=1)

train_df['avgBillAmt'] = train_df[['bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6']].mean(axis=1)
test_df['avgBillAmt'] = test_df[['bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6']].mean(axis=1)

test_df['placeholder'] = 0

train_df.drop(train_df.columns[[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]], axis=1, inplace=True)
test_df.drop(test_df.columns[[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]], axis=1, inplace=True)

# Train and model

X_train = train_df.drop("default_oct", axis=1)
Y_train = train_df["default_oct"]
X_test  = test_df.drop("customer_id", axis=1)
X_train.shape, Y_train.shape, X_test.shape

# Trying different classifiers

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(acc_random_forest)

# linear_svc = LinearSVC()
# linear_svc.fit(X_train, Y_train)
# Y_pred = linear_svc.predict(X_test)
# acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
# print(acc_linear_svc)

# svc = SVC()
# svc.fit(X_train, Y_train)
# Y_pred = svc.predict(X_test)
# acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
# print(acc_svc)

# decision_tree = DecisionTreeClassifier()
# decision_tree.fit(X_train, Y_train)
# Y_pred = decision_tree.predict(X_test)
# acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
# print(acc_decision_tree)

# Output

output = pd.DataFrame({
        "customer_id": test_df["customer_id"],
        "pr_y": Y_pred
    })
output.to_csv('./output/Conlin_predictions.csv', index=False)

