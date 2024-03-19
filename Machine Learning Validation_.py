import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/e99a526b2e7ccc6c3b70f53db43a87d2/raw/1605fc74aa778066bf2e6695e24d53cf65f2f447/machine-learning-carros-simulacao.csv"
data = pd.read_csv(uri).drop(columns=["Unnamed: 0"], axis=1)
data.head()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

X = data[["price", "model_age","km_per_year"]]
y = data["sold"]

SEED = 158020
np.random.seed(SEED)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.25, stratify = y)
print("We will train with %d elements and test with %d elements" % (len(train_X), len(test_X)))

from sklearn.dummy import DummyClassifier

dummy_stratified = DummyClassifier()
dummy_stratified.fit(train_X, train_y)
accuracy = dummy_stratified.score(test_X, test_y) * 100

print("The accuracy of dummy stratified was %.2f%%" % accuracy)

from sklearn.tree import DecisionTreeClassifier

SEED = 158020
np.random.seed(SEED)
model = DecisionTreeClassifier(max_depth=2)
model.fit(train_X, train_y)
predictions = model.predict(test_X)

accuracy = accuracy_score(test_y, predictions) * 100
print("The accuracy was %.2f%%" % accuracy)

X = data[["price", "model_age","km_per_year"]]
y = data["sold"]

SEED = 158020
np.random.seed(SEED)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.25, stratify = y)
print("We will train with %d elements and test with %d elements" % (len(train_X), len(test_X)))

model = DecisionTreeClassifier(max_depth=2)
model.fit(train_X, train_y)
predictions = model.predict(test_X)

accuracy = accuracy_score(test_y, predictions) * 100
print("The accuracy was %.2f%%" % accuracy)

from sklearn.model_selection import cross_validate

SEED = 301
np.random.seed(SEED)

model = DecisionTreeClassifier(max_depth=2)
results = cross_validate(model, X, y, cv = 3, return_train_score=False)
mean = results['test_score'].mean()
std_dev = results['test_score'].std()
print("Accuracy with cross validation, 3 = [%.2f, %.2f]" % ((mean - 2 * std_dev)*100, (mean + 2 * std_dev) * 100))

SEED = 301
np.random.seed(SEED)

model = DecisionTreeClassifier(max_depth=2)
results = cross_validate(model, X, y, cv = 10, return_train_score=False)
mean = results['test_score'].mean()
std_dev = results['test_score'].std()
print("Accuracy with cross validation, 10 = [%.2f, %.2f]" % ((mean - 2 * std_dev)*100, (mean + 2 * std_dev) * 100))

SEED = 301
np.random.seed(SEED)

model = DecisionTreeClassifier(max_depth=2)
results = cross_validate(model, X, y, cv = 5, return_train_score=False)
mean = results['test_score'].mean()
std_dev = results['test_score'].std()
print("Accuracy with cross validation, 5 = [%.2f, %.2f]" % ((mean - 2 * std_dev)*100, (mean + 2 * std_dev) * 100))

# Randomness in cross validate

def print_results(results):
  mean = results['test_score'].mean()
  std_dev = results['test_score'].std()
  print("Average Accuracy: %.2f" % (mean * 100))
  print("Accuracy interval: [%.2f, %.2f]" % ((mean - 2 * std_dev)*100, (mean + 2 * std_dev) * 100))

from sklearn.model_selection import KFold

SEED = 301
np.random.seed(SEED)

cv = KFold(n_splits = 10)
model = DecisionTreeClassifier(max_depth=2)
results = cross_validate(model, X, y, cv = cv, return_train_score=False)
print_results(results)

SEED = 301
np.random.seed(SEED)

cv = KFold(n_splits = 10, shuffle = True)
model = DecisionTree