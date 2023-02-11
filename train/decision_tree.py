import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import pickle
from sklearn.tree import DecisionTreeClassifier
import numpy as np

X_train = np.load("TRAIN_X.npy", allow_pickle=True)
X_test = np.load("TEST_X.npy", allow_pickle=True)
y_train = np.load("TRAIN_y.npy", allow_pickle=True)
y_test = np.load("TEST_y.npy", allow_pickle=True)

tree = DecisionTreeClassifier()
print ('Fitting tree')
tree.fit(X_train, y_train)
print ('Predicting...')
y_pred = tree.predict(X_test)
print (f'Accuracy: ',  tree.score(X_test, y_test))
print(classification_report(y_test, y_pred))
print("\n")

# save model 
filename = 'tree.pkl'
pickle.dump(tree, open(filename, 'wb'))
