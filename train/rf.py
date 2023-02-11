import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import numpy as np

X_train = np.load("TRAIN_X.npy", allow_pickle=True)
X_test = np.load("TEST_X.npy", allow_pickle=True)

y_train = np.load("TRAIN_y.npy", allow_pickle=True)
y_test = np.load("TEST_y.npy", allow_pickle=True)

rf = RandomForestClassifier()

print ('Fitting rf')
rf.fit(X_train, y_train)

print ('Predicting...')
y_pred = rf.predict(X_test)

print (f'Accuracy: ',  rf.score(X_test, y_test))
print(classification_report(y_test, y_pred))
print("\n")

# save model 
filename = 'rf.pkl'
pickle.dump(rf, open(filename, 'wb'))
