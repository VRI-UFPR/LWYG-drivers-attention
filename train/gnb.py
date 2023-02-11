import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import pickle
from sklearn.naive_bayes import GaussianNB
import numpy as np

X_train = np.load("TRAIN_X.npy", allow_pickle=True)
X_test = np.load("TEST_X.npy", allow_pickle=True)

y_train = np.load("TRAIN_y.npy", allow_pickle=True)
y_test = np.load("TEST_y.npy", allow_pickle=True)

gnb = GaussianNB()

print ('Fitting gnb')
nb.fit(X_train, y_train)

print ('Predicting...')
y_pred = gnb.predict(X_test)

print (f'Accuracy: ', gnb.score(X_test, y_test))
print(classification_report(y_test, y_pred))
print("\n")


# save model 
filename = 'gnb.pkl'
pickle.dump(gnb, open(filename, 'wb'))
