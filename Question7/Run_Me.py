import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


data_pd = pd.read_csv('US Presidential Data.csv')


data_pd = data_pd.sample(frac=1)

train, test = train_test_split(data_pd, test_size=0.2)

X_tr = train[train.columns[1:train.shape[1]]]
Y_tr = train[train.columns[0]]
X_ts = test[test.columns[1:test.shape[1]]]
Y_ts = test[test.columns[0]]


classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(X_tr,Y_tr)
y_pred = classifier.predict(X_tr)

print(confusion_matrix(Y_tr, y_pred))
print("Train Classfication Error rate = {}".format(np.mean(y_pred != Y_tr)*100))


y_pred = classifier.predict(X_ts)
print("Test Classfication Error rate = {}".format(np.mean(y_pred != Y_ts)*100))



error = []
# Calculating error for K values between 1 and 50
Max_K = 50
for i in range(1, Max_K):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_tr, Y_tr)
    pred_i = knn.predict(X_ts)
    error.append(np.mean(pred_i != Y_ts))


plt.figure(figsize=(12, 6))
plt.plot(range(1, Max_K), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

