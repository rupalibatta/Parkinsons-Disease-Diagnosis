from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB() 
bnb.fit(x_train, y_train) 
y_pred = bnb.predict(x_test) 
# comparing actual response values (y_test) with predicted response values (y_pred) 
from sklearn import metrics 
print("Bayes Test Accuracy:", metrics.accuracy_score(y_test, y_pred)*100)
