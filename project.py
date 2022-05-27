from sklearn import datasets
from sklearn.model_selection import train_test_split

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler 


# ---------------------------------------------------------------------------------

wine = datasets.load_wine()

#print("Features : " , wine.feature_names)

#print("--------------------------------------------------------------")
#print("Labels : " , wine.target_names)

# ---------------------------------------------------------------------------------

X = wine.data
Y = wine.target

X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size=0.3 , random_state=109)

gnb = GaussianNB()

gnb.fit(X_train , Y_train)

predicted = gnb.predict(X_test)

print("accuracy score :- ", (accuracy_score(Y_test, predicted)*100) , "%" )







