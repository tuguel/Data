from sklearn import svm

X = [[1], [2], [3], [4], [5], [6]]

y = [7000,9000,5000,11000,10000,13000]

regr = svm.SVR(gamma='scale')

regr.fit(X,y)
print(regr.predict([[7]]))

