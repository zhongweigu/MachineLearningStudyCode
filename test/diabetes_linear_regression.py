from skimage.metrics import mean_squared_error
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

print("均方误差: %.2f" % mean_squared_error(y_train, y_pred_train) )
print("均方误差: %.2f" % mean_squared_error(y_test, y_pred_test) )