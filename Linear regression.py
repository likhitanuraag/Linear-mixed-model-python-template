import pandas as pd
import numpy as np
import sklearn
from sklearn import linear model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("your file here.csv", sep = ";")
data = data[["x-col1", "x-col2", "x-col3", "x-col4"]]
predict = "y-col"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)


# Comment to use other model.pickle-----------------------------------

best = 0
    for _in range(number of times you want to run):
        
    x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

    linear = Linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
            best = acc
            with open("whatever file name you want.pickle", "wb") as f:
                pickle.dump(linear, f)

#-----------------------------------

pickle_in = open("the file name that you created before.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coeff_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# to easily see correlations
p = 'x-col'
style.use("ggplot")
pyplot.scatter(data[p], data["y-col"])
pyplot.xlabel(p)
pyplot.ylabel("what ever label you want to put on y axis")
pyplot.show()