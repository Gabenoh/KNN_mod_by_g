import numpy as np
from model.KNN import KNearest_Neighbours
from sklearn.model_selection import train_test_split
from sklearn import datasets


def load_data():

    iris = datasets.load_iris()

    train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)

    return train_x, test_x, train_y, test_y


train_set_x, test_set_x, train_set_y, test_set_y = load_data()

k = 4  # кількість сусідів

model = KNearest_Neighbours(k)
model.fit(train_set_x, train_set_y)

y_predictions = model.predict(test_set_x)

actual = list(test_set_y)
accuracy = (y_predictions == test_set_y).mean()
print('modify KNN with_g score - ', accuracy)

'''
test model with KNN in sklearn
'''

from sklearn.neighbors import KNeighborsClassifier
modell = KNeighborsClassifier(n_neighbors=4)
modell.fit(train_set_x, train_set_y)
pred = modell.predict(test_set_x)
print('default KNeighborsClassifier score - ', (test_set_y == pred).mean())

'''
results:

modify KNN with_g score -  1.0
default KNeighborsClassifier score -  0.98

'''
