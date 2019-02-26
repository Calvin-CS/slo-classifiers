from keras.models import Sequential
from keras.layers import Dense
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

model = Sequential()
model.add(Dense(units=8, activation='relu', input_dim=2))
model.add(Dense(units=3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

X = iris.data[:, :2]
Y = iris.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

model.fit(x_train, y_train, epochs=5, batch_size=32)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

print(model.metrics_names)
print(loss_and_metrics)
