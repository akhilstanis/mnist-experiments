# Learning
from sklearn.ensemble import RandomForestClassifier
import mnist

X = mnist.read_train_images()
Y = mnist.read_train_labels()
clf = RandomForestClassifier(n_estimators=20)
clf = clf.fit(X,Y)

# Testing
TX = mnist.read_test_images()
TY = mnist.read_test_labels()

predictions = clf.predict(TX)

from sklearn import metrics
print metrics.classification_report(TY, predictions)
