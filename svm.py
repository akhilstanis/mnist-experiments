# Learning
from sklearn import svm
import mnist

X = mnist.read_train_images()
Y = mnist.read_train_labels()
clf = svm.LinearSVC(penalty='l1', loss='hinge')
clf = clf.fit(X,Y)

# Testing
TX = mnist.read_test_images()
TY = mnist.read_test_labels()

predictions = clf.predict(TX)

from sklearn import metrics
print metrics.classification_report(TY, predictions)
