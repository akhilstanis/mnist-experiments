# Learning
from sklearn import naive_bayes
import mnist

X = mnist.read_train_images()
Y = mnist.read_train_labels()
# clf = naive_bayes.GaussianNB()
# clf = naive_bayes.MultinomialNB()
clf = naive_bayes.BernoulliNB()
clf = clf.fit(X,Y)

# Testing
TX = mnist.read_test_images()
TY = mnist.read_test_labels()

predictions = clf.predict(TX)

from sklearn import metrics
print metrics.classification_report(TY, predictions)
