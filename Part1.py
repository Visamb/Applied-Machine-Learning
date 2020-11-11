from sklearn.tree import DecisionTreeClassifier,plot_tree
from numpy import*
from sklearn import datasets, metrics
import matplotlib.pyplot as plt


class Part1:
    def __init__(self):

        #Import digits
        self.digits = datasets.load_digits()

        self.data = self.digits.data
        self.labels = self.digits.target

        #Split data
        trainingset = int(len(self.data)*0.7)
        self.train_data = self.data[:trainingset]
        self.train_labels = self.labels[:trainingset]
        self.test_data = self.data[trainingset:]
        self.test_labels = self.labels[trainingset:]

    def fit(self):

        #Fit model
        decisiontree = DecisionTreeClassifier()
        decisiontree.fit(self.train_data,self.train_labels)

        #Plot
        plot_tree(decisiontree)
        plt.show()

        prediction = decisiontree.predict(self.test_data)
        score = metrics.classification_report(self.test_labels, prediction)
        print(metrics.confusion_matrix(self.test_labels, prediction))
        print(score)









