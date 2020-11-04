# This is a sample Python script.

# Press Skift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from numpy import *
from sklearn import svm
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt


#Digits are the matrices of digits
digits = datasets.load_digits()
num_examples = len(digits.data)
num_split = int(0.7*num_examples)

#Split into training and test data
train_features = digits.data[:num_split]
train_labels =  digits.target[:num_split]
test_features = digits.data[num_split:]
test_labels = digits.target[num_split:]

#Classifier
classifier = KNeighborsClassifier(n_neighbors=5, algorithm = 'brute')
classifier.fit(train_features,train_labels)


distances,neighbours = classifier.kneighbors(test_features)
predicted = classifier.predict(test_features)

images_and_predictions = list(zip(digits.images[num_split:], predicted))
neighbourimages = neighbours

#Plot of nearest neighbours
"""for ind in range (0,4):
    for i in range (0,5):
        plt.subplot(1, 5, i + 1)
        plt.axis('off')
        index = neighbourimages[ind,i]
        plt.imshow(digits.images[index], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()"""

#Plot of predictions
"""for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)
plt.show()"""

#Metrics for KNN
print(metrics.classification_report(test_labels,predicted))

print(metrics.confusion_matrix( test_labels, predicted))

#Cluster the data
clustering = KMeans(n_clusters=10)
clusters = clustering.fit(train_features)
centers = clustering.cluster_centers_

#Plotting the centers
for i in range (0,9):
    plt.subplot(1, 10, i + 1)
    plt.axis('off')
    image=reshape(centers[i], (8,8))
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

prediction = clustering.predict(test_features)

"""Poor classification report. Clustering is not classification"""
print(metrics.classification_report(test_labels,prediction))

"""Evaluation tools
Completeness - All members of the same class mbelong to the same cluster.
Homogeneity - All clusters contain only data points which are members of the same class.
Mutual information compares predicted and actual labels."""
print(metrics.completeness_score(test_labels, prediction))
print(metrics.homogeneity_score( test_labels, prediction))
print(metrics.adjusted_mutual_info_score( test_labels, prediction))

















