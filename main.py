import Part1
from numpy import*
from sklearn.tree import DecisionTreeClassifier,plot_tree
import graphviz as g
import ToyData as td
import ID3
from collections import OrderedDict
from sklearn import tree, metrics, datasets

def returndigits ():
    """Formating of the digit dataset for classification algorithm.
    :return attributes, data, classes, target, testdata and testtarget"""

    # Load and split digits data set
    digits = datasets.load_digits()

    num_split = int(0.7 * len(digits.data))
    X_train = digits.data[:num_split]  # train data (features)
    y_train = digits.target[:num_split]  # train labels (y-values)
    X_test = digits.data[num_split:]  # test data (features)
    y_test = digits.target[num_split:]  # test labels(y-values)

    # Changing values from pixel strength to strings
    new_X_train = []
    new_X_test = []

    for pic in X_train:
        pic_list = []
        for pixel in pic:
            if pixel < 5.0:
                pic_list.append('dark')
            elif pixel > 10.0:
                pic_list.append('light')
            else:
                pic_list.append('grey')
        new_X_train.append(pic_list)

        for pic in X_test:
            pic_list = []
            for pixel in pic:
                if pixel < 5.0:
                    pic_list.append('dark')
                elif pixel > 10.0:
                    pic_list.append('light')
                else:
                    pic_list.append('grey')
            new_X_test.append(pic_list)

    #Corresponds to data
    digitdata = []
    for i in range(len(X_train)):
        new_X_train[i] = tuple(new_X_train[i])
        digitdata.append(new_X_train[i])

    #Corresponds to testdata
    digitdata2 = []
    for i in range(len(X_test)):
        new_X_test[i] = tuple(new_X_test[i])
        digitdata2.append(new_X_test[i])

    #Corresponds to classes
    digitsclasses = tuple(digits.target_names)
    print(digitsclasses)

    #Corresponds to target
    digittarget = tuple(y_train)

    #Corresponds to testtarget
    digittarget2 = tuple(y_test)

    names = digits.feature_names
    dict = OrderedDict({})
    for name in names:
        dict[name] = ['light','grey','dark']
    digitattributes = dict

    return digitattributes, digitsclasses, digitdata, digittarget,digitdata2,digittarget2

def toytree():
    """Classification of toys using ID3-Classification tree"""
    attributes, classes, data, target, data2, target2 = td.ToyData().get_data()
    id3 = ID3.ID3DecisionTreeClassifier()
    myTree = id3.fit(data, target, attributes, classes)
    print(myTree)
    plot = id3.make_dot_data()
    plot.render("testTree")
    predicted = id3.predict(data2, attributes)
    score = metrics.classification_report(list(target2), list(predicted))
    print(metrics.confusion_matrix(list(target2), predicted))
    print(score)
    return

def digittree():
    """Classification of digits using ID3-Classification tree"""
    attributes, classes, data, target,data2,target2 = returndigits()
    id3 = ID3.ID3DecisionTreeClassifier()
    myTree = id3.fit(data, target, attributes, classes)
    print(myTree)
    plot = id3.make_dot_data()
    plot.render("testTree")
    predicted = id3.predict(data2, attributes)
    score = metrics.classification_report(list(target2), list(predicted))
    print(metrics.confusion_matrix(list(target2), predicted))
    print(score)
    return

def sklearntree():
    """Classification of digits using sklearn"""
    tree = Part1.Part1()
    tree.fit()


def main():

  digittree()
  #toytree()
  #sklearntree()








if __name__ == "__main__": main()



