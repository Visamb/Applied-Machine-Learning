from collections import Counter
from graphviz import *
from graphviz import Digraph
from numpy import*


class ID3DecisionTreeClassifier :


    def __init__(self, minSamplesLeaf = 1, minSamplesSplit = 2) :

        self.__nodeCounter = 0

        # the graph to visualise the tree
        self.__dot = Digraph(comment='The Decision Tree')

        # suggested attributes of the classifier to handle training parameters
        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit

        #attributes
        self.attributes = 0
        self.classes = 0
        self.target = 0
        self.data = 0
        self.mostc = 0
        self.val = 0





    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def new_ID3_node(self):



        #Find best split = root node

        node = {'id': self.__nodeCounter, 'value': None, 'label': None, 'attribute': None, 'entropy': None, 'samples': None,
                         'classCounts': None, 'nodes': None}






        self.__nodeCounter += 1
        return node

    # adds the node into the graph for visualisation (creates a dot-node)
    def add_node_to_graph(self, node, parentid=-1):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != -1):
            self.__dot.edge(str(parentid), str(node['id']))
            nodeString += "\n" + str(parentid) + " -> " + str(node['id'])

        print(nodeString)

        return


    # make the visualisation available
    def make_dot_data(self) :
        return self.__dot

        # For you to fill in; Suggested function to find the best attribute to split with, given the set of
        # remaining attributes, the currently evaluated data and target.
    def find_split_attr(self):
        """Finds the best split attribute"""

        #Targets and data
        targets = self.target
        data = self.data

        #Classes (labels, 0-9 for digits)
        classes = self.classes

        #Attributes are stored in Dictionary (Each column is an attribute with values of rows in digits)
        attributes = self.attributes

        #Total entropy
        totentr = 0

        #Count occurences of classes in targets
        count_tar = {}

        for c in classes:
            count_tar[c] = 0

        for target in targets:
            count_tar[target] += 1

        # Calculating entropy
        for c in classes:
            if int(count_tar[c]) != 0:
             totentr += -int(count_tar[c]) / sum(list(count_tar.values())) * math.log(int(count_tar[c]) / sum(list((count_tar.values()))), 2)


        #Entropy for each attribute
        attr_entr = {}
        #Attribute is key, for example color,size or column for digits
        for attr in attributes:
            attr_entr[attr] = 0

        #Iterate through dictionary
        for i, attr in enumerate(attributes):

            #For each attribute value, example y,g,b in color or rows in digit
            for sample in attributes[attr]:
                count_samp = {}

                #Counter for each of the classes in the sample (ex. Count occurences of + and - for yellow)
                for c in classes:
                    count_samp[c] = 0
                entr_samp = 0  # entropy for each sample
                for j in range(len(data)):
                    if data[j][i] == sample:
                        count_samp[targets[j]] += 1  # counting + and - for this sample

                #Calculate entropy for this sample

                if sum(list(count_samp.values())) != 0:
                    for c in count_samp:
                        if count_samp[c] != 0:
                            entr_samp += -count_samp[c] / sum(list(count_samp.values())) * math.log(count_samp[c] / sum(list(count_samp.values())), 2)

                entr_samp *= sum(list(count_samp.values())) / len(data)
                #Add entropy to entropy of attribute
                attr_entr[attr] += entr_samp

            attr_entr[attr] = totentr - attr_entr[attr] #Information gain from attribute


        best_attr = sorted(attr_entr, key=attr_entr.get, reverse=True)[0]


        return best_attr #Returns best attribute as string

    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def fit(self, data, target, attributes, classes):
        """Recursive implementation of the ID3-algorithm"""

        #Add to self
        self.attributes = attributes
        self.data = data
        self.target = target
        self.classes = classes


        root = self.new_ID3_node()
        root.update({'label': None, 'attribute': None, 'entropy': None, 'samples': len(target),
                         'classCounts': Counter(target).most_common(), 'nodes': None })



        #If samples is empty, add leaf node with label = most common value in samples
        if len(target) == 0:
            root.update({'label': self.mostc})
            self.add_node_to_graph(root)
            return root

        # If all targets are same return the node (leaf)
        if target.count(target[0]) == len(target):
            mostcommon = Counter(target).most_common()
            root.update({'label': mostcommon[0][0]})
            root.update({'classCounts': mostcommon})
            self.add_node_to_graph(root)
            return root

        if len(attributes) == 0:
            mostcommon = Counter(target).most_common()
            root.update({'label':mostcommon[0][0]})
            root.update({'classCounts': mostcommon})
            self.add_node_to_graph(root)
            return root

        splitattribute = self.find_split_attr()
        newattributes = attributes.copy()
        newattributes.pop(splitattribute)
        root.update({'attribute': splitattribute})



        #Length of data
        l = len(data)
        #Number of different values in attribute
        nbr = len(attributes[splitattribute])
        #Index of attribute
        attribute_index = list(attributes).index(splitattribute)
        #Splitted data
        datasplit = []
        targetsplit = []

        for n in range(nbr):
            templist = []
            temptarget = []
            for i in range(l):
                #If the data at index(i,attribute) == value at attribute
                if data[i][attribute_index] == attributes[splitattribute][n]:
                    tlist = list(data[i])
                    tlist.remove(data[i][attribute_index])
                    tup = tuple(tlist)
                    templist.append(tup)
                    temptarget.append(target[i])
            datasplit.append(templist)
            targetsplit.append(temptarget)

        mostcommon = Counter(target).most_common()
        self.mostc = mostcommon[0][0]
        self.add_node_to_graph(root)
        for n in range(nbr):
            child = self.fit(datasplit[n],targetsplit[n],newattributes,classes)
            val = attributes[splitattribute][n]
            root.update({'Nodes': child['id']})
            child.update({'value':val})
            self.add_node_to_graph(child,root['id'])
        return root

    def predict(self, data, tree) :

        predicted = list()
        root = self.fit(self.data,self.target,self.attributes,self.classes)






        # fill in something more sensible here... root should become the output of the recursive tree creation
        return predicted