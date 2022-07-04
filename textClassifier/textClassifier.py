import re
from collections import Counter
from typing import List
import math
import sys

# Hyperparameters
laplace = 0.1

# stopwords List
stopwords = "about all along also although among and any anyone anything are around because " \
            "been before being both but came come coming could did each else every for from get getting going got gotten had has have having her here hers him his how " \
            "however into its like may most next now only our out particular same she " \
            "should some take taken taking than that the then there these they this those " \
            "throughout too took very was went what when which while who why will with " \
            "without would yes yet you your" \
            "com doc edu encyclopedia fact facts free home htm html http information " \
            "internet net new news official page pages resource resources pdf site " \
            "sites usa web wikipedia www " \
            "one ones two three four five six seven eight nine ten tens eleven twelve " \
            "dozen dozens thirteen fourteen fifteen sixteen seventeen eighteen nineteen " \
            "twenty thirty forty fifty sixty seventy eighty ninety hundred hundreds " \
            "thousand thousands million millions "


# --------------------------------Step0-------------------------------

# use regular expressions to cut words
def textParse(strings):
    # strings.replace("'", " ")
    # strings.replace(",", " ")
    # strings.replace(".", " ")
    # Compile the regular expression engine
    listOfWords = re.split(r"\W", strings)
    # return lowercase
    return [word.lower() for word in listOfWords if len(word) > 1]


# stopwords set
stopwords_set = set(textParse(stopwords))


# person info
class person:
    def __init__(self, name, category, description):
        self.name = name
        self.category = category
        self.description = description
        self.wordList = []  # non-repeating word list

    def __str__(self):
        return "name:{}\ncategory:{}\ndescription:{}\nwordList:{}\n". \
            format(self.name, self.category, self.description, self.wordList)


# Remove duplicate words and filter vocabulary
def filter_persons(p: person):
    p.wordList = sorted(set(textParse(p.description)) - stopwords_set)


# Read the txt file and return the character information list
def readTxt(path):
    persons = []
    with open(path, 'r', encoding='utf8') as f:
        file = f.read().splitlines()
        index = 0
        p = person('', '', '')
        for lines in file:
            if len(lines) > 0:  # valid info
                if index == 0:  # name
                    p.name = lines
                    index += 1
                elif index == 1:  # category
                    p.category = lines
                    index += 1
                else:  # description
                    p.description += lines + ' '
                    index += 1
            else:  # space line
                index = 0
                if len(p.name) > 0:  # valid person-info
                    persons.append(p)
                p = person('', '', '')  # clear
        if len(p.name) > 0:
            persons.append(p)
        for p in persons:
            filter_persons(p)
        return persons


# Divide into training set and test set
def split_persons(persons, N):
    return persons[:N], persons[N:]


# read txt and return trainning set and test set
def prework(path, N):
    persons = readTxt(path)
    return split_persons(persons, N)


# Build a list of unique words that appear in all documents
def createVocabList(persons: List[person]):
    vocabSet = set([])
    for p in persons:
        vocabSet = vocabSet | set(p.wordList)
    return sorted(list(vocabSet))


# --------------------------------Step1-------------------------------

# Sort the labels in the training samples lexicographically, and then calculate the probability
# FreqT (C) = OccT (C)/|T|
def categoryFreq(persons_train: List[person]):
    category = sorted([p.category for p in persons_train])
    # print("category: ", category)
    category_counter = Counter(category)
    # print("Counter: ", category_counter)
    # Deduplication
    category_list = list(set(category))
    # print("list: ", category_list)
    # calculate probabilities and save as dictionariy
    category_freq = {}
    for category_name in category_list:
        category_freq[category_name] = category_counter[category_name] / len(category)
    # print("prob: ", category_prob)
    return category_freq


# Create a word vector matrix, a vocabulary on the horizontal axis,
# a category on the vertical axis, and calculate the probability
# that each word in the training set appears in this category
# FreqT (W|C)
def word2categoryFreq(persons_train: List[person]):
    vocabList = createVocabList(persons_train)
    categoryOrigin = [p.category for p in persons_train]
    categoryList = sorted(list(set(categoryOrigin)))
    # use dictionary to save FreqT (W|C), initialize 0
    word2category = {}
    for vocab in vocabList:
        word2category[vocab] = {}
        for c in categoryList:
            word2category[vocab][c] = 0

    categoryNum = Counter(categoryOrigin)
    # Use Counter to count how many times each word appears in each category
    categoryCounter = {}
    for c in categoryList:
        categoryCounter[c] = Counter()
    for p in persons_train:
        categoryCounter[p.category] += Counter(p.wordList)
    # calculate FreqT (W|C)
    for vocab in vocabList:
        for c in categoryList:
            if categoryCounter[c].get(vocab) is not None:
                word2category[vocab][c] = categoryCounter[c][vocab] / categoryNum[c]
            else:
                word2category[vocab][c] = 0.0
    return vocabList, categoryList, word2category


# --------------------------------Step2-------------------------------
# For each classification C and word W, compute the probabilities using the Laplacian correction. Let  = 0.1.
# calculate P(C)
def categoryProb(categoryFreq: dict):
    category_prob = {}
    for c, freq in categoryFreq.items():
        category_prob[c] = (freq + laplace) / (1 + len(categoryFreq) * laplace)
    return category_prob


# calculate P(W|C)
def word2categoryProb(vocabList: List, categoryList: List, word2category: dict):
    word2category_prob = {}
    for vocab in vocabList:
        word2category_prob[vocab] = {}
        for c in categoryList:
            word2category_prob[vocab][c] = (word2category[vocab][c] + laplace) / (1 + 2 * laplace)
    return word2category_prob


# --------------------------------Step3-------------------------------
#  Compute negative log probabilities to avoid underflow
nlog = lambda x: -math.log(x, 2)


def categoryProbLog(categoryProb: dict):
    category_log = {}
    for c, prob in categoryProb.items():
        category_log[c] = nlog(prob)
    return category_log


# calculate P(W|C)
def word2categoryProbLog(vocabList: List, categoryList: List, word2category_prob: dict):
    word2category_log = {}
    for vocab in vocabList:
        word2category_log[vocab] = {}
        for c in categoryList:
            word2category_log[vocab][c] = nlog(word2category_prob[vocab][c])
    return word2category_log


def trainNativeBayes(persons_train: List[person]):
    """
    :param persons_train:
    :return:
    """
    # calculate L(C)
    category_freq = categoryFreq(persons_train)
    category_prob = categoryProb(category_freq)
    category_log = categoryProbLog(category_prob)

    # calculate L(W/C)
    vocabList, categoryList, word2category = word2categoryFreq(persons_train)
    word2category_prob = word2categoryProb(vocabList, categoryList, word2category)
    word2category_log = word2categoryProbLog(vocabList, categoryList, word2category_prob)

    return vocabList, categoryList, category_log, word2category_log


# --------------------------------Step4-------------------------------
# Applying the classifier to the test data
# The prediction of the algorithm is the category C with the smallest value of L(C|B).
def classifyPerson(p: person, vocabList: List, categoryList: List, category_log: dict, word2category_log: dict):
    category_classify = {}
    p.wordList = set(p.wordList) & set(vocabList)  # Only keep the vocabulary in the training set
    for c in categoryList:
        category_classify[c] = category_log[c]
        for vocab in p.wordList:
            category_classify[c] += word2category_log[vocab][c]
    # Calculate the probability of each category
    m = min(list(category_classify.values()))
    classify_prob = {}
    for c in categoryList:
        ci = category_classify[c]
        if ci - m < 7:
            classify_prob[c] = 2 ** (ci - m)
        else:
            classify_prob[c] = 0
    s = sum(classify_prob.values())
    for c in categoryList:
        classify_prob[c] = classify_prob[c] / s
    return classify_prob


# output function
def output(p: person, categoryList: List, classify_prob: dict):
    output_info = ""
    prediction = ""
    prob = 0
    for c in categoryList:
        if classify_prob[c] > prob:
            prob = classify_prob[c]
            prediction = c
    flag = "Right." if prediction == p.category else "Wrong."
    output_info += "{}\t Prediction:{}\t {}\n".format(p.name, prediction, flag)
    probility = ""
    for c in categoryList:
        probility += "%s:%.2f   " % (c, classify_prob[c])
    output_info += probility
    return (prediction == p.category, output_info)


if __name__ == "__main__":
    path = sys.argv[1]
    N = int(sys.argv[2])
    persons_train, persons_test = prework(path, N)
    result = []
    vocabList, categoryList, category_log, word2category_log = trainNativeBayes(persons_train)
    correct = 0
    for p in persons_test:
        classify_prob = classifyPerson(p, vocabList, categoryList, category_log, word2category_log)
        tuples = output(p, categoryList, classify_prob)
        correct += tuples[0]
        result.append(tuples[1])
    result_info = "Overall accuracy: %d out of %d = %.2f." % (correct, len(persons_test), correct / len(persons_test))
    result.append(result_info)
    # write into txt
    with open("./Output.txt", mode='w') as f:
        for info in result:
            f.write(info + '\n')
            f.write('\n')