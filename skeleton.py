import sys, os
import numpy as np
from operator import itemgetter as ig
from sklearn.linear_model import LogisticRegression as LR
from collections import Counter
import string

# vocab = []  # the features used in the classifier

# build vocabulary


def buildvocab(numwords):
    vocab = []
    temp_words = []
    base_dir = os.getcwd()
    training_dirs = [base_dir + '/' + dirname for dirname in ('pos', 'neg')]
    stopwords = open('stopwords.txt').read().lower().split()

    # Populate vocab list with numwords most frequent words in training data,
    # minus stopwords
    for training_dir in training_dirs:
        for training_file in os.listdir(training_dir):
            with open(training_dir + '/' + training_file) as f:
                words = f.read().lower().split()
            for word in words:
                if word not in stopwords and word not in string.punctuation:
                    temp_words.append(word)
    # Counter returns list of ('word', count)
    vocab = [item[0] for item in Counter(temp_words).most_common(numwords)]
    return vocab


def vectorize(file_name, vocab):
    vector = np.zeros(len(vocab))
    with open(file_name) as fn:
        review = []
        review.extend(fn.read().lower().split())
    for word in review:
        try:
            vector[vocab.index(word)] += 1
        except ValueError:
            pass
    return vector


def make_classifier(vocab):
    base_dir = os.getcwd()
    directories = [base_dir + '/' + dirname for dirname in ('pos', 'neg')]
    reviews = []
    y = []
    for directory in directories:
        for review_file in os.listdir(directory):
            reviews.append(vectorize(directory + '/' + review_file, vocab))
            if os.path.basename(directory) == 'pos':
                y.append(1)
            else:
                y.append(-1)
    reviews = np.asarray(reviews)
    y = np.asarray(y)
    lr = LR()
    lr.fit(reviews, y)
    return lr


def test_classifier(lr, vocab):
    test = np.zeros((len(os.listdir('test')), len(vocab)))
    testfn = []
    i = 0
    y = []
    for fn in os.listdir('test'):
        testfn.append(fn)
        test[i] = vectorize(os.path.join('test', fn), vocab)
        ind = int(fn.split('_')[0][-1])
        y.append(1 if ind == 3 else -1)
        i += 1

    assert(sum(y) == 0)
    p = lr.predict(test)

    r, w = 0, 0
    for i, x in enumerate(p):
        if x == y[i]:
            r += 1
        else:
            w += 1
            print(testfn[i])
    print(r, w)


if __name__ == '__main__':
    numwords = int(raw_input("\nEnter number of words for classifiers: "))
    vocab = buildvocab(numwords)
    lr = make_classifier(vocab)
    test_classifier(lr, vocab)
