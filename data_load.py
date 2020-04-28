# Analysis imports
import numpy as np


# Doc2vec imports
import gensim
TaggedDocument = gensim.models.doc2vec.TaggedDocument


# Miscellaneous imports
from os import getcwd
from os.path import join


def load_all():
    data_dir = getcwd() + "/data/"
    preparer = Label_Reviews(data_dir)

    X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, train_unsup_reviews = preparer.load_reviews()
    X_TRAIN, X_TEST, train_unsup_reviews = preparer.label_data(X_TRAIN, X_TEST, train_unsup_reviews)

    print("X_TRAIN", len(X_TRAIN))
    print("X_TEST", len(X_TEST))
    print("Y_TRAIN", len(Y_TRAIN))
    print("Y_TEST", len(Y_TEST))
    print("train_unsup_reviews", len(train_unsup_reviews))

    return X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, train_unsup_reviews

class Label_Reviews(object):
    """
    The Label_Reviews class takes the data and labels it in the correct format for the doc2vec model
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir


    def load_reviews(self):
        # Import all data
        with open(join(self.data_dir, "test-pos.txt"), "r+") as input_file:
            test_pos_reviews = input_file.readlines()
        with open(join(self.data_dir, "test-neg.txt"), "r+") as input_file:
            test_neg_reviews = input_file.readlines()
        with open(join(self.data_dir, "train-pos.txt"), "r+") as input_file:
            train_pos_reviews = input_file.readlines()
        with open(join(self.data_dir, "train-neg.txt"), "r+") as input_file:
            train_neg_reviews = input_file.readlines()
        with open(join(self.data_dir, "train-unsup.txt"), "r+") as input_file:
            train_unsup_reviews = input_file.readlines()

        # Organize positive/negative sentiment data into arrays
        print("The length of the positive reviews used for training is: ",len(train_pos_reviews))
        print("The length of the negative reviews used for training is: ",len(train_neg_reviews))
        X_TRAIN = np.concatenate((train_pos_reviews, train_neg_reviews))
        Y_TRAIN = np.concatenate((np.ones(len(train_pos_reviews)), np.zeros(len(train_neg_reviews))))

        print("The length of the positive reviews used for testing is: ",len(test_pos_reviews))
        print("The length of the negative reviews used for testing is: ",len(test_neg_reviews))
        print("The length of the unlabeled reviews used for training is: ",len(train_unsup_reviews))
        
        X_TEST = np.concatenate((test_pos_reviews, test_neg_reviews))
        Y_TEST = np.concatenate((np.ones(len(test_pos_reviews)), np.zeros(len(test_neg_reviews))))

        return X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, train_unsup_reviews

    @staticmethod
    def label_each_review(corpus, label_prefix):
        # Label each review, as required by Doc2vec
        labeled_reviews = []
        for idx, review in enumerate(corpus):
            label = "%s_%s" % (label_prefix, idx)
            words = review.split()
            labeled_reviews.append(TaggedDocument(words=words, tags= [label]))

        return labeled_reviews


    def label_data(self, X_TRAIN, X_TEST, train_unsup_reviews):
        # Label all corpus data
        X_TRAIN = self.label_each_review(X_TRAIN, "TRAIN")
        X_TEST = self.label_each_review(X_TEST, "TEST")
        train_unsup_reviews = self.label_each_review(train_unsup_reviews, "UNLABELED")

        return X_TRAIN, X_TEST, train_unsup_reviews

if __name__ == "__main__":
    load_all()