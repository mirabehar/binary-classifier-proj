import re
import math
import collections
from features import *
import string
import timeit
import pandas as pd
from  numpy import shape
import scipy as sp
from zlib import crc32

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords

from nltk.metrics import precision, recall, f_measure

from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

stop_words = list(stopwords.words('english'))

'''
This module contains all classifiers
'''


def classify_bow_NB(csv_file, get_feats=bow_feats_NB):
    '''
    NaiveBayes classifier using Bag of Words, no counts.

    Get_feats is a callable for etxracting feats: 
    - bow_feats: unigram bow
    - bgram_feats: bigram bow
    - NE_feats_NB: Named Entity (NE) bow unigrams (?)

    '''
    print("\nclassify_bow_NB ", get_feats.__name__)
    df = pd.read_csv(csv_file)
    ## randomize - not needed because split_test_train randomizes
    # df = dataset.sample(frac=1)

    ## cast to unicode
    df["sent"] = df.sent.values.astype('U')

    X = df['sent']
    y = df["s_spoiler"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    test = list(zip(X_test, y_test))
    train = list(zip(X_train, y_train))

    test_feats = [(get_feats(sent), label) for sent, label in test]
    train_feats = [(get_feats(sent), label) for sent, label in train]


    classify = nltk.NaiveBayesClassifier.train(train_feats)


    print("NLTK Naive Bayes:")

    classify.show_most_informative_features(40)

    ## METRICS
    ## calculate precision/recall/f1
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(test):
        """ makes dicts refsets/testsets where for ex:
        {0: {<sents> 1, 3, 42}, 1: {2, 6, 55}}
        recording which sents are at labels 0/1 per
        gold standard vs prediction
        """
        refsets[label].add(i)
        observed = classify.classify(get_feats(feats))
        testsets[observed].add(i)


    for label in [0,1]:
        print("\nFor label", label, ":")
        #The support is the number of occurrences of each class in y_true
        print("support: ", len(refsets[label]))
        # print("support: ", len(testsets[label]))
        prec = precision(refsets[label], testsets[label])
        print( 'Precision:', prec )
        rec = recall(refsets[label], testsets[label])
        print( 'Recall:', rec )
        print( 'F1:', f_measure(refsets[label], testsets[label], alpha=0.5) )

    acc = nltk.classify.accuracy(classify, test_feats)
    print("Accuracy score:", acc)


    

## SAME AS classify_many_feats(feat_list=[])
def classify_bow_counts(csv_file, 
                        classifier_type=LinearSVC, 
                        ngram_range=(1,1)):
    '''
    Classifier with counts vetcorizer.
    Classifier_type: LinearSVC, MultinomialNB, LinearRegression (?)
    Ngram_range: (1,1) unigrams, (2,2) only bigrams, (1,2) unigrams + bigrams
    '''
    print("\nclassify_bow_counts")
    df = pd.read_csv(csv_file)
    ## randomize - not needed because split_test_train randomizes
    # df = dataset.sample(frac=1)

    ## Count vect
    cvect = CountVectorizer(ngram_range=ngram_range)
    X = cvect.fit_transform(df.sent.values.astype('U'))
    y = df["s_spoiler"]

    X_columns=cvect.get_feature_names()
    print("X_cols[50:60]: ",X_columns[50:60])


    print("Total size y,X:" ,y.shape[0],X.shape[0])

    ## SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    classifier = experiment(X_train, X_test, y_train, y_test, classifier_type)

    predict_unseen(cvect, classifier)



def classify_many_feats(csv_file, 
                        feat_list=[], 
                        classifier_type=LinearSVC, 
                        ngram_range=(1,1),
                        ## FOR DEVELOPMENT ONLY
                        num_rows=None):
    '''
    Classifier with count vectorizer using Metadata and tetx features.

    Classifier_type: LinearSVC, MultinomialNB, LinearRegression (?)
    Ngram_range: (1,1) unigrams, (2,2) only bigrams, (1,2) unigrams + bigrams
                -> WARNING when combining ngram_range and text feats

    Feat_list: list of target metadata or text feats, can be
    metadata: 'userID', 'bookID', 'rating', 
            'date_published','authorID', 'genre', 's_loc'
    text_feats: 'pos_bigrams', 'NE_unigrams'

    NOTE: date_published has NaN entries, does not work
    '''
    print("\nclassify_many_feats")

    if (ngram_range != (1,1) 
        and ('NE_unigrams' in feat_list or 'pos_bigrams' in feat_list)):
        raise ValueError("Text feature incompatible with ngram range")

    df = pd.read_csv(csv_file, nrows=num_rows)
    ## randomize - not needed because split_test_train randomizes
    # df = dataset.sample(frac=1)


    ### Modify data in string format: cast to float by hashing
    df["userID"] = df["userID"].map(lambda s: float(crc32(s.encode("utf-8")) & 0xffffffff) / 2**32)
    df["genre"] = df["genre"].map(lambda s: float(crc32(s.encode("utf-8")) & 0xffffffff) / 2**32)
    ## cast to unicode
    df["sent"] = df.sent.values.astype('U')


    # Create vectorizer for function to use
    cvect = CountVectorizer(ngram_range=ngram_range)

    print("FEATURE_LIST: ", feat_list)
    
    ## TEXT FEATURES
    if 'pos_bigrams' in feat_list:
        df["sent"] = df["sent"].map(lambda x: pos_bgram_feats(x))
        print("Pos bigrams: ", df["sent"][:10])
        feat_list.remove('pos_bigrams')
    elif 'NE_unigrams' in feat_list:
        df["sent"] = df["sent"].map(lambda x: NE_feats(x))
        print("NE unigrams: ", df["sent"][:10])
        feat_list.remove('NE_unigrams')

    X = sp.sparse.hstack((cvect.fit_transform(df["sent"]), df[feat_list].values),format='csr')
    # X = sp.sparse.hstack((cvect.fit_transform(df["sent"]), df[feat_list].values),format='csr')
    y = df["s_spoiler"]
    # SPLIT 
    X_columns=cvect.get_feature_names()+df[feat_list].columns.tolist()
    print("X_cols[120:130]: ",X_columns[120:130])

    print("Total size y,X:" ,y.shape[0],X.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    classifier = experiment(X_train, X_test, y_train, y_test, classifier_type)




def experiment(X_train, X_test, y_train, y_test, classifier_type):
    '''
    Runs training and predictions on classifier of type classifier_type
    '''
    
    ## CLASSIFIER
    classifier = classifier_type()

    ## TRAIN
    classifier.fit(X_train, y_train)

    ## PREDICT
    # print('TEST', X_test[:10])
    y_test_predicted = classifier.predict(X_test)

    ## RESULTS
    print(classifier, " results")
    
    '''
    Print metrics: accuracy, precision, recall, F1, AUC, ROC AUC
    '''
    print(metrics.classification_report(y_test, y_test_predicted))
    print(metrics.accuracy_score(y_test, y_test_predicted))

    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test, y_test_predicted, pos_label=1)
    pr_auc = auc(recall, precision)
    print("Precision-Recall AUC: %.2f" % pr_auc)
    # ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_test_predicted, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print("ROC AUC: %.2f" % roc_auc)

    return classifier




def predict_unseen(vectorizer, classifier):
    '''
    Predicts spoiler label for unseen sentences
    '''

    # PREDICT UNSEEN
    unseen_texts = [
        "H kills V at the end.", 
        "I never expected it to end with a big wedding...",
        "a b c d e f g",
        "ajvhsdbnmu Woooooow fs qon vjk vkuhg dkj!!!!",
        "ajvhsdbnmu fs qon vjk vkuhg dkj!!!!",
        "What a cliffhanger!",
        "How sad that A K dies.",
        "I don't believe it!!",
        "Another one for the hell of it.",
        "Really I am trying to trick you, why did it have to happen",
        "NOOOOOOOOO",
        "NO",
        "YES",
        "This isn't a spoiler.",
        "I can't believe DV was L's father!"
        ]


    unseen_feats = vectorizer.transform(unseen_texts)
    # print("mat:",  unseen_feats)

    predictions = classifier.predict(unseen_feats)

    for text, predicted in zip(unseen_texts, predictions):
        print('"{}"'.format(text))
        print("  - Predicted as: '{}'".format(predicted))
        print("")



if __name__=="__main__":

    file = 'data/balanced_fan.csv'
    print(file)
    # classify_bow_NB(file)
    # classify_bow_NB(file, get_feats=bgram_feats_NB)
    # classify_bow_NB(file, get_feats=NE_feats_NB)

    # baseline bow_counts (1,1)
    for nr in [(1,1), (1,2), (1,3)]:
        print(nr)
        classify_bow_counts(file,
                            ngram_range=nr,
                            classifier_type=MultinomialNB)

        for f in [['s_loc'], ['userID'], ['rating']]:
            classify_many_feats(file, 
                                ngram_range=nr,
                                classifier_type=MultinomialNB,
                                feat_list=f
                                )
                                
        classify_many_feats(file,
                            ngram_range=nr,
                            classifier_type=MultinomialNB,
                            feat_list=['userID', 's_loc'])

    classify_many_feats(file,
                        ngram_range=(1,1),
                        classifier_type=MultinomialNB,
                        feat_list=['pos_bigrams'])


