import re
import string
import nltk
from nltk.corpus import stopwords
stop_words = list(stopwords.words('english'))

from nltk.tokenize import word_tokenize


'''
This module contains methods to extract features from sents
'''

def tokenizer(s):
    ## nltk tokenizer
    # return word_tokenize(s)

    t = re.split(r"(\W)", s)
    tokens = [i for i in t 
            if i not in ["\t","\n","\r","\f","\v"," ",""]
            and (i not in string.punctuation)
            # and (i not in stop_words)
            ]
    return tokens


## for classify_bow_NB
def bgram_feats_NB(sent, need_tokenize=True):
    '''
    For NaiveBayes classifier
    Returns bigrams and unigrams from sent 
    '''
    if need_tokenize:
        sent = tokenizer(str(sent))

    bigrams = list(zip(sent[:-1], sent[1:]))

    d = {'-'.join(b).lower() :1 for b in bigrams}
    
    ## add unigrams too
    for tok in sent:
        d[tok.lower()] = 1

    # print(bigrams[:10])
    return d



def bow_feats_NB(sent, need_tokenize=True):
    '''
    For NaiveBayes classifier
    Returns bow unigrams from sent
    '''
    bow = {}
    if need_tokenize:
        sent = tokenizer(str(sent))
        
    for tokens in sent:
        bow[tokens.lower()] = 1
    return bow



def NE_feats_NB(sent, need_tokenize=True):
    '''
    For NaiveBayes classifier
    Returns Named Entity (NE) unigrams from sent
    '''
    if need_tokenize:
        sent = tokenizer(str(sent))

    NNP_unigrams = {"#EMPTY#":1}
    tags = nltk.pos_tag(sent)
    # print(tags)

    if not tags: 
        #if tags is empty, none
        ## sometimes dataset has null entry
        return NNP_unigrams

    unilist, poslist = zip(*tags)
    
    for i, element in enumerate(poslist):
        if element in ["NN", "NNP"]:
            uni = unilist[i] 
            NNP_unigrams[uni.lower()] = 1   

    # print(NNP_unigrams)
    return NNP_unigrams


def NE_feats(sent, need_tokenize=True):
    '''
    For classify_bow_counts and classify_many_feats
    Returns Named Entity (NE) unigrams from sent
    '''
    return ' '.join(NE_feats_NB(sent).keys())


## for classify_many
def pos_bgram_feats(sent, need_tokenize=True, sub=True):
    '''
    For classify_bow_counts and classify_many_feats
    Text feature: Parts of Speech bigrams
    with PLACEHOLDER substitution from sent if sub==True

    eg. "The white cat walks"
    Returns: (the, white) (white, NN) (NN, walks)
    '''
    ## if tokenized already
    if not need_tokenize:
        sent = ' '.join(sent)

    placeholder_bgrams = ["#EMPTY#"]
    tags = nltk.pos_tag(word_tokenize(sent))

    # if not tags: #if tags is empty, error
    #     return 

    ugram_to_pos = dict(tags)
    # print(ugram_to_pos)

    ugrams = [ugr for (ugr, pos) in tags]
    bigrams = list(zip(ugrams[:-1], ugrams[1:]))

    ## GUO paper
    nouns = ['NN', 'NNP']
    verbs = ["VBP", "VBZ", "VBD", "VBG", "VBN"]

    for (ugram1, ugram2) in bigrams:
        ## GUO keep only pairs: JJ-nouns, nouns-verbs 
        # but hide Nouns if sub=True
        pos1 = ugram_to_pos[ugram1]
        pos2 = ugram_to_pos[ugram2]

        if pos1 in nouns and (pos2 in verbs or pos2 == 'JJ'):
            word = pos1 if sub else ugram1.lower()
            placeholder_bgrams.append('-'.join([word, ugram2.lower()]))
        elif pos2 in nouns and (pos1 in verbs or pos1 == 'JJ'):
            word = pos2 if sub else ugram2.lower()
            placeholder_bgrams.append('-'.join([ugram1.lower(), word]))

    return ' '.join(placeholder_bgrams)