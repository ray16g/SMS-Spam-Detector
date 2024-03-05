'''
    SMS Spam Detector using Bernoulli Naive Bayes Classifier 
    https://github.com/ray16g
'''

import numpy as np
import nltk
import string
import sys

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')

# Load data from given path into an array
def load_data(path):
    return np.genfromtxt(path, dtype='str', delimiter = "\t") 

# Preprocessing steps:
# Lowercase the string
# Remove punctuation/symbols
# Tokenization
# Remove stop words
# Lemmatization
    
def preprocess(messages):
    
    messages = np.char.lower(messages)

    remove = str.maketrans('', '', string.punctuation)
    messages = np.array([x.translate(remove) for x in messages])
    
    messages = np.char.split(messages, " ")
    
    filterWords = stopwords.words("english")
    for x in range(len(messages)):
        messages[x] = list(filter(lambda var: not (var in filterWords or var == ''), messages[x]))
    
    lemmatizer = WordNetLemmatizer()
    for x in range(len(messages)):
        messages[x] = [lemmatizer.lemmatize(word) for word in messages[x]]
    
    return messages


def createVocabDict(messages):
    # Return a dict where [word = index]
    vocab = set(messages.sum())
    return dict([(v,i) for i, v in enumerate(vocab)])

def processMessages(vocab, messages):
    # Return 2d matrix where newMessages[x] = array where each word is marked true or false depending on if it appears
    newMessages = np.array([[False]*len(vocab)]*len(messages))
 
    for x in range(len(messages)):
        for y in messages[x]:
            newMessages[x][vocab[y]] = True
    
    return newMessages


def buildProbabilities(data):
    trainLabels = data[:,0] == 'spam'
    rawTrainMessages = data[:,1]

    # smsMessages is an array of list of words
    trainMessages = preprocess(rawTrainMessages)
    vocabDict = createVocabDict(trainMessages)

    trainMessages = processMessages(vocabDict, trainMessages)

    hamSum = trainMessages[trainLabels == 0].sum(axis=0)
    hamProb = (hamSum + 1) / (np.sum(trainLabels == 0) + 2) # Laplace Smoothing (m+1/n+2)

    spamSum = trainMessages[trainLabels == 1].sum(axis=0)
    spamProb = (hamSum + 1) / (np.sum(trainLabels == 1) + 2)
    
    # Build
    np.save('./data/Vocab', vocabDict)
    np.save('./data/HamProbabilities', hamProb)
    np.save('./data/SpamProbabilities', spamProb)

if __name__ == "__main__":
    data = load_data(sys.argv[1])

    if(len(sys.argv) == 3):
        buildProbabilities(data[sys.argv[2]:sys.argv[3]])
    else:
        buildProbabilities(data)
    

