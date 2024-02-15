'''
    SMS Spam Detector using Naive Bayes Classifier 
    https://github.com/ray16g
'''

import numpy as np
import nltk
import string

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

data = load_data("./data/SMSSpamCollection")
spamLabels = data[:,0]
smsMessages = data[:,1]

# smsMessages is a list of words
smsMessages = preprocess(smsMessages)

    

print(smsMessages)