import numpy as np
from build import load_data
from build import preprocess
from build import processMessages

import sys

def classify(vocab, hamProb, spamProb, testData):
    testMessages = preprocess(testData)
    testMessages = processMessages(vocab, testMessages)

    testHam = np.tile(hamProb, (len(testMessages), 1))
    testHam[testMessages == 0] = 1 - testHam[testMessages == 0]
    testHam = np.log(testHam).sum(axis = 1)

    testSpam = np.tile(spamProb, (len(testMessages),1))
    testSpam[testMessages == 0] = 1 - testSpam[testMessages == 0]
    testSpam = np.log(testSpam).sum(axis=1)

    return testSpam > testHam

if __name__ == "__main__": 
    testData = load_data(sys.argv[1])

    vocab = np.load('./data/Vocab.npy', allow_pickle=True)
    hamProb = np.load('./data/HamProbabilities.npy', allow_pickle=True)
    spamProb = np.load('./data/SpamProbabilities.npy', allow_pickle=True)

    slice1 = 0
    slice2 = len(testData)-1
    if(len(sys.argv) == 4):
        slice1 = int(sys.argv[2])
        slice2 = int(sys.argv[3])

    predictions = classify(vocab[()], hamProb, spamProb, testData[slice1:slice2][:,1])
    actual = testData[slice1:slice2][:,0] == 'spam'
    
    print((predictions == actual).sum() / len(predictions))
    print((predictions == actual).sum(),len(predictions))


    
