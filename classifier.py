import numpy as np
from build import load_data
from build import preprocess
from build import processMessages
from build import buildProbabilities

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

def computeAccuracy(predictions, actual):
    return (predictions == actual).sum() / len(predictions)

# Testing accuracy
def kFoldCrossValidation(data, k):
    accuracies = []
    step = int(len(data)/k)

    for i in range(k):
        trainFeat = np.append(data[:i*step], data[i*step+step+1:], axis = 0)
        testFeat = data[i*step:i*step+step]
        actual = data[i*step:i*step+step][:,0] == 'spam'
        buildProbabilities(trainFeat)

        vocab = np.load('./data/Vocab.npy', allow_pickle=True)
        hamProb = np.load('./data/HamProbabilities.npy', allow_pickle=True)
        spamProb = np.load('./data/SpamProbabilities.npy', allow_pickle=True)
        predictions = classify(vocab[()], hamProb, spamProb, testFeat[:,1])
        accuracies.append(computeAccuracy(predictions, actual))

    return np.array(accuracies).sum()/len(accuracies)


if __name__ == "__main__": 
    testData = load_data(sys.argv[1])

    vocab = np.load('./data/Vocab.npy', allow_pickle=True)
    hamProb = np.load('./data/HamProbabilities.npy', allow_pickle=True)
    spamProb = np.load('./data/SpamProbabilities.npy', allow_pickle=True)

    # predictions = classify(vocab[()], hamProb, spamProb, testData[slice1:slice2][:,1])
    # actual = testData[slice1:slice2][:,0] == 'spam'
    
    print(kFoldCrossValidation(testData, 10))


    
