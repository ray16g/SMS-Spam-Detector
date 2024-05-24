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

def classifyText(text):
    vocab = np.load('./data/Vocab.npy', allow_pickle=True)
    hamProb = np.load('./data/HamProbabilities.npy', allow_pickle=True)
    spamProb = np.load('./data/SpamProbabilities.npy', allow_pickle=True)

    testMessages = np.array([text])

    processedMessages = preprocess(testMessages)
    vocab = vocab[()]
    messageArray = processMessages(vocab, processedMessages)

    testHam = np.tile(hamProb, (len(messageArray), 1))
    testHam[messageArray == 0] = 1 - testHam[messageArray == 0]
    testHam = np.log(testHam).sum(axis = 1)

    testSpam = np.tile(spamProb, (len(messageArray),1))
    testSpam[messageArray == 0] = 1 - testSpam[messageArray == 0]
    testSpam = np.log(testSpam).sum(axis=1)

    classes = [None] * len(processedMessages[0])
    for i in range(len(processedMessages[0])):
        if processedMessages[0][i] in vocab:
            classes[i] = spamProb[vocab[processedMessages[0][i]]] - hamProb[vocab[processedMessages[0][i]]]
        else:
            classes[i] = 0

    return {
        'spam': (testSpam - testHam)[0],
        'text': processedMessages[0],
        'class': classes
    }

if __name__ == '__main__':
    msg = "BANK OF AMERICA: Your transaction of $5,500 will be automatically approved. To deny the transfer or report suspicious activity please click [link] to re-activate your BANK OF AMERICA account and continue using it."
    print(classifyText(msg))

#     testData = load_data(sys.argv[1])

#     vocab = np.load('./data/Vocab.npy', allow_pickle=True)
#     hamProb = np.load('./data/HamProbabilities.npy', allow_pickle=True)
#     spamProb = np.load('./data/SpamProbabilities.npy', allow_pickle=True)

#     # predictions = classify(vocab[()], hamProb, spamProb, testData[slice1:slice2][:,1])
#     # actual = testData[slice1:slice2][:,0] == 'spam'
#     print(spamProb)
#     print(hamProb)

