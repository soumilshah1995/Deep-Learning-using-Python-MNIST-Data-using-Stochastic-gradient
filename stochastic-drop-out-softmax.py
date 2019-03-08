'''
This Example shows Deep Dense Neural Network using  Softmax Activation
Thanking you,
Soumil Nitin Shah

Bachelor in Electronic Engineering
Master in Electrical Engineering
Master in Computer Engineering

Graduate Teaching/Research Assistant

Python Developer

soushah@my.bridgeport.edu
——————————————————
Linkedin:	https://www.linkedin.com/in/shah-soumil

Github
https://github.com/soumilshah1995

Youtube channel
https://www.youtube.com/channel/UC_eOodxvwS_H7x2uLQa-svw

------------------------------------------------------



'''
try:
    # Import library
    import os
    import sys
    import cv2
    import numpy as np
    from sklearn.utils import shuffle
    import matplotlib.pyplot as plt
    import time
    import datetime

except:
    print("Library not found ")



now_time = datetime.datetime.now()                          # Create Time

x_data_epoch = []                                           # Create a list to append Epoch
y_data_error = []                                           # Create a list to append Loss

train = np.empty((1000, 28, 28), dtype='float64')           # create a Training Data
trainY = np.zeros((1000,10, 1))                             # Create output Expected

test = np.empty((10000, 28, 28), dtype='float64')           # Prepare Test Data
testY = np.zeros((10000, 10, 1))                            # Prepare expected Training Output

# --------------------------------------------------------Load in Image--------------------------------------

i = 0
for filename in os.listdir('/Users/soumilshah/IdeaProjects/Deep Learning/MNIST /Training1000'):
    y = int(filename[0])
    trainY[i,y] = 1.0

    train[i] = cv2.imread('/Users/soumilshah/IdeaProjects/Deep Learning/MNIST /Training1000/{0}'.format(filename), 0)/255.0
    i = i+1

# -------------------------------------------------LOAD TEST IMAGE ------------------------------------------------
i = 0
# read test data
for filename in os.listdir('/Users/soumilshah/IdeaProjects/Deep Learning/MNIST /Test10000'):
    y = int(filename[0])

    testY[i, y] = 1.0

    test[i] = cv2.imread('/Users/soumilshah/IdeaProjects/Deep Learning/MNIST /Test10000/{0}'.format(filename), 0)/255.0
    i = i+1
# ---------------------------------------------------------------------------------------------------------------------

trainX = train.reshape(train.shape[0],train.shape[1]*train.shape[2], 1)
testX = test.reshape(test.shape[0], test.shape[1] * test.shape[2], 1)

# ---------------------------------  Neural Network ---------------------------------------

numNeuronsLayer1 = 200              # Number of Neuron in Layer 1
numNeuronsLayer2 = 10               # Output Neuron
numEpochs = 30                      # number of Epoch
learningRate = 0.1                 # define Learning Rate
zero_out = np.random.binomial(n=1, p=0.8, size=(numNeuronsLayer1, 1)) / 0.8


# Define Weight Matrix Randomly
w1 = np.random.uniform(low=-0.1, high=0.1, size=(numNeuronsLayer1, 784))
b1 = np.random.uniform(low=-1,   high= 1,   size=(numNeuronsLayer1, 1))
w2 = np.random.uniform(low=-0.1, high=0.1, size=(numNeuronsLayer2, numNeuronsLayer1))
b2 = np.random.uniform(low=-0.1, high=0.1, size=(numNeuronsLayer2, 1))

for n in range(0, numEpochs):

    loss = 0
    trainX, trainY = shuffle(trainX, trainY)

    for i in range(trainX.shape[0]):

        s1 = np.dot(w1,trainX[i]) + b1                          # S1 = W.X + B
        a1 = 1 / (1 + np.exp(-1 * s1))                          # A1 = 1 / 1 + EXP(-S1)
        a1 = np.multiply(a1,zero_out)

        s2 = np.dot(w2, a1) + b2                                # S2 = A1.W2 + B2
        a2 = np.exp(s2) / np.exp(s2).sum()                      # A2 = e(s2)/ e(s2).sum()

        loss = - np.sum(trainY[i] * np.log(a2))                 # Cross Entropy loss
        # L = - Y * log(A2)

        # -------------------------------------- BACK Propogate --------------------------------------

        delta2 = a2 - trainY[i]                                 # Delta A2 - Y

        a1_act = np.multiply(a1, (1 - a1))                      # A1 = A1.(1 - A1)
        a1_act = np.multiply(a1_act, zero_out)

        error_2 = np.dot(w2.T, delta2)                          # E_2 = Delta2 . W2
        delta1 = np.multiply(error_2, a1_act)                   # Delta1 = E_2 . A1

        gradw2 = np.dot(delta2, a1.T)                           # GradW2 = Delta2 . A1
        gradw1 = np.dot(delta1, trainX[i].T)                    # GradW1 = Delta1 . TrainX
        gradb2 = delta2                                         # GRADB2 = Delta2
        gradb1 = delta1                                         # GradB1 = Delta1

        w2 = w2 - learningRate * gradw2                         # W = W - learning rate . Grad
        b2 = b2 - learningRate * gradb2                         # B = B - learning rate . Grad

        w1 = w1 - learningRate * gradw1                         # W = W - learning rate . Grad
        b1 = b1 - learningRate * gradb1                         # B = B - learning rate . Grad

    x_data_epoch.append(n)
    y_data_error.append(loss)

    print("epoch = " + str(n) + " loss = " + (str(loss)))

print("done training , starting testing..")
accuracyCount = 0

# ------------------------------ Test Data -----------------------------------------------


for i in range(testY.shape[0]):

    s1 = np.dot(w1, testX[i]) + b1                      # S1 = W1.X + B1
    a1 = 1/(1+np.exp(-1*s1))                            # A1 = 1/ 1+ EXP(S1)

    s2 = np.dot(w2,a1) + b2                             # S2 = A1.W2 + B2

    a2 = np.exp(s2) / np.exp(s2).sum()                  # A2 = E(S2) / E(S2).sum()

    a2index = a2.argmax(axis=0)                         # Select Max from 10 Neuron

    if testY[i, a2index] == 1:
        accuracyCount = accuracyCount + 1
        print("Accuracy count = " + str(accuracyCount/10000.0))

end = datetime.datetime.now()
t = end-now_time
print('time{}'.format(t))                               # Print total Time Taken

plt.plot(x_data_epoch,y_data_error)
plt.xlabel('X axis Neuron {}'.format(numNeuronsLayer1))
plt.ylabel('Loss')
plt.title('\n stochastic gradient descent \n Execution Time:{} \n Accuracy Count {}\n Loss :{}'.format(t,accuracyCount,loss))
plt.show()
