'''
Using RELU in Outer layer that is last 10 Neurons

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

-----------------

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
    from os import system
    from twilio.rest import Client

except:
    print("Library not found ")

# Create Time
now_time = datetime.datetime.now()

# Create a list to append Epoch
x_data_epoch = []

# Create a list to append Loss
y_data_error = []

# create a Training Data
train = np.empty((1000, 28, 28), dtype='float64')

# Create output Expected
trainY = np.zeros((1000,10, 1))

# Prepare Test Data
test = np.empty((10000, 28, 28), dtype='float64')

# Prepare expected Training Output
testY = np.zeros((10000, 10, 1))

# -------------Load in Image----------------

i = 0
for filename in os.listdir('/Users/soumilshah/IdeaProjects/Deep Learning/MNIST /Training1000'):
    y = int(filename[0])
    trainY[i,y] = 1.0

    train[i] = cv2.imread('/Users/soumilshah/IdeaProjects/Deep Learning/MNIST /Training1000/{0}'.format(filename), 0)/255.0
    i = i+1

# ------------LOAD TEST IMAGE -------------
i = 0
# read test data
for filename in os.listdir('/Users/soumilshah/IdeaProjects/Deep Learning/MNIST /Test10000'):
    y = int(filename[0])

    testY[i, y] = 1.0

    test[i] = cv2.imread('/Users/soumilshah/IdeaProjects/Deep Learning/MNIST /Test10000/{0}'.format(filename), 0)/255.0
    i = i+1
# ---------------------------------------

trainX = train.reshape(train.shape[0],train.shape[1]*train.shape[2], 1)
testX = test.reshape(test.shape[0], test.shape[1] * test.shape[2], 1)

# -------------- Neural Network ------------------

numNeuronsLayer1 = 100    # Number of Neuron in Layer 1
numNeuronsLayer2 = 10     # Output Neuron
numEpochs = 30            # number of Epoch
learningRate = 0.1        # define Learning Rate

# Define Weight Matrix Randomly
w1 = np.random.uniform(low=-0.1, high=0.1, size=(numNeuronsLayer1, 784))
b1 = np.random.uniform(low=-1,   high= 1,   size=(numNeuronsLayer1, 1))
w2 = np.random.uniform(low=-0.1, high=0.1, size=(numNeuronsLayer2, numNeuronsLayer1))
b2 = np.random.uniform(low=-0.1, high=0.1, size=(numNeuronsLayer2, 1))
# -------------------------------------------------------------------------------

def derivative_relu(a2):
    a2[a2 < 0] = 0
    a2[a2 >=  0] = 1
    return a2


def my_text(text):                  # create a function to execute Text to Speech
    system("say {}".format(text))


def send_sms_alert(body):

    # Define your body
    my_body= body
    # define client
    client = Client('AC437b2ebb5b00389b17e14990012090ec','4a0cb31f36c5a3beb46b537a9568cdc6')
    client.messages.create(to='+16462045957',
                           from_= '+19738603855',
                           body=my_body)

for n in range(0, numEpochs):

    loss = 0
    trainX, trainY = shuffle(trainX, trainY)

    for i in range(trainX.shape[0]):

        # S1 = W.X + B
        s1 = np.dot(w1, trainX[i]) + b1

        # A1 = 1 / 1 + EXP(-S1)
        a1 = 1 / (1 + np.exp(-1 * s1))

        # S2 = A1.W2 + B2
        s2 = np.dot(w2, a1) + b2

        # Activation function Relu
        # a2 = s2
        a2 = np.maximum(0, s2)
        loss += (0.5 * ((a2-trainY[i])*(a2-trainY[i]))).sum()   # L = 0.5.(y - a) ^^ 2

        #  ------------- BACK Progate  ------------

        # 10x1
        error = -1 * (trainY[i] - a2)
        a2_act = derivative_relu(a2)

        # Delta2 = - (Y - A2)
        delta2 = error * a2_act

        a1_act = np.multiply(a1, (1 - a1))
        error_2 = np.dot(w2.T, delta2)

        delta1 = np.multiply(error_2, a1_act)

        # GradW2 = Delta2 . A1
        gradw2 = np.dot(delta2, a1.T)

        # GradW1 = Delta1 . TrainX
        gradw1 = np.dot(delta1, trainX[i].T)

        # GRADB2 = Delta2
        gradb2 = delta2

        # GradB1 = Delta1
        gradb1 = delta1

        # W = W - learning rate . Grad
        w2 = w2 - learningRate * gradw2

        # B = B - learning rate . Grad
        b2 = b2 - learningRate * gradb2

        # W = W - learning rate . Grad
        w1 = w1 - learningRate * gradw1

        # B = B - learning rate . Grad
        b1 = b1 - learningRate * gradb1

    x_data_epoch.append(n)
    y_data_error.append(loss)

    print("epoch = " + str(n) + " loss = " + (str(loss)))

print("done training , starting testing..")
accuracyCount = 0

for i in range(testY.shape[0]):

    # S1 = W1.X + B1
    s1 = np.dot(w1, testX[i]) + b1

    # A1 = 1/ 1+ EXP(S1)
    a1 = 1/(1+np.exp(-1*s1))

    # S2 = A1.W2 + B2
    s2 = np.dot(w2,a1) + b2

    # A2 = S2 RELU
    a2 = s2

    # Select Max from 10 Neuron
    a2index = a2.argmax(axis=0)

    if testY[i, a2index] == 1:
        accuracyCount = accuracyCount + 1
        print("Accuracy count = " + str(accuracyCount/10000.0))

end = datetime.datetime.now()
t = end-now_time

# Print total Time Taken
print('time{}'.format(t))

text ='Neural Network predicted Accuracy {}'.format(accuracyCount)
my_text(text)
send_sms_alert(text)

plt.plot(x_data_epoch,y_data_error)
plt.xlabel('X axis Neuron {}'.format(numNeuronsLayer1))
plt.ylabel('Loss')
plt.title('\n stochastic gradient descent \n Execution Time:{} \n Accuracy Count {}\n Loss :{}'.format(t,accuracyCount,loss))
plt.show()
