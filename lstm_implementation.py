#Implementation by: Guilherme Bailoni

#Based on the article Backpropogating an LSTM: A Numerical Example by: Aidan Gomez

import numpy as np

def sigmoid(z):
    return 1/(1 + np.e**-z)

def der_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def sech(z):
    return 1/np.cosh(z)

def der_tanh(z):
    return 1 - np.tanh(z)**2

def output_neural_network(Wy,ht):
    return np.matmul(Wy,ht)

def l2(yt,ot):
    return ((ot-yt)**2)/2

def lossOvertime(ys,os,t):
    totalLoss = 0
    for i in range(0,t):
        totalLoss += l2(ys[i],os[i])
    return totalLoss/t

#receives input, past h and past c, Weight matrixes
def lstm_timestep(xt,ph,pc,W,U,b):
    # print("Entrada LSTM PASS:")
    # print(xt)
    # print(ph)
    # print(pc)
    # print("Fim Entrada")
    at = np.tanh(np.matmul(W[0],xt) + np.matmul(U[0],ph) + b[0])
    it = sigmoid(np.matmul(W[1],xt) + np.matmul(U[1],ph) + b[1])
    ft = sigmoid(np.matmul(W[2],xt) + np.matmul(U[2],ph) + b[2])    
    ot = sigmoid(np.matmul(W[3],xt) + np.matmul(U[3],ph) + b[3])
    ct = np.multiply(at,it) + np.multiply(ft,pc)
    ht = np.multiply(np.tanh(ct),ot)
    return ht, ct, at, it, ft, ot

def lstm(train_bool, xs, ys, t, n, W, U, b):
    #1 - State vectors

    #hidden states throught time
    hiddenStates = [[0] for i in range(t+1)]
    #cell states throught time
    cellStates = [[0] for i in range(t+1)]
    aCells = [0] * (t+1)
    inputCells = [0] * (t+1)
    forgetCells = [0] * (t+1)
    outputCells = [0] * (t+1)

    #3 - Gradients
    DeltaHiddenStates = [0] * (t+1)
    dHiddenStates = [0] *  t
    dCellStates = [0] * (t+1)
    dACells = [0] * t
    dInputCells = [0] * t
    dForgetCells = [0] * t
    dOutputCells = [0] * t
    
    #forward propagation
    for i in range(0,t):
        result = lstm_timestep(xs[i],hiddenStates[i],cellStates[i],W,U,b)
        print("RESULTs after LSTM timestep:")
        print(result)
        hiddenStates[i+1] = result[0]
        cellStates[i+1] = result[1]
        aCells[i] = result[2]
        inputCells[i] = result[3]
        forgetCells[i] = result[4]
        outputCells[i] = result[5]

    #backward propagation
    if train_bool == True:
        for i in reversed(range(0,t)):
            dHiddenStates[i] = (hiddenStates[i+1]-ys[i]) + DeltaHiddenStates[i+1]
            #dCellStates[i] = np.dot(dHiddenStates[i],outputCells[i],der_tanh(hiddenStates[i])) + np.dot(dCellStates[i+1],forgetCells[i+1])
            dCellStates[i] = np.multiply(np.multiply(dHiddenStates[i],outputCells[i]),der_tanh(cellStates[i+1])) + np.multiply(dCellStates[i+1],forgetCells[i+1])
            dACells[i] = np.multiply(np.multiply(dCellStates[i],inputCells[i]),(1 - aCells[i]**2))
            dInputCells[i] = np.multiply(np.multiply(np.multiply(dCellStates[i],aCells[i]),inputCells[i]),(1 - inputCells[i]))
            dForgetCells[i] = np.multiply(np.multiply(np.multiply(dCellStates[i],cellStates[i]),forgetCells[i]),(1 - forgetCells[i]))
            dOutputCells[i] = np.multiply(np.multiply(np.multiply(dHiddenStates[i],np.tanh(cellStates[i+1])),outputCells[i]),(1 - outputCells[i]))
            DeltaHiddenStates[i] = np.matmul(np.transpose(U),[dACells[i],dInputCells[i],dForgetCells[i],dOutputCells[i]])[0]
            print("Debug BackPropagation at timestep: " + str(i))
            print(dHiddenStates)
            print(dCellStates)
            print(dACells)
            print(dInputCells)
            print(dForgetCells)
            print(dOutputCells)
            print(DeltaHiddenStates)
            print("End of Backpropagation Debug")

    dW = [[0 for i in range(len(xs[0]))] for j in range (4)]
    for i in range(0,t):
        # print("debug dGates e entrada no timestep" + str(i+1))
        # print(xs[i])
        # print([dACells[i],dInputCells[i],dForgetCells[i],dOutputCells[i]])
        # print("fim debug")
        dW += np.outer([dACells[i],dInputCells[i],dForgetCells[i],dOutputCells[i]],xs[i])

    dU = [[0 for i in range(len(hiddenStates[0]))] for j in range (4)]
    for i in range(1,t):
        dU += np.outer([dACells[i],dInputCells[i],dForgetCells[i],dOutputCells[i]],[hiddenStates[i]])

    db = [[0 for i in range(len(b[0]))] for j in range (4)]
    for i in range(0,t):
        db = np.add(db,[dACells[i],dInputCells[i],dForgetCells[i],dOutputCells[i]])
    
    print("DEBUG dWeights and dBias")
    print(dW)
    print(dU)
    print(db)
    for i in range(0,4):
        W[i] -= np.multiply(n,dW[i])
        U[i] -= np.multiply(n,dU[i])
        b[i] -= np.multiply(n,db[i])
    
    return W, U, b, hiddenStates[1:t+1]

#0 - hyper parameters

#timesteps
t = 2
#batch size
batch = 1
#learning rate
n = 0.1

#input vector
xs = np.array([[1,2],[0.5,3],[0.8,1.5],[0.9,0.4],[1,1.4],[0.1,0.9],[0.3,0.5]])
print("Input Vector")
print(xs)

#output vector
ys = np.array([[0.5],[1.25],[1],[0.8],[0.3],[0.7],[1.2]])
print("Output Vector")
print(ys)

outputs = [0] * t

#Weight Matrixes

#input weight matrixes
W = [[0.45,0.25],[0.95,0.8],[0.7,0.45],[0.6,0.4]]
#hidden state weight matrixes
U = [[0.15],[0.8],[0.1],[0.25]]
#biasis initialization
b = [[0.2],[0.65],[0.15],[0.1]]

W, U, b, outputs = lstm(True, xs, ys, t, n, W, U, b)

print("Outputs")
print(outputs)
print("Expected outputs")
print(ys[0:t])

print("Input Weight matrixes")
print(W)
print("Hidden state Weight matrixes")
print(U)
print("Bias")
print(b)

print("Total Error:")
print(list(map(l2,ys[0:t],outputs)))
input()

#training the network
# for i in range(500):
#     W, U, b, outputs = lstm(True, xs, ys, t, n, W, U, b)
# input()