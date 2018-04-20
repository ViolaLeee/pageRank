from numpy import *

# shape of the network
network = array([[0,0,0,0,1],
                 [1,0,0,0,0],
                 [1,0,0,0,0],
                 [1,1,0,0,0],
                 [0,1,1,1,0]],dtype = float)

def pageMatrix(network):
    Matrix = zeros((network.shape), dtype = float)
    [rows,cols] = network.shape
    for i in range(rows):
        for j in range(cols):
            Matrix[i][j] = network[i][j]/(network.sum(axis = 0)[j])
    # print (Matrix)
    return Matrix

def initProb(Matrix):
    prob = zeros((Matrix.shape[0],1), dtype = float)
    for i in range(Matrix.shape[0]):
        prob[i] = float(1)/Matrix.shape[0]
    # print (prob)
    return prob

def pageRank(beta, m, v):
    while((v == beta*dot(m, v) + (1-beta)*(1/network.shape[0])).all() == False):
        v = beta*dot(m, v) + (1-beta)*(1/network.shape[0])
    print (v)
    return v

if __name__=="__main__":
    M = pageMatrix(network)
    prob = initProb(M)
    beta = 0.8
    pageRank(beta, M, prob)