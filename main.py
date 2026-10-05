import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def sigmoid(z):
    return 1/(1+np.exp(-z))

def drawDecisionBoundary(X_train, w1, b1, w2, b2):
    x1_min = X_train[0, :].min() - 0.5
    x1_max = X_train[0, :].max() + 0.5
    x2_min = X_train[1, :].min() - 0.5
    x2_max = X_train[1, :].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x1_min, x1_max, 0.01),
        np.arange(x2_min, x2_max, 0.01)
    )

    grid_points = np.c_[
        xx.ravel(),
        yy.ravel()
    ].T

    grid_Z1 = w1 @ grid_points + b1
    grid_A1 = np.tanh(grid_Z1)

    grid_Z2 = w2 @ grid_A1 + b2
    grid_A2 = sigmoid(grid_Z2)

    grid_predictions = np.where(
        grid_A2 >= 0.5,
        1,
        0
    )

    grid_predictions = grid_predictions.reshape(xx.shape)

    plt.contourf(
        xx,
        yy,
        grid_predictions,
        alpha=0.4
    )


    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision Boundary")
    plt.show()

def forward_propagation(X, w1, b1, w2, b2):
    z1=w1@X+b1
    a1=np.tanh(z1)
    z2=w2@a1+b2
    a2=sigmoid(z2)
    return z1,a1,z2,a2

def predict(X,Y,w1,b1,w2,b2,word):
    _,_,_,a2=forward_propagation(X, w1, b1, w2, b2)
    predictions=np.where(a2>0.5,1,0)
    accuracy = np.mean(predictions == Y)
    print(word,"Accuracy:", accuracy)


X,y=make_moons(n_samples=500, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train=X_train.T
Y_train=y_train.reshape(1,-1)
X_test=X_test.T
Y_test=y_test.reshape(1,-1)

n_x=2
n_h=4
n_y=1

np.random.seed(42)
w1=np.random.randn(n_h,n_x)*0.01
b1=np.zeros((n_h,1))
w2=np.random.randn(n_y,n_h)*0.01
b2=np.zeros((n_y,1))

learning_rate=1.2
num_iterations=10000

for i in range(num_iterations):
    ## forward propagation
    z1,a1,z2,a2=forward_propagation(X_train, w1, b1, w2, b2)

    ## cost
    m=Y_train.shape[1]
    cost=-1/m*np.sum(Y_train*np.log(a2)+(1-Y_train)*np.log(1-a2))

    ## back propagation 
    dz2=a2-Y_train
    dw2=(1/m)*(dz2@a1.T)
    db2=(1/m)*np.sum(dz2,axis=1,keepdims=True)
    da1=w2.T@dz2
    dz1=da1*(1-np.power(a1,2))
    dw1=(1/m)*(dz1@X_train.T)
    db1=(1/m)*np.sum(dz1,axis=1,keepdims=True)

    ## update parameters
    w1=w1-learning_rate*dw1
    b1=b1-learning_rate*db1
    w2=w2-learning_rate*dw2
    b2=b2-learning_rate*db2

    if(i%1000==0):
        print("Iteration:", i, "Cost:", cost)

## training accuracy
predict(X_train, Y_train, w1, b1, w2, b2, "Training")

## draw the decision boundary
## drawDecisionBoundary(X_train, w1, b1, w2, b2)

## testing accuracy
predict(X_test,Y_test, w1, b1, w2, b2, "Testing")
