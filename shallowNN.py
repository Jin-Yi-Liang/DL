import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def sigmoid(z):
    return 1/(1+np.exp(-z))

def draw_decision_boundary(X, Y, w1, b1, w2, b2):
    # 1. 确定绘图范围
    x1_min = X[0, :].min() - 0.5
    x1_max = X[0, :].max() + 0.5
    x2_min = X[1, :].min() - 0.5
    x2_max = X[1, :].max() + 0.5

    # 2. 在整个平面上生成大量坐标点
    xx, yy = np.meshgrid(
        np.arange(x1_min, x1_max, 0.01),
        np.arange(x2_min, x2_max, 0.01)
    )

    # 3. 整理成神经网络要求的 (2, m)
    grid_points = np.c_[
        xx.ravel(),
        yy.ravel()
    ].T

    # 4. 使用训练好的网络预测所有网格点
    _, _, _, grid_a2 = forward_propagation(
        grid_points,
        w1,
        b1,
        w2,
        b2
    )

    grid_predictions = np.where(
        grid_a2 >= 0.5,
        1,
        0
    )

    # 5. 恢复成二维网格
    grid_predictions = grid_predictions.reshape(xx.shape)

    # 6. 画模型预测区域
    plt.contourf(
        xx,
        yy,
        grid_predictions,
        alpha=0.4
    )

    # 7. 把真实样本也画上去
    plt.scatter(
        X[0, :],
        X[1, :],
        c=Y.ravel()
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

## testing accuracy
predict(X_test,Y_test, w1, b1, w2, b2, "Testing")

## draw the decision boundary
draw_decision_boundary(X_train, Y_train, w1, b1, w2, b2)