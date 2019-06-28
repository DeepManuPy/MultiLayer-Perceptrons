import numpy as np

"""
Neural Netwrok Architecture:
input => 25-neurons => 50-neurons => 50-neurons => 25-neurons => 1-neuron + sigmoid (output)
"""
NN_ARCHITECTURE = [
    {"input_dim":2,"output_dim":25,"activation":"relu"},
    {"input_dim":25,"output_dim":50,"activation":"relu"},
    {"input_dim":50,"output_dim":50,"activation":"relu"},
    {"input_dim":50,"output_dim":25,"activation":"relu"},
    {"input_dim":25,"output_dim":1,"activation":"sigmoid"}
]
# initiating the weights and biases if the hidden units in the Network
def initialize_layers(nn_arch,seed=189):
    # random seed initiation
    np.random.seed(seed)
    num_layers = len(nn_arch)
    params_values = {}
    # iterate over Netwrok layers
    for idx,layer in enumerate(nn_arch):
        layer_id = idx+1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        # initiating the values for W matrix and b vector
        params_values["W"+str(layer_id)] = np.random.random((layer_output_size,layer_input_size)) * 0.1
        params_values["b"+str(layer_id)] = np.random.random((layer_output_size,1))*0.1

    return params_values
# Defining Activation Functions used in the Network
def sigmoid(z):
    return 1/(1+np.exp(-z))
def relu(z):
    return np.maximum(0,z)
def sigmoid_backprop(dA,z):
    sig = sigmoid(z)
    return dA*sig*(1-sig)
def relu_backprop(dA,z):
    dz = np.array(dA,copy=True)
    dz[z<=0] = 0
    return dz
# Implementing single step forward propagation
def single_forward_prop(A_prev,W_curr,b_curr,activation="relu"):
    Z_curr = np.dot(W_curr,A_prev) + b_curr
    if activation is "relu":
        activation_func = relu
    elif activation is "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception("Nope! Not Gonna Use That!")
    return activation_func(Z_curr),Z_curr

def forward_prop(X,params_values,nn_arch):
    cache = {}
    A_curr = X

    for idx,layer in  enumerate(nn_arch):
        layer_id = idx+1

        A_prev = A_curr

        activ_function_curr = layer["activation"]
        W_curr = params_values["W"+str(layer_id)]
        b_curr = params_values["b"+str(layer_id)]
        A_curr,Z_curr = single_forward_prop(A_prev,W_curr,b_curr,activ_function_curr)

        cache["A"+str(idx)] = A_prev
        cache["Z"+str(layer_id)] = Z_curr
    return A_curr,cache
def get_cost_value(Y_hat,Y):
    m = Y_hat.shape[1]
    cost = -1/m * ((np.dot(Y,np.log(Y_hat).T)) + np.dot(1-Y,np.log(1-Y_hat).T))
    return np.squeeze(cost)
def convert_prob_to_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_>0.5] = 1
    probs_[probs_<=0.5] = 0
    return probs_
def get_accuracy(Y_hat,Y):
    Y_hat = convert_prob_to_class(Y_hat)
    return (Y_hat == Y).all(axis=0).mean()

def single_backward_prop(dA_curr,W_curr,b_curr,Z_curr,A_prev,activation="relu"):
    m = A_prev.shape[1]
    if activation is "relu":
        activation_func = relu_backprop
    elif activation is "sigmoid":
        activation_func = sigmoid_backprop
    else:
        raise Exception("Nope! Not Gonna Use That!")
    dZ_curr = activation_func(dA_curr,Z_curr)

    dW_curr = np.dot(dZ_curr,A_prev.T)/m
    db_curr = np.sum(dZ_curr,axis=1,keepdims=True)/m
    dA_prev = np.dot(W_curr.T,dZ_curr)

    return dA_prev,dW_curr,db_curr

def backwards_prop(Y_hat,Y,cache,params_values,nn_arch):
    gradient_values = {}

    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)

    dA_prev = -(np.divide(Y,Y_hat) - np.divide(1-Y,1-Y_hat))

    for layer_id_prev,layer in reversed(list(enumerate(nn_arch))):
        layer_id = layer_id_prev + 1
        activ_function_curr = layer["activation"]

        dA_curr = dA_prev

        A_prev = cache["A"+str(layer_id_prev)]
        Z_curr = cache["Z"+str(layer_id)]

        W_curr = params_values["W"+str(layer_id)]
        b_curr = params_values["b"+str(layer_id)]

        dA_prev,dW_curr,db_curr = single_backward_prop(dA_curr,W_curr,b_curr,Z_curr,A_prev,activ_function_curr)

        gradient_values["dW"+str(layer_id)] = dW_curr
        gradient_values["db"+str(layer_id)] = db_curr
    return gradient_values

def update(params_values,gradient_values,nn_arch,learning_rate):
    for layer_id,layer in enumerate(nn_arch,1):
        params_values["W"+str(layer_id)] -= learning_rate*gradient_values["dW"+str(layer_id)]
        params_values["b"+str(layer_id)] -= learning_rate*gradient_values["db"+str(layer_id)]
    return params_values
# Training the Neural Network
def train(X,Y,nn_arch,epochs,learning_rate,verbose=False,callback=None):
    params_values = initialize_layers(nn_arch)
    loss_history = []
    accuracy_history = []

    for epoch in range(epochs):
        Y_hat,cache = forward_prop(X,params_values,nn_arch)
        loss = get_cost_value(Y_hat,Y)
        loss_history.append(loss)
        acc = get_accuracy(Y_hat,Y)
        accuracy_history.append(acc)

        gradient_values = backwards_prop(Y_hat,Y,cache,params_values,nn_arch)
        # updating model state
        params_values = update(params_values,gradient_values,nn_arch,learning_rate)
        if epoch%50==0 :
            if(verbose):
                print("Epoch:{:05} - loss:{:.3f} - acc:{:.4f}".format(epoch,loss,acc))
            if(callback is not None):
                callback(i, params_values)
    return params_values
# Let's Create an artificial Dataset to train and test our model
import os
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
sns.set_style("whitegrid")
from sklearn.metrics import accuracy_score

# number of samples in the data set
N_SAMPLES = 1000
# ratio between training and test sets
TEST_SIZE = 0.1
# create an artificial Dataset
X,y = make_moons(n_samples=N_SAMPLES,noise=0.2,random_state=100)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=TEST_SIZE,random_state=45)
print("[INFO]printing shape of features....")
print(X.shape)
print("[INFO]printing shape of the labels....")
print(y.shape)

def make_plot(X, y, plot_name, file_name=None, XX=None, YY=None, preds=None, dark=False):
    if (dark):
        plt.style.use('dark_background')
    else:
        sns.set_style("whitegrid")
    plt.figure(figsize=(16,12))
    axes = plt.gca()
    axes.set(xlabel="$X_1$", ylabel="$X_2$")
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    if(XX is not None and YY is not None and preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha = 1, cmap=cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], cmap="Greys", vmin=0, vmax=.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors='black')
    if(file_name):
        plt.savefig(file_name)
        plt.close()
print("[INFO]Plotting graph to Visualize Dataset....")
make_plot(X,y,"Dataset","Dataset.png")
#training
print("[INFO]Training the Neural Network on train data.....")
params_values=train(np.transpose(X_train),np.transpose(y_train.reshape((y_train.shape[0],1))),NN_ARCHITECTURE,10000,0.01,verbose=True)
print("[INFO]Training Successfully Finished.")
#prediction
print("[INFO]Testing the trained model on test data....")
y_test_hat,_ = forward_prop(np.transpose(X_test),params_values,NN_ARCHITECTURE)
#Accuracy achieved on the test set
test_acc = get_accuracy(y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
print("Test set accuracy: {:.2f}".format(test_acc))
