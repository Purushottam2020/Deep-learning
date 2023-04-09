import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
# print(iris)

idxs = np.where(iris.target<2)
x= iris.data[idxs]
y= iris.target[idxs]
# print(y)
# print(x[y==0][:,0])
# print(x[y==0])

plt.scatter(x[y==0][:,0],x[y==0][:,2],color='black',label='Iris-Setosa')
plt.scatter(x[y==1][:,0],x[y==1][:,2],color='red',label='Iris-Versicolor')
plt.title('Iris Plants Database 9920004518')
plt.xlabel('Sepal Length in cm')
plt.ylabel('Petal Length in cm')
plt.legend()
plt.show()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2000)
print("x train",x_train)
# print(x_test)
# print(y_train)
# print(y_test)

weights=np.random.normal(size=x_train.shape[1])
bias=1

# print(weights)

learning_rate=0.5
n_epochs=100
# print(np.zeros(weights.shape))
# del_w=np.zeros(weights.shape)
hist_loss=[]
hist_accuracy=[]


for i in range(n_epochs):
    output1=np.where(1/(1+np.exp(-(x_train.dot(weights)+bias)))>0.9,1,0)
    output=1/(1+np.exp(-(x_train.dot(weights)+bias)))
    print("train")
    print(output)
    
    error=np.mean((y_train-output)**2)
    print("Error",error)
    
    weights=learning_rate * np.dot((output-y_train),x_train)
    print("weights",weights)
    
    loss = np.mean((output-y_train)**2)
    hist_loss.append(loss)
    output_val=1/(1+np.exp(-(x_train.dot(weights)+bias)))
    output_val1= np.where(1/(1+np.exp(-(x_train.dot(weights)+bias)))>0.9,1,0)
    accuracy = np.mean(np.where(y_test==output_val,1,0))
    
    hist_accuracy.append(accuracy)
    
fig= plt.figure(figsize=(8,4))
a = fig.add_subplot(1,2,1)
imgplot=plt.plot(hist_loss)
plt.xlabel('epochs')
a.set_title('training set,9920004518')

a=fig.add_subplot(1,2,2)
imgplot=plt.plot(hist_accuracy)
a.set_title('validation accuracy')
plt.show()



