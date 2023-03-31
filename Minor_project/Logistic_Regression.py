import numpy as np
import pandas as pd
class Logistic_Regression():
  def __init__(self,learning_rate,no_of_iteration):
    self.learningRate = learning_rate
    self.no_of_iteration = no_of_iteration

  def fit(self,x,y):
    self.m,self.n = x.shape
    self.w = np.zeros(self.n)
    self.b =0
    self.x =x
    self.y =y
    for i in range(self.no_of_iteration):
      self.update_weight()

  def update_weight(self):
    y_hat = 1/(1+np.exp(-(self.x.dot(self.w)+self.b)))
    dw = (1/self.m)*np.dot(self.x.T,(y_hat-self.y)) 
    db = (1/self.m)*np.sum(y_hat-self.y)

    self.w -= self.learningRate*dw
    self.b -= self.learningRate*db

  def predict(self,x):
    y_pred = 1/(1+np.exp(-(self.x.dot(self.w)+self.b)))
    y_pred = np.where(y_pred>0.5,1,0)
    return y_pred