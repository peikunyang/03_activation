import os,sys
import numpy as np
import torch

def Correct(pred):
  diff=pred[:,0]-pred[:,1]
  err=torch.count_nonzero(diff).numpy()
  total=diff.shape[0]
  per=1-err/total
  return total,err,100*per

def Read_data(train):
  fn=os.listdir("%s"%(train))
  pred=[]
  for i in range (len(fn)):
    fr=open("%s/%s"%(train,fn[i]),"r")
    for line in fr:
      lx=line.split()
      for i in range (len(lx)):
        pred.append(int(lx[i]))
  fr.close()
  pred2=torch.tensor(pred).reshape(-1,2)
  return pred2

def Main():
  fw=open("qiskit_accuracy","w")
  train_pred=Read_data("train")
  test_pred=Read_data("test")
  total_train,err_train,per_train=Correct(train_pred)
  total_test,err_test,per_test=Correct(test_pred)
  fw.write("Train %6d %6d %6.2f\n"%(total_train,err_train,per_train))
  fw.write("Test  %6d %6d %6.2f\n"%(total_test,err_test,per_test))
  fw.close()

Main()

