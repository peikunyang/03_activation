import sys
import numpy as np
import torch
import random
from Global import *
from torch.nn.functional import normalize

def Para():
  n1=N_par*N_par
  n2=N_pix1*N_pix1*10
  par1=torch.rand((N_map*n1),device=dev,dtype=Dtypec,requires_grad=True)
  par2=torch.rand((N_map*n2+1),device=dev,dtype=Dtype,requires_grad=True)
  return par1,par2,n1,n2

def Con_Mat(par):
  par2=torch.zeros(N_map,N_pix1,N_pix1,N_fn*N_ent,N_fn*N_ent).to(dev).cfloat()
  for i in range (N_pix1):
    for j in range (N_pix1):
      par2[:,i,j,2*i*N_ent:(2*i+N_con)*N_ent,2*j*N_ent:(2*j+N_con)*N_ent]=par
  return par2.reshape(N_map*N_pix1*N_pix1,N_fn*N_ent*N_fn*N_ent)

def IMG2(img):
  n=img.shape[0]
  edge=int((N_ent-1)/2)
  M=torch.zeros(n,N_fn+N_ent-1,N_fn+N_ent-1).to(dev)
  M[:,edge:N_fn+edge,edge:N_fn+edge]=img
  N=torch.zeros(n,N_fn,N_fn,N_ent,N_ent)
  for i in range (N_fn):
    for j in range (N_fn):
      N[:,i,j,:,:]=M[:,i:i+N_ent,j:j+N_ent]
  img2=img.reshape(n,N_fn,N_fn,1,1)
  img3=N.reshape(n,N_fn,N_fn,N_ent,N_ent).to(dev)
  img4=torch.mul(img2,img3).reshape(n,N_fn,N_fn,N_ent,N_ent).permute(0,1,3,2,4).reshape(n,N_fn*N_ent,N_fn*N_ent)
  del M,N,img2,img3
  return img4

def IMG3(img):
  n=img.shape[0]
  M=torch.zeros(n,N_pix1,N_pix1,N_con*N_ent,N_con*N_ent).to(dev)
  for i in range (N_pix1):
    for j in range (N_pix1):
      M[:,i,j,:,:]=img[:,2*i*N_ent:(2*i+N_con)*N_ent,2*j*N_ent:(2*j+N_con)*N_ent]
  img2=torch.nn.functional.normalize(M.reshape(n,-1),p=2.0,dim=1).reshape(n,N_pix1*N_pix1,N_con*N_ent*N_con*N_ent).permute(2,1,0)
  #img3=torch.nn.functional.normalize(M.reshape(n,-1),p=2.0,dim=1)
  #print(torch.square(torch.nn.functional.normalize(img3.reshape(n,-1),p=2.0,dim=1)).sum(1))
  del M
  return img2

def Read_MNIST(data):
  fr=open("../../../../1_data/fmnist/%d_%d/fmnist_%s_%d_%d"%(N_fn,N_fn,data,N_fn,N_fn),"r")
  img=[]
  trg=[]
  for line in fr:
    lx=line.split()
    if len(lx)==1:
      trg.append(int(lx[0]))
    if len(lx)>3:
      for i in range (len(lx)):
        img.append(float(lx[i]))
  fr.close()
  img2=torch.tensor(img).reshape(-1,N_fn*N_fn).to(dev)
  del img
  return trg,img2

def One_Hot(target):
  t=[]
  for i in range (len(target)):
    if target[i]==0:
      t=t+[1,0,0,0,0,0,0,0,0,0]
    if target[i]==1:
      t=t+[0,1,0,0,0,0,0,0,0,0]
    if target[i]==2:
      t=t+[0,0,1,0,0,0,0,0,0,0]
    if target[i]==3:
      t=t+[0,0,0,1,0,0,0,0,0,0]
    if target[i]==4:
      t=t+[0,0,0,0,1,0,0,0,0,0]
    if target[i]==5:
      t=t+[0,0,0,0,0,1,0,0,0,0]
    if target[i]==6:
      t=t+[0,0,0,0,0,0,1,0,0,0]
    if target[i]==7:
      t=t+[0,0,0,0,0,0,0,1,0,0]
    if target[i]==8:
      t=t+[0,0,0,0,0,0,0,0,1,0]
    if target[i]==9:
      t=t+[0,0,0,0,0,0,0,0,0,1]
  t2=torch.tensor(t).reshape(len(target),10).to(dev)
  return t2

def Read_Data():
  test_tar,test_img=Read_MNIST("test")
  test_tar_one=One_Hot(test_tar)
  train_tar,train_img=Read_MNIST("train")
  train_tar_one=One_Hot(train_tar)
  return test_tar,test_tar_one,test_img,train_tar,train_tar_one,train_img

def OutPara(n_epoch,fw,par1,par2):
  fw.write("epoch %7d %8d %8d\n"%(n_epoch,len(par1),len(par2)))
  for i in range (len(par1)):
    fw.write("%12.8f %12.8f "%(par1[i].real,par1[i].imag))
    if i%5==4:
      fw.write("\n")
  if i%5!=4:
    fw.write("\n")
  for i in range (len(par2)):
    fw.write("%12.8f "%(par2[i].real))
    if i%10==9:
      fw.write("\n")
  if i%10!=9:
    fw.write("\n")
  fw.flush()

def Correct(epoch,data,fw1,pred,target):
  count=0
  fw1.write("%-10s %6d\n"%(data,epoch))
  for i in range (len(target)):
    fw1.write("%3d %3d   "%(pred[i],target[i]))
    if i%10==9:
      fw1.write("\n")
    if pred[i]==target[i]:
      count=count+1
  if i%10!=9:
    fw1.write("\n")
  per=100*count/len(target)
  fw1.write("\n")
  fw1.flush()
  return per

def OutAcc(i,fw,acc_tran,acc_test):
  fw.write("%8d %7.2f %7.2f\n"%(i,acc_tran,acc_test))
  fw.flush()


