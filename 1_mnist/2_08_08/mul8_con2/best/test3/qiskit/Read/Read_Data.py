import sys
import numpy as np
import torch
import random
from torch.nn.functional import normalize
from Global import *

def Con_Mat(par):
  par2=torch.zeros(N_map,N_pix1,N_pix1,N_fn*N_ent,N_fn*N_ent).cdouble()
  for i in range (N_pix1):
    for j in range (N_pix1):
      par2[:,i,j,2*i*N_ent:(2*i+N_con)*N_ent,2*j*N_ent:(2*j+N_con)*N_ent]=par
  return par2.reshape(N_map*N_pix1*N_pix1,N_fn*N_ent*N_fn*N_ent)

def IMG2(img):
  edge=int((N_ent-1)/2)
  M=torch.zeros(N_fn+N_ent-1,N_fn+N_ent-1)
  M[edge:N_fn+edge,edge:N_fn+edge]=img
  N=torch.zeros(N_fn,N_fn,N_ent,N_ent)
  for i in range (N_fn):
    for j in range (N_fn):
      N[i,j,:,:]=M[i:i+N_ent,j:j+N_ent]
  img2=img.reshape(N_fn,N_fn,1,1)
  img3=N.reshape(N_fn,N_fn,N_ent,N_ent)
  img4=torch.mul(img2,img3).reshape(N_fn,N_fn,N_ent,N_ent).permute(0,2,1,3).reshape(N_fn*N_ent,N_fn*N_ent)
  del M,N,img2,img3
  return img4.cdouble()

def IMG3(img):
  M=torch.zeros(N_pix1,N_pix1,N_con*N_ent,N_con*N_ent).cdouble()
  for i in range (N_pix1):
    for j in range (N_pix1):
      M[i,j,:,:]=img[2*i*N_ent:(2*i+N_con)*N_ent,2*j*N_ent:(2*j+N_con)*N_ent]
  img2=torch.nn.functional.normalize(M.reshape(-1),p=2.0,dim=0).reshape(N_pix1*N_pix1,N_con*N_ent*N_con*N_ent).permute(1,0)
  del M
  return img2

def Read_MNIST(data):
  fr=open("../../../../../1_data/mnist/%d_%d/mnist_%s_%d_%d"%(N_fn,N_fn,data,N_fn,N_fn),"r")
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
  img2=torch.tensor(img).reshape(-1,N_fn*N_fn)
  del img
  return trg,img2

def Read_Data():
  test_tar,test_img=Read_MNIST("test")
  train_tar,train_img=Read_MNIST("train")
  return test_tar,test_img,train_tar,train_img

def Para():
  n1=N_par*N_par
  n2=N_pix1*N_pix1*10
  par1=torch.rand((N_map*n1),device=dev,dtype=Dtypec,requires_grad=True)
  par2=torch.rand((N_map*n2+1),device=dev,dtype=Dtype,requires_grad=True)
  return par1,par2,n1,n2

def Read_Para():
  n1=2*N_map*N_par*N_par
  n2=N_map*N_pix1*N_pix1*10
  par=[]
  fr=open("par","r")
  for line in fr:
    if line[:5]!="epoch":
      lx=line.split()
      for i in range (len(lx)):
        par.append(float(lx[i]))
  par1=torch.tensor(par[:n1],dtype=torch.float).reshape(N_map*N_par*N_par,2)
  par2=torch.tensor(par[n1:])
  par3=torch.complex(par1[:,0],par1[:,1]).cdouble()
  return par3,par2

def Correct(fw,pred,target):
  for i in range (len(pred)):
    fw.write("%3d %3d   "%(pred[i],target[i]))
    if i%10==9:
      fw.write("\n")
  if i%10!=9:
    fw.write("\n")
  fw.write("\n")
  fw.flush()

def OutAcc(fw,acc_tran,acc_test):
  fw.write("train: %7.2f\n"%(acc_tran))
  fw.write("pred:  %7.2f\n"%(acc_test))
  fw.flush()


