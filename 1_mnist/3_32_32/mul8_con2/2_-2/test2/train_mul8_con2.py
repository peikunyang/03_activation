import os,sys
import torch
import numpy as np
from torch import optim
from Global import *
from Read.Read_Data import *
import datetime

def Train_Pyt(one,img):
  sam=one.shape[0]
  n_batch=int(np.ceil(sam/Batch))
  num=np.arange(sam)
  random.shuffle(num)
  batch=Batch
  for i in range (n_batch):
    beg_frm=i*Batch
    end_frm=(i+1)*Batch
    if end_frm>sam:
      end_frm=sam
      batch=end_frm-beg_frm
    one_b=one[num[beg_frm:end_frm]]
    imgb1=img[num[beg_frm:end_frm]].reshape(-1,N_fn,N_fn)
    imgb=IMG3(IMG2(imgb1)).cfloat().reshape(N_con*N_ent*N_con*N_ent,N_pix1*N_pix1*batch)
    pred0=torch.matmul(Con_Unitary(Par1),imgb).reshape(N_map,N_pix1*N_pix1,batch)
    pred1=torch.mul(pred0,torch.conj(pred0)).reshape(N_map*N_pix1*N_pix1,batch).float()
    err=Par2[-1:]*torch.matmul(Par2[:-1].reshape(10,N_map*N_pix1*N_pix1),pred1).permute(1,0)-one_b
    loss=torch.square(err).sum()
    Opt.zero_grad()
    loss.backward()
    Opt.step()

def Pred_Pyt(img,par1,par2):
  pdig=[]
  sam=img.shape[0]
  n_batch=int(np.ceil(sam/Batch))
  num=np.arange(sam)
  batch=Batch
  cm=Con_Unitary(par1)
  for i in range (n_batch):
    beg_frm=i*Batch
    end_frm=(i+1)*Batch
    if end_frm>sam:
      end_frm=sam
      batch=end_frm-beg_frm
    imgb1=img[num[beg_frm:end_frm]].reshape(-1,N_fn,N_fn)
    imgb=IMG3(IMG2(imgb1)).cfloat().reshape(N_con*N_ent*N_con*N_ent,N_pix1*N_pix1*batch)
    pred0=torch.matmul(cm,imgb).reshape(N_map,N_pix1*N_pix1,batch)
    pred1=torch.mul(pred0,torch.conj(pred0)).reshape(N_map*N_pix1*N_pix1,batch).float()
    pred2=par2[-1:]*torch.matmul(par2[:-1].reshape(10,N_map*N_pix1*N_pix1),pred1).permute(1,0)
    dig=torch.argmax(pred2,dim=1).to("cpu").tolist()
    pdig=pdig+dig
  del cm
  return pdig

def Con_Unitary(par):
  U,S,VT=torch.linalg.svd(par.reshape(N_map,N_par*N_par))
  Q=torch.matmul(U,VT[:N_map,:]).to(dev)
  del U,S,VT
  return Q

def Main():
  global Par1,Par2,N1,N2,Opt
  Par1,Par2,N1,N2=Para()
  test_tar,test_one,test_img,train_tar,train_one,train_img=Read_Data()
  Opt=optim.SGD([Par1,Par2],lr=learning_rate,momentum=0.9)
  fw1=open("accu_case_ori","w")
  fw3=open("accu_sum","w")
  fw4=open("par","w")
  fw3.write("   epoch   train    test\n")
  i=0
  while (i<N_Ite):
    Train_Pyt(train_one,train_img)
    i=i+1
    if i%10==0:
      pdig_train=Pred_Pyt(train_img,Par1.detach(),Par2.detach())
      pdig_test=Pred_Pyt(test_img,Par1.detach(),Par2.detach())
      acc_tran=Correct(i,"train",fw1,pdig_train,train_tar)
      acc_test=Correct(i,"test",fw1,pdig_test,test_tar)
      OutAcc(i,fw3,acc_tran,acc_test)
      OutPara(i,fw4,Par1.detach(),Par2.detach())
  fw1.close()
  fw3.close()
  fw4.close()

start=datetime.datetime.now()
Main()
end=datetime.datetime.now()
print("執行時間：",end-start)

