import os
import torch
import numpy as np
import datetime
from qiskit import *
from qiskit.quantum_info import Statevector
from qiskit.execute_function import execute
from qiskit.extensions import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit,transpile
from Read.Read_Data import *
from Global import *

N_Shot=100000 #100000

def Plot_Circuit(par1):
  fw=open("circuit.text","w")
  qc=QuantumCircuit(qubit)
  qc.unitary(par1,[0,1,2,3,4,5,6,7],label='unitary')
  qc_gate=transpile(qc,basis_gates=['cx','rx','ry','rz'])
  #qc_gate.draw(output="mpl",filename="qc")
  fw.write("%120s"%(qc_gate))
  fw.write("\n")
  fw.close()

def Convert_Pro(counts):
  pro=torch.zeros(256)
  for key,count in counts.items():
    dig=int(key,2)
    pro[dig]=count/N_Shot
  return pro

def Pred_prob(par1,par2,img,epo1,epo2):
  pdig=[]
  cm=Con_Unitary(par1)
  cm2=torch.zeros(256,256).cdouble()
  for i in range (16):
    cm2[16*i:16*(i+1),16*i:16*(i+1)]=cm
  for i in range (epo1,epo2,1):
    qc=QuantumCircuit(qubit)
    imgb=IMG3(IMG2(img[i].reshape(N_fn,N_fn))).permute(1,0).reshape(-1)
    imgb2=torch.zeros(256).cdouble()
    imgb2[:256]=imgb
    img2=torch.nn.functional.normalize(imgb2.reshape(-1),p=2.0,dim=0)
    qc.initialize(img2.tolist(),qc.qubits)
    qc.unitary(cm2,[0,1,2,3,4,5,6,7],label='unitary')
    job=execute(qc,backend)
    result=job.result()
    outputstate=result.get_statevector()
    probs=Statevector(outputstate).probabilities()
    pred1=torch.tensor(probs).float().reshape(16,16)[:16,:N_map].permute(1,0).reshape(-1)
    pred2=par2[-1:]*torch.matmul(par2[:-1].reshape(10,N_map*N_pix1*N_pix1),pred1)
    dig=torch.argmax(pred2,dim=0).tolist()
    pdig.append(dig)
  return pdig

def Pred_shot(par1,par2,img,epo1,epo2):
  simulator=AerSimulator()
  pdig=[]
  cm=Con_Unitary(par1)
  cm2=torch.zeros(256,256).cdouble()
  for i in range (16):
    cm2[16*i:16*(i+1),16*i:16*(i+1)]=cm
  for i in range (epo1,epo2,1):
    qc=QuantumCircuit(qubit)
    imgb=IMG3(IMG2(img[i].reshape(N_fn,N_fn))).permute(1,0).reshape(-1)
    imgb2=torch.zeros(256).cdouble()
    imgb2[:256]=imgb
    img2=torch.nn.functional.normalize(imgb2.reshape(-1),p=2.0,dim=0)
    qc.initialize(img2.tolist(),qc.qubits)
    qc.unitary(cm2,[0,1,2,3,4,5,6,7],label='unitary')
    qc.measure_all()
    compiled_circuit=transpile(qc,simulator)
    job=simulator.run(compiled_circuit,shots=N_Shot)
    result=job.result()
    counts=result.get_counts(compiled_circuit)
    pred1=Convert_Pro(counts).reshape(16,16)[:16,:N_map].permute(1,0).reshape(-1)
    pred2=par2[-1:]*torch.matmul(par2[:-1].reshape(10,N_map*N_pix1*N_pix1),pred1)
    dig=torch.argmax(pred2,dim=0).tolist()
    pdig.append(dig)
  return pdig

def Qiskit_pro(par1,par2,train_img,test_img,train_tar,test_tar):
  global backend
  backend=BasicAer.get_backend('statevector_simulator')
  fw=open("pred_prob/pred_%s_%03d"%(sys.argv[1],int(sys.argv[2])),"w")
  if sys.argv[1]=="tran":
    pred=Pred_prob(par1,par2,train_img,int(sys.argv[3]),int(sys.argv[4]))
    Correct(fw,pred,train_tar[int(sys.argv[3]):int(sys.argv[4])])
  if sys.argv[1]=="test":
    pred=Pred_prob(par1,par2,test_img,int(sys.argv[3]),int(sys.argv[4]))
    Correct(fw,pred,test_tar[int(sys.argv[3]):int(sys.argv[4])])
  fw.close()

def Qiskit_shot(par1,par2,train_img,test_img,train_tar,test_tar):
  global backend
  backend=BasicAer.get_backend('statevector_simulator')
  fw=open("pred_shot_%06d/pred_%s_%03d"%(N_Shot,sys.argv[1],int(sys.argv[2])),"w")
  if sys.argv[1]=="tran":
    pred=Pred_shot(par1,par2,train_img,int(sys.argv[3]),int(sys.argv[4]))
    Correct(fw,pred,train_tar[int(sys.argv[3]):int(sys.argv[4])])
  if sys.argv[1]=="test":
    pred=Pred_shot(par1,par2,test_img,int(sys.argv[3]),int(sys.argv[4]))
    Correct(fw,pred,test_tar[int(sys.argv[3]):int(sys.argv[4])])
  fw.close()

def Con_Unitary(par):
  U,S,VT=torch.linalg.svd(par.reshape(N_map,N_par*N_par))
  U2=torch.eye(N_par*N_par).cdouble()
  dim=U.shape[0]
  U2[:dim,:dim]=U
  Q=torch.matmul(U2,VT)
  return Q

def Main():
  test_tar,test_img,train_tar,train_img=Read_Data()
  par1,par2=Read_Para()
  #Plot_Circuit(Con_Unitary(par1))
  Qiskit_pro(par1,par2,train_img,test_img,train_tar,test_tar)
  #Qiskit_shot(par1,par2,train_img,test_img,train_tar,test_tar)

start=datetime.datetime.now()
Main()
end=datetime.datetime.now()
print("執行時間：",end-start)

