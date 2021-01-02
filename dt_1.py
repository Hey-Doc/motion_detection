from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import math

#functions
def dot(u,v): 
    return u[0]*v[0]+u[1]*v[1]+u[2]*v[2]


#data loading
fall_acc = pd.read_csv("dataset/fall1_accm.csv")
fall_grv = pd.read_csv("dataset/fall1_grvm.csv")
nfall_acc = pd.read_csv("dataset/notfall1_accm.csv")
nfall_grv = pd.read_csv("dataset/notfall1_grvm.csv")

#data formatting
fall_acc = fall_acc/5
fall_acc['time']=fall_acc['time']*5
nfall_acc = nfall_acc/5
nfall_acc['time']=nfall_acc['time']*5
f_acc=fall_acc.rolling(window=5,min_periods=1).mean()
nf_acc=nfall_acc.rolling(window=5,min_periods=1).mean()
vf=[]
vnf=[]


for i in range(len(f_acc)//500):
    s=[0,0,0]
    for j in range(500):
        if 500*i+j>=len(f_acc):
            break;
        s[0]+=f_acc['X_value'][500*i+j]
        s[1]+=f_acc['Y_value'][500*i+j]
        s[2]+=f_acc['Z_value'][500*i+j]
        
    s[0]/=500
    s[1]/=500
    s[2]/=500
    size=math.sqrt(s[0]**2 + s[1]**2 + s[2]**2)    
    vf.append([s[0]/size,s[1]/size,s[2]/size])

for i in range(len(nf_acc)//500):
    s=[0,0,0]
    for j in range(500):
        if 500*i+j>=len(nf_acc):
            break;
        s[0]+=nf_acc['X_value'][500*i+j]
        s[1]+=nf_acc['Y_value'][500*i+j]
        s[2]+=nf_acc['Z_value'][500*i+j]
        
    s[0]/=500
    s[1]/=500
    s[2]/=500
    size=math.sqrt(s[0]**2 + s[1]**2 + s[2]**2)    
    vnf.append([s[0]/size,s[1]/size,s[2]/size])

vf_np=np.array(vf)
vnf_np=np.array(vnf)

pf=[]
pnf=[]
hf=[]
hnf=[]

for i in range(len(f_acc)):
    pf+=dot(f_acc.loc[i],vf[i//500])*vf[i//500]