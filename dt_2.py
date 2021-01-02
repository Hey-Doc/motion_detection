import pandas as pd
import math

#functions
def dot(u,v): 
    return u[3]*v[3]+u[1]*v[1]+u[2]*v[2]


#data loading
fall_acc = pd.read_csv("dataset/cor-fall-1_acc.csv")
nfall_acc = pd.read_csv("dataset/cor-notfall-1_acc.csv")

#data formatting
fall_acc = fall_acc/5
fall_acc['time']=fall_acc['time']*5
nfall_acc = nfall_acc/5
nfall_acc['time']=nfall_acc['time']*5
f_acc=fall_acc.rolling(window=5,min_periods=1).mean()
nf_acc=nfall_acc.rolling(window=5,min_periods=1).mean()
vf=f_acc.rolling(window=70,min_periods=1).mean()
vnf=nf_acc.rolling(window=70,min_periods=1).mean()
"""
f_acc=f_acc.dropna(axis=0)
nf_acc=nf_acc.dropna(axis=0)
vf=vf.dropna(axis=0)
vnf=vnf.dropna(axis=0)

f_acc=f_acc.reset_index(drop=False, inplace=False)
nf_acc=nf_acc.reset_index(drop=False, inplace=False)
vf=vf.reset_index(drop=False, inplace=False)
vnf=vnf.reset_index(drop=False, inplace=False)
"""
#feature extraction
pf=[]
pnf=[]
hf=[]
hnf=[]
psf=[]
psnf=[]
mf=[]
mnf=[]
for i in range(len(f_acc)):
    sizesq=vf['X_value'][i]**2+vf['Y_value'][i]**2+vf['Z_value'][i]**2
    size=math.sqrt(sizesq)
    cons=dot(f_acc.loc[i],vf.loc[i])/sizesq
    pf.append([cons*vf['X_value'][i],cons*vf['Y_value'][i],cons*vf['Z_value'][i]])
    hf.append(math.sqrt((f_acc['X_value'][i]-pf[i][0])**2 + (f_acc['Y_value'][i]-pf[i][1])**2 + (f_acc['Z_value'][i]-pf[i][2])**2))
    psf.append(cons*size)
    mf.append(math.sqrt(hf[i]**2 + psf[i]**2))

for i in range(len(nf_acc)):
    sizesq=vnf['X_value'][i]**2+vnf['Y_value'][i]**2+vnf['Z_value'][i]**2
    size=math.sqrt(sizesq)
    cons=dot(nf_acc.loc[i],vnf.loc[i])/sizesq
    pnf.append([cons*vnf['X_value'][i],cons*vnf['Y_value'][i],cons*vnf['Z_value'][i]])
    hnf.append(math.sqrt((nf_acc['X_value'][i]-pnf[i][0])**2 + (nf_acc['Y_value'][i]-pnf[i][1])**2 + (nf_acc['Z_value'][i]-pnf[i][2])**2))
    psnf.append(cons*size)
    mnf.append(math.sqrt(hnf[i]**2 + psnf[i]**2))

meanv_f=pd.Series(psf).rolling(window=10).mean()
meanh_f=pd.Series(hf).rolling(window=10).mean()
meanm_f=pd.Series(mf).rolling(window=10).mean()

stdv_f=pd.Series(psf).rolling(window=10).std()
stdh_f=pd.Series(hf).rolling(window=10).std()
stdm_f=pd.Series(mf).rolling(window=10).std()

meanv_nf=pd.Series(psnf).rolling(window=10).mean()
meanh_nf=pd.Series(hnf).rolling(window=10).mean()
meanm_nf=pd.Series(mnf).rolling(window=10).mean()

stdv_nf=pd.Series(psnf).rolling(window=10).std()
stdh_nf=pd.Series(hnf).rolling(window=10).std()
stdm_nf=pd.Series(mnf).rolling(window=10).std()
#extracted feature to csv file
dataset_fall = pd.concat([meanv_f,meanh_f,meanm_f,stdv_f,stdh_f,stdm_f],axis=1,keys=['meanV','meanH','meanM','stdV','stdH','stdM'])
dataset_nfall = pd.concat([meanv_nf,meanh_nf,meanm_nf,stdv_nf,stdh_nf,stdm_nf],axis=1,keys=['meanV','meanH','meanM','stdV','stdH','stdM'])

dataset_fall=dataset_fall.dropna(axis=0)
dataset_nfall=dataset_nfall.dropna(axis=0)

dataset_fall=dataset_fall.reset_index(drop=True, inplace=False)
dataset_nfall=dataset_nfall.reset_index(drop=True, inplace=False)

#dataset fusion
dataset_fall['dofall']='yes'
dataset_nfall['dofall']='no'
dataset_net=pd.concat([dataset_fall,dataset_nfall],axis=0)

#file export
dataset_fall.to_csv('./formatted_dataset/data_fall_3.csv', sep=',',na_rep=0)
dataset_nfall.to_csv('./formatted_dataset/data_nfall_3.csv', sep=',',na_rep=0)
dataset_net.to_csv('./formatted_dataset/data_net_3.csv',sep=',',na_rep=0)