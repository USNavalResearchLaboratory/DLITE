import numpy as np
from dliteTools import *
import os,sys,datetime
#script takes a single argument, the name of the directory
#containing the DLITE correlator output
#
#outputs numpy binary file in OUT directory
#file name is DLITE-YYYY-MM-DDThhmmss_vis.npz
#can tweak flagging parameters below
#to turn off RFI subtraction, set SUB=False
#
indir=sys.argv[1]
fac=np.pi/180.
OUT='./'

Wf=16 ; Wt=120 ; Thresh=5#flagging parameters
NCHMIN=485
NTAV=59
NCHAV=8
CLOB=False
SUB=True
WSUB=59

files=os.listdir(indir)
n2=[]
for f in files:
    if('visfile' in f):
        n2.append(int(f[9:10]))

na=np.max(n2)

SETUP=os.path.join(indir,'SETUP')
exec(open(SETUP).read())
nav=int(TINT*BANDWIDTH/NCHAN) ; tint=nav*NCHAN/BANDWIDTH
nthr=int(3600/tint)
ts=get_tstamps(indir) ; t0=ts[0]
dstr=datetime.datetime.utcfromtimestamp(t0).strftime('%Y-%m-%dT%H%M%S')
outfile=os.path.join(OUT,'DLITE-'+dstr+'_vis.npz')
if((os.path.exists(outfile)) & (CLOB==False)):
    sys.exit('Output file exists')
print('Reading in visibilities')
v0=read_vis(indir,na=na)
print('Flagging visibilities')
flg=flag_vis(v0,Wf,Wt,Thresh)
flg+=flag_vis(v0,Wf,nthr,Thresh) ; np.place(flg,flg>0,1)
if(t0<1573700000):
    #fixes CPU clock offsets before 11/14/2019
    if(sta=='POM'):
        ts+=313. ; t0+=313.
    if(sta=='NM'):
        ts+=267. ; t0+=267.

#compute some parameters
nb,npol,ntmast,nchmast=v0.shape
nt=int(ntmast/NTAV) ; nch=int(nchmast/NCHAV)
nbad=np.sum(flg,3)
nbad=np.repeat(nbad.reshape(nb*npol*ntmast),nchmast).reshape(v0.shape)
np.place(flg,nbad>nchmast-NCHMIN,1)
del nbad
vis=np.zeros((nb,npol,nt,nch),'complex')
nflg=np.zeros(vis.shape,'int')
for i in range(nt):
    for j in range(nch):
        tmp=np.ma.masked_where(flg[:,:,i*NTAV:(i+1)*NTAV,j*NCHAV:(j+1)*NCHAV]==1,v0[:,:,i*NTAV:(i+1)*NTAV,j*NCHAV:(j+1)*NCHAV])
        vis[:,:,i,j]=np.mean(tmp.reshape(nb,npol,NTAV*NCHAV),2).data
        nflg[:,:,i,j]=np.sum(flg[:,:,i*NTAV:(i+1)*NTAV,j*NCHAV:(j+1)*NCHAV].reshape(nb,npol,NTAV*NCHAV),2)

if(SUB):
    print('Doing RFI subtraction')
    vis=rfisub(vis,wdetrend=WSUB)

np.place(vis,nflg>int(NTAV*NCHAV/2),0.)
t=ts[np.arange(nt)*NTAV+int(NTAV/2)]
np.savez(outfile,tstamp=t,bw=BANDWIDTH,freq=FREQ,vis=vis)
