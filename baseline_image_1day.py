from dliteTools import *
from astropy.io import fits as py
from scinTools import *
from scipy.special import gamma
import datetime,glob,os,sys
fac=np.pi/180.
sfac=np.sqrt(np.array([25.,110.,27.,4.,1.,1.,1.]))
infile,parfile=sys.argv[1:3]
#read in parameter file
exec(open(parfile,'r').read())
rname=str.split(infile,'/')[-1].replace('_vis.npz','').replace('DLITE',tel)
outf=os.path.join(outdir,rname+'.fits')
ical=np.where(np.array(name)==calsour)[0][0]
#read in visibility data
data=np.load(infile)
vis=data['vis'] ; ts=data['tstamp']
frq=data['freq'] ; bw=data['bw']
del data.f ; data.close()
nb,npol,nt,nch=vis.shape
npol=2#only use XX and YY for imaging
dt=datetime.datetime(1970,1,1)+datetime.timedelta(seconds=int(ts[0]))
yr=dt.year
doy=(dt-datetime.datetime(dt.year,1,1,0)).days+1
#read in antenna and array calibration data
alat,alon=np.loadtxt(ANT,usecols = [0,1],unpack = True)
lat0=alat[0] ; lng0=alon[0]
na=len(alat)
data=np.load(calfile)
na=int(data['xx'].shape[0])+1
x=np.zeros(na) ; y=np.zeros(na) ; z=np.zeros(na) ; d=np.zeros(na)
x[1:]=data['xx'][:,pbest] ; y[1:]=data['yy'][:,pbest] ; z[1:]=data['zz']
d[1:]=data['dd'][:,pbest]
del data.f ; data.close()
dcorr=np.zeros(nb) ; dx=np.zeros(nb) ; dy=np.zeros(nb) ; dz=np.zeros(nb)
b=0
for i in range(na-1):
    for j in range(i+1,na):
        if((i+1==an1) & (j+1==an2)):
            bidx=b
        dx[b]=x[i]-x[j] ; dy[b]=y[i]-y[j] ; dz[b]=z[i]-z[j]
        dcorr[b]=d[i]-d[j] ; b+=1
#compute number of images
nim=int((nt-navg)/nstep)+1
#setup some arrays
tmid=np.zeros(nim)
idx=np.indices((navg,nch))
hamm=0.54-0.46*np.cos(2*np.pi*idx[0,:]/float(navg))
hamm*=0.54-0.46*np.cos(2*np.pi*idx[1,:]/float(nch))
lpw=np.zeros((nim,npol,navg,nch))
ome=2*np.pi/24./3600.#angular rotation rate of the Earth
dmax=np.sqrt(dx[bidx]**2+dy[bidx]**2)/2.998e8#maximum baseline delay
fmax=ome*np.sqrt(dx[bidx]**2+dy[bidx]**2)*35.e6/2.998e8#maximum baseline fring rate
delt=(np.arange(navg)-int(navg/2))*(ts[1]-ts[0])
fg=np.fft.fftshift(np.fft.fftfreq(navg,d=ts[1]-ts[0]))#fringe rates on image grid
dfg=fg[1]-fg[0] ; f1=fg[0]-dfg/2. ; f2=fg[-1]+dfg/2.
dg=np.fft.fftshift(np.fft.fftfreq(nch,d=bw/float(nch)))#delays on image grid
ddg=dg[1]-dg[0] ; d1=dg[0]-ddg/2. ; d2=dg[-1]+ddg/2.
#locate region of image where sources should be
r=np.sqrt(((dg[idx[1,:]]+dcorr[bidx])/dmax)**2+(fg[idx[0,:]]/fmax)**2)
iif,iid=np.unravel_index(np.where(r.reshape(navg*nch)<1)[0],r.shape)
#setup matrix for linear fit used to fill in gaps from flagged data
nfit=len(iif)
mat=np.zeros((navg*nch,nfit),'complex')
fchan=(np.arange(nch)-int(nch/2))*bw/float(nch)+frq#array of channel frequencies
#approximate bandpass shape
wg=0.57*nch ; xg=np.arange(nch)-nch/2.+0.5
bp=np.exp(-np.pi*(xg/wg)**8)
for i in range(nfit):
    p=2*np.pi*(fchan[idx[1,:]]*dg[iid[i]]+delt[idx[0,:]]*fg[iif[i]])
    mat[:,i]=(bp[idx[1,:]]*np.exp(1j*p)).reshape(navg*nch)
#arrays for A-Team positions precessed to observing date
ra_at=np.zeros((nim,ns)) ; dec_at=np.zeros((nim,ns))
for i in range(nim):
    i1=i*nstep ; i2=i1+navg
    tmid[i]=np.mean(ts[i1:i2])
    ppos_a=pos_a.transform_to(FK5(equinox=Time(tmid[i],format='unix')))
    ra_at[i,:]=np.array(ppos_a.ra)
    dec_at[i,:]=np.array(ppos_a.dec)
    for pol in range(npol):
        vtmp=np.copy(vis[bidx,pol,i1:i2,:])
        ind=np.where(np.abs(vtmp).reshape(navg*nch)>0)[0]
        nnz=len(ind)
        #only image if more than 3/4 of this chuck of data are non-zero (i.e., not flagged)
        if(nnz>=navg*nch*3/4):
            #perform linear fit of point sources to non-flagged data
            vtmp=vtmp.reshape(navg*nch)
            p=np.linalg.lstsq(mat[ind,:],vtmp[ind],rcond=None)[0]
            #build up array of visibilities from linear fit
            vsub=np.zeros(navg*nch,'complex')
            for k in range(nfit):
                vsub+=p[k]*mat[:,k]
            #replaced flagged visibilities with linear fit results
            jnd=np.where(np.abs(vtmp)==0)[0]
            vtmp[jnd]=vsub[jnd]
            vtmp=vtmp.reshape(navg,nch)
            #per channel, perform high-pass filtering by subtracting linear fit to time series
            for k in range(nch):
                ind=np.where(np.abs(vtmp[:,k])>0)[0]
                if(len(ind)>2):
                    c=np.polyfit(ind,vtmp[ind,k],1)
                    vtmp[ind,k]-=np.polyval(c,ind)
            #generate image with 2-D FFT, applying Hamming window
            tmp=fft2shift(np.fft.fft2(vtmp*hamm))/np.sum(hamm)
            lpw[i,pol,:]=np.log10(np.abs(tmp))

#get A-Team coordinates
lst=tstamp2lst(lng0,tmid)
idx=np.indices((nim,ns))
ha_at=np.mod(lst[idx[0,:]]*15-ra_at,360.)
alt,az=comp_altaz_ha(ha_at,dec_at,lat0)
#flux calibration
#find fringe rate/delay pixels corresponding to calibrator
f_a=ome*np.cos(dec_at[:,ical]*fac)*(dy[bidx]*np.sin(ha_at[:,ical]*fac)*np.sin(lat0*fac)-dx[bidx]*np.cos(ha_at[:,ical]*fac))*frq/2.998e8
d_a=(dy[bidx]*np.sin(dec_at[:,ical]*fac)*np.cos(lat0*fac)-dy[bidx]*np.cos(ha_at[:,ical]*fac)*np.cos(dec_at[:,ical]*fac)*np.sin(lat0*fac)-dx[bidx]*np.sin(ha_at[:,ical]*fac)*np.cos(dec_at[:,ical]*fac))/2.998e8
x1=(d1+dcorr[bidx])/dmax
x2=(d2+dcorr[bidx])/dmax
y1=f1/fmax ; y2=f2/fmax
x_a=(nch*(d_a/dmax-x1)/(x2-x1)).astype('int')
y_a=(navg*(f_a/fmax-y1)/(y2-y1)).astype('int')
#calibrate with data where image is non-zero and calibrator is above 30 deg. elevation
#assumes antenna pattern = sin(elevation)**1.6
ind=np.where((np.sum(lpw[:,:].reshape(nim,2*nch*navg),1)!=0) & (alt[:,ical]>30))[0]
ni=len(ind)
if(ni>2):
    for k in range(npol):
        ftmp=np.zeros(ni)
        for j in range(ni):
            ftmp[j]=10**lpw[ind[j],k,y_a[ind[j]],x_a[ind[j]]]/np.sin(alt[ind[j],ical]*fac)**1.6
        lpw[:,k,:]+=np.log10(f38[ical]/np.median(ftmp))
#find fringe rate/delay pixels corresponding to all A-Team sources
f_a=ome*np.cos(dec_at*fac)*(dy[bidx]*np.sin(ha_at*fac)*np.sin(lat0*fac)-dx[bidx]*np.cos(ha_at*fac))*frq/2.998e8
d_a=(dy[bidx]*np.sin(dec_at*fac)*np.cos(lat0*fac)-dy[bidx]*np.cos(ha_at*fac)*np.cos(dec_at*fac)*np.sin(lat0*fac)-dx[bidx]*np.sin(ha_at*fac)*np.cos(dec_at*fac))/2.998e8
x_a=np.zeros((nim,ns),'int') ; y_a=np.zeros((nim,ns),'int')
for i in range(ns):
    x_a[:,i]=(nch*(d_a[:,i]/dmax-x1)/(x2-x1)).astype('int')
    y_a[:,i]=(navg*(f_a[:,i]/fmax-y1)/(y2-y1)).astype('int')
#find fringe rate/delay pixels corresponding to the Sun
pos_s=pos_s=get_body('sun',Time(tmid,format='unix'))
ha_s=np.mod(lst*15-np.array(pos_s.ra),360.)*fac
dec_s=np.array(pos_s.dec)*fac
f_s=ome*np.cos(dec_s)*(dy[bidx]*np.sin(ha_s)*np.sin(lat0*fac)-dx[bidx]*np.cos(ha_s))*frq/2.998e8
d_s=(dy[bidx]*np.sin(dec_s)*np.cos(lat0*fac)-dy[bidx]*np.cos(ha_s)*np.cos(dec_s)*np.sin(lat0*fac)-dx[bidx]*np.sin(ha_s)*np.cos(dec_s))/2.998e8
alt_s,az_s=comp_altaz_ha(ha_s/fac,dec_s/fac,lat0)
x_s=(nch*(d_s/dmax-x1)/(x2-x1)).astype('int')
y_s=(navg*(f_s/fmax-y1)/(y2-y1)).astype('int')
#find peak and standard deviation of intensity per A-Team source
ut=np.mod(tstamp2ut(tmid),24.)
asig=np.zeros((nim,ns,npol)) ; amax=np.zeros((nim,ns,npol))
for i in range(nim):
    for j in range(ns):
        if(alt[i,j]>0):
            for k in range(npol):
                nfloor=np.mean(lpw[i,k,:10,x_a[i,j]])
                pk=np.max(lpw[i,k,y_a[i,j]-2:y_a[i,j]+3,x_a[i,j]])
                asig[i,j,k]=np.sqrt(navg)*10**nfloor
                amax[i,j,k]=10**pk-10**nfloor
#scintillation analysis
ll=np.cos(alt*fac)*np.sin(az*fac)
mm=np.cos(alt*fac)*np.cos(az*fac)
nn=np.sin(alt*fac)
#pierce point locations
plat,plon=dircos2ipp(ll,mm,nn,zion=zion,lat=lat0,lng=lng0)
#B-field geometry
mdip,mdec=mag_geom(plat,plon,yr,magfile)
phi=np.mod(np.arctan2(ll,mm)/fac-mdec,360.)
zdis=90-alt
#Geometric factors from Rino (1979)
gg,ff,reff,rperp=geofac(zdis*fac,phi*fac,mdip*fac,nu,A,B,1.,1.)
#compute parameters for converting S4 index to CkL
rel=2.818e-15 ; lam=2.998e8/frq
fac1=(rel*lam)**2*(2*np.pi/1000.)**(2*nu+1)
fac2=gamma(1.5-nu)/(np.pi*gamma(nu+0.5)*(2*nu-1)*2**(2*nu-1))
fac3=(lam*zion*1000/2./np.pi)**(nu-0.5)
fac4=gamma(1.25-nu*0.5)/(2**(nu+0.5)*np.sqrt(np.pi)*gamma(0.5*nu+0.25)*(nu-0.5))
cklfac=fac1*fac3*fac4/np.abs(np.sin(alt*fac))**(nu+0.5)
#simultaneously fit for CkL and system noise
ckl10=np.zeros((npol,nim)) ; asys10=np.zeros((npol,nim))
for i in range(nim):
    for j in range(npol):
        ind=np.where((amax[i,:,j]>0) & (alt[i,:]>10))[0]
        if(len(ind)>=3):
            x=ff[i,ind]*cklfac[i,ind]*amax[i,:,j][ind]**2/sfac[ind]
            y=asig[i,:,j][ind]**2
            c=np.polyfit(x,y,1)
            if((c[0]>=0) & (c[1]>=0)):
                ckl10[j,i]=c[0] ; asys10[j,i]=np.sqrt(c[1])
            if(c[0]<0):
                ckl10[j,i]=0. ; asys10[j,i]=np.sqrt(np.mean(y))
            if(c[1]<0):
                ckl10[j,i]=np.mean(y/x) ; asys10[j,i]=0.
#compute CkL per A-Team source
ckl10_a=np.zeros((nim,ns,npol))
for i in range(ns):
    for j in range(npol):
        ind=np.where((asig[:,i,j]>0) & (amax[:,i,j]>0))[0]
        ckl10_a[ind,i,j]=sfac[i]*(asig[ind,i,j]**2-asys10[j,ind]**2)/amax[ind,i,j]**2/cklfac[ind,i]/ff[ind,i]
#Write FITS file
nlist=name[0]
for i in range(1,ns):
    nlist+=', '+name[i]
if(os.path.exists(outf)):
    jnk=os.system('rm -f '+outf)
hdu=py.HDUList(py.PrimaryHDU(lpw[:,:]))
hdr=hdu[0].header
hdr['year']=yr
hdr['doy']=doy
hdr['sources']=nlist
hdr['alat1']=(alat[an1-1],'latitude of antenna 1')
hdr['alon1']=(alon[an1-1],'longitude of antenna 1')
hdr['alat2']=(alat[an2-1],'latitude of antenna 2')
hdr['alon2']=(alon[an2-1],'longitude of antenna 2')
hdr['alt']=(zsta,'array altitude (m)')
hdr['bunit']='log10 Jy'
hdr['dmax']=(dmax,'maximum delay (s)')
hdr['fmax']=(fmax,'maximum fringe rate (s)')
hdr['ctype4']='universal time'
hdr['crpix4']=1
hdr['crval4']=ut[0]
hdr['cdelt4']=ut[1]-ut[0]
hdr['ctype3']='polarization: 0=X, 1=Y'
hdr['crpix3']=1
hdr['crval3']=0
hdr['cdelt3']=1
d=(f2-f1)/float(navg)
hdr['ctype2']='fringe rate / FDOA (Hz)'
hdr['crpix2']=1
hdr['crval2']=f1+d/2.
hdr['cdelt2']=d
d=(d2-d1)/float(nch)
hdr['ctype1']='delay / TDOA (s)'
hdr['crpix1']=1
hdr['crval1']=d1+d/2.
hdr['cdelt1']=d
tdat=[py.Column(name='tstamp',format='E',array=tmid[:])]
tdat.append(py.Column(name='ut',format='E',array=ut[:]))
tdat.append(py.Column(name='lst',format='E',array=lst[:]))
tdat.append(py.Column(name='el_ateam',format=str(ns)+'E',array=alt[:,:]))
tdat.append(py.Column(name='az_ateam',format=str(ns)+'E',array=az[:,:]))
tdat.append(py.Column(name='x_ateam',format=str(ns)+'J',array=x_a[:,:]+1))
tdat.append(py.Column(name='y_ateam',format=str(ns)+'J',array=y_a[:,:]+1))
tdat.append(py.Column(name='el_sun',format='E',array=alt_s[:]))
tdat.append(py.Column(name='az_sun',format='E',array=az_s[:]))
tdat.append(py.Column(name='x_sun',format='J',array=x_s[:]+1))
tdat.append(py.Column(name='y_sun',format='J',array=y_s[:]+1))
tdat.append(py.Column(name='flux_ateam_x',format=str(ns)+'E',array=amax[:,:,0]))
tdat.append(py.Column(name='flux_ateam_y',format=str(ns)+'E',array=amax[:,:,1]))
tdat.append(py.Column(name='sigma_ateam_x',format=str(ns)+'E',array=asig[:,:,0]))
tdat.append(py.Column(name='sigma_ateam_y',format=str(ns)+'E',array=asig[:,:,1]))
tdat.append(py.Column(name='plat_ateam',format=str(ns)+'E',array=plat[:,:]))
tdat.append(py.Column(name='plon_ateam',format=str(ns)+'E',array=plon[:,:]))
tdat.append(py.Column(name='nsys_x',format='E',array=asys10[0,:]))
tdat.append(py.Column(name='nsys_y',format='E',array=asys10[1,:]))
tdat.append(py.Column(name='ckl_x',format='E',array=ckl10[0,:]))
tdat.append(py.Column(name='ckl_y',format='E',array=ckl10[1,:]))
tdat.append(py.Column(name='ckl_ateam_x',format=str(ns)+'E',array=ckl10_a[:,:,0]))
tdat.append(py.Column(name='ckl_ateam_y',format=str(ns)+'E',array=ckl10_a[:,:,1]))
cols=py.ColDefs(tdat)
hdu.append(py.BinTableHDU.from_columns(cols))
hdu.writeto(outf)
hdu.close()
