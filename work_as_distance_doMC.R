fileDir="../../"
library("matrixStats"); # rowSds(), rowVars()
#library(foreach); library(doMC);
fname2 <- as.matrix(read.table("COORDINATE_NAME_LIST.dat"));
nsim=dim(fname2)[1]*dim(fname2)[2]; fname=as.vector(t(fname2));
options("scipen"=100, "digits"=4); # enforce no-scientific way to show numbers
TPindex=as.matrix(read.table("select_TPs_tmp.dat"));
nTot=length(TPindex[,1]);

nLen2=40000; nbin=400; nCircle=1; ncut=40; ncut2=8; ncpu=2; njobs=2; nFcomp=1; #nTot=32;

hkbT=2.48/2;

#setup parallel backend to use 4 processors
#registerDoMC(ncpu)
#start time
strt<-Sys.time()

# get the start and end index of dihedrals
nDihB=dim(fname2)[1]*2-3; nDihE=nsim-5;
nsim2=nsim; #nsim2=nsim-6;
kl_ind <- as.matrix(read.table("kl_ind.dat"));

PBcut=0.995; R0=(1-PBcut); R1=PBcut;
dR=(R1-R0)/nbin;

# estimate the averaged work along a coordinate
avedW=array(0, dim=c(nsim2*nCircle,nbin)); dWcount=array(0, dim=c(nsim2*nCircle,nbin));   # nCircle: the case with different lengths
avedW2=array(0, dim=c(nsim2*nCircle,nbin)); dWcount2=array(0, dim=c(nsim2*nCircle,nbin));  # record average work from q->q+dq and q+dq->q, respectively.
nn=1; while(nn<=nTot){
  simIndex=TPindex[nn,1];

  dWfile=paste(fileDir,"/traj_",simIndex,"/intQ_all_coordinate.xvg",sep="");
  newdata = file(dWfile, "rb")
  readDat=readBin(newdata,double(),nsim*nLen2,endian="little")
  dim(readDat) <- c(nsim,nLen2);
  readDat=t(readDat);
  close(newdata)
  intQmat=readDat;

  dWfile=paste(fileDir,"/traj_",simIndex,"/geneFcomp_all_coordinate_",nFcomp,".xvg",sep="");
  newdata = file(dWfile, "rb")
  readDat=readBin(newdata,double(),nsim*nLen2,endian="little")
  dim(readDat) <- c(nsim,nLen2);
  readDat=t(readDat);
  close(newdata)
  geneFmat=readDat;

PBfile=paste("fitModelRes_",simIndex,".dat",sep="");
datPB=as.matrix(read.table(PBfile));
nins=80; datPB[,1]=(datPB[,1]-2)*nins+1;
# linear interpolation
dimPB=dim(datPB)[1]; lenPB=datPB[dimPB,1]; ngapPB=datPB[2,1]-datPB[1,1]; dataPB2=rep(NA,lenPB);
dataPB2[1]=datPB[2,2]; for(ii in 1:(dimPB-2)){ dataPB2[(ii-1)*ngapPB+(1:ngapPB)+1]=datPB[ii+1,2]+(datPB[(ii+2),2]-datPB[ii+1,2])/ngapPB*(1:ngapPB); }
endIndex=lenPB; startIndex=1;
for(ii in 1:lenPB){ if(dataPB2[ii]>PBcut){ endIndex=ii; break;} }
for(ii in seq(endIndex,1,-1)){ if(dataPB2[ii]<(1-PBcut)){ startIndex=ii; break;} }
nS=startIndex; nE=endIndex;

intQdat=dataPB2;

for(uu in 1:nsim){
  kk=kl_ind[uu];
  #if(kk >= nDihE) next;

# get the change of internal coordinates
delQmat=intQmat[2:nLen2,uu]-intQmat[1:(nLen2-1),uu];
if(any(delQmat > pi)){ delQmat[which(delQmat > pi)]=delQmat[which(delQmat > pi)]-2*pi; } # include the correction to dihedrals
if(any(delQmat < -pi)){ delQmat[which(delQmat < -pi)]=delQmat[which(delQmat < -pi)]+2*pi; }

dWdat2=delQmat;
#for(jj in 1:(nLen2-1)){ dWdat2[jj]=(geneFmat[jj,uu]+geneFmat[jj+1,uu]+dlnGdqiMat[jj,uu]+dlnGdqiMat[jj+1,uu])*delQmat[jj]/2; }
for(jj in 1:(nLen2-1)){ dWdat2[jj]=(geneFmat[jj,uu]+geneFmat[jj+1,uu])*delQmat[jj]/2; }
dWdat=dWdat2;

for(ncc in 1:nCircle){

  ii=nS; nLen=nE;

iSeg=ceiling((intQdat[ii]-R0)/dR); if(iSeg>nbin){ iSeg=nbin; }
if(iSeg<1){ iSeg=1; }
iSeg2=iSeg;
while(ii<nLen){
  iSeg=ceiling((intQdat[ii+1]-R0)/dR);  if(iSeg>nbin){ iSeg=nbin; } # avoid overflow
  if(iSeg<1){ iSeg=1; }
  if(iSeg2==iSeg){
    if(intQdat[ii+1]>intQdat[ii]){
      avedW[kk+(ncc-1)*nsim2,iSeg2]=avedW[kk+(ncc-1)*nsim2,iSeg2]+dWdat[ii]; dWcount[kk+(ncc-1)*nsim2,iSeg2]=dWcount[kk+(ncc-1)*nsim2,iSeg2]+abs((intQdat[ii+1]-intQdat[ii])/dR);
    }else{
      avedW2[kk+(ncc-1)*nsim2,iSeg2]=avedW2[kk+(ncc-1)*nsim2,iSeg2]+dWdat[ii]; dWcount2[kk+(ncc-1)*nsim2,iSeg2]=dWcount2[kk+(ncc-1)*nsim2,iSeg2]+abs((intQdat[ii+1]-intQdat[ii])/dR);
    }
    ii=ii+1; next;
  }
  ngap=abs(iSeg-iSeg2);
    dWtmp=dR/abs(intQdat[ii+1]-intQdat[ii])*dWdat[ii];
    if(iSeg>iSeg2){
      avedW[kk+(ncc-1)*nsim2,iSeg2]=avedW[kk+(ncc-1)*nsim2,iSeg2]+((R0+iSeg2*dR)-intQdat[ii])/(intQdat[ii+1]-intQdat[ii])*dWdat[ii];
      dWcount[kk+(ncc-1)*nsim2,iSeg2]=dWcount[kk+(ncc-1)*nsim2,iSeg2]+abs(((R0+iSeg2*dR)-intQdat[ii])/dR);
      jj=1; while(jj<ngap){ avedW[kk+(ncc-1)*nsim2,(iSeg2+jj)]=avedW[kk+(ncc-1)*nsim2,(iSeg2+jj)]+dWtmp; dWcount[kk+(ncc-1)*nsim2,(iSeg2+jj)]=dWcount[kk+(ncc-1)*nsim2,(iSeg2+jj)]+1; jj=jj+1; }
      avedW[kk+(ncc-1)*nsim2,iSeg]=avedW[kk+(ncc-1)*nsim2,iSeg]+(intQdat[ii+1]-(R0+(iSeg-1)*dR))/(intQdat[ii+1]-intQdat[ii])*dWdat[ii];
      dWcount[kk+(ncc-1)*nsim2,iSeg]=dWcount[kk+(ncc-1)*nsim2,iSeg]+abs(((intQdat[ii+1]-(R0+(iSeg-1)*dR))/dR));
    }else{
      avedW2[kk+(ncc-1)*nsim2,iSeg2]=avedW2[kk+(ncc-1)*nsim2,iSeg2]+((R0+(iSeg2-1)*dR)-intQdat[ii])/(intQdat[ii+1]-intQdat[ii])*dWdat[ii];
      dWcount2[kk+(ncc-1)*nsim2,iSeg2]=dWcount2[kk+(ncc-1)*nsim2,iSeg2]+abs(((R0+(iSeg2-1)*dR)-intQdat[ii])/dR);
      jj=1; while(jj<ngap){ avedW2[kk+(ncc-1)*nsim2,(iSeg2-jj)]=avedW2[kk+(ncc-1)*nsim2,(iSeg2-jj)]+dWtmp; dWcount2[kk+(ncc-1)*nsim2,(iSeg2-jj)]=dWcount2[kk+(ncc-1)*nsim2,(iSeg2-jj)]+1; jj=jj+1; }
      avedW2[kk+(ncc-1)*nsim2,iSeg]=avedW2[kk+(ncc-1)*nsim2,iSeg]+(intQdat[ii+1]-(R0+iSeg*dR))/(intQdat[ii+1]-intQdat[ii])*dWdat[ii];
      dWcount2[kk+(ncc-1)*nsim2,iSeg]=dWcount2[kk+(ncc-1)*nsim2,iSeg]+abs((intQdat[ii+1]-(R0+iSeg*dR))/dR);
    }
    ii=ii+1;
    iSeg2=iSeg;
}
}
}
nn=nn+1;
buf=gc();
}

print(Sys.time()-strt)

write.table(avedW,file="avedW_f.dat");
write.table(dWcount,file="dWcount_f.dat");
write.table(avedW2,file="avedW_b.dat");
write.table(dWcount2,file="dWcount_b.dat");

q()

