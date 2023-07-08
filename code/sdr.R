discretize=function(y,h){
  n=length(y);m=floor(n/h)
  y=y+.00001*mean(y)*rnorm(n)
  yord = y[order(y)]
  divpt=numeric();for(i in 1:(h-1)) divpt = c(divpt,yord[i*m+1])
  y1=rep(0,n);y1[y<divpt[1]]=1;y1[y>=divpt[h-1]]=h
  for(i in 2:(h-1)) y1[(y>=divpt[i-1])&(y<divpt[i])]=i
  return(y1)}

matpower = function(a,alpha){
  a = round((a + t(a))/2,7); tmp = eigen(a)
  return(tmp$vectors%*%diag((tmp$values)^alpha)%*%t(tmp$vectors))}


wls=function(x,y,w){
  n=dim(x)[1];p=dim(x)[2]-1
  out=c(solve(t(x*w)%*%x/n)%*%apply(x*y*w,2,mean))
  return(list(a=out[1],b=out[2:(p+1)]))}

kern=function(x,h){
  x=as.matrix(x);n=dim(x)[1]
  k2=x%*%t(x);k1=t(matrix(diag(k2),n,n));k3=t(k1);k=k1-2*k2+k3
  return(exp(-(1/(2*h^2))*(k1-2*k2+k3)))}

standvec=function(x) return((x-mean(x))/sd(x))



sir=function(x,y,h,r,ytype){
  y = as.vector(y)
  p=ncol(x);n=nrow(x)
  signrt=matpower(var(x),-1/2)
  xc=t(t(x)-apply(x,2,mean))
  xst=xc%*%signrt
  if(ytype=="continuous") ydis=discretize(y,h)
  if(ytype=="categorical") ydis=y
  yless=ydis;ylabel=numeric()
  for(i in 1:n) {if(var(yless)!=0) {ylabel=
    c(ylabel,yless[1]);yless=yless[yless!=yless[1]]}}
  ylabel=c(ylabel,yless[1])
  prob=numeric();exy=numeric()
  for(i in 1:h) prob=c(prob,length(ydis[ydis==ylabel[i]])/n)
  for(i in 1:h) exy=rbind(exy,apply(xst[ydis==ylabel[i],],2,mean))
  sirmat=t(exy)%*%diag(prob)%*%exy
  return(signrt%*%eigen(sirmat)$vectors[,1:r])}



save=function(x,y,h,r,ytype){
  y = as.vector(y)
  p=ncol(x);n=nrow(x)
  signrt=matpower(var(x),-1/2)
  xc=t(t(x)-apply(x,2,mean))
  xst=xc%*%signrt
  if(ytype=="continuous") ydis=discretize(y,h)
  if(ytype=="categorical") ydis=y
  yless=ydis;ylabel=numeric()
  for(i in 1:n) {
    if(var(yless)!=0) {ylabel=c(ylabel,yless[1])
    yless=yless[yless!=yless[1]]}}
  ylabel=c(ylabel,yless[1])
  prob=numeric()
  for(i in 1:h) prob=c(prob,length(ydis[ydis==ylabel[i]])/n)
  vxy = array(0,c(p,p,h))
  for(i in 1:h) vxy[,,i] = var(xst[ydis==ylabel[i],])
  savemat=0
  for(i in 1:h){
    savemat=savemat+prob[i]*(vxy[,,i]-diag(p))%*%(vxy[,,i]-diag(p))}
  return(signrt%*%eigen(savemat)$vectors[,1:r])}


dr = function(x,y,h,r,ytype){
  y = as.vector(y)
  p=ncol(x);n=nrow(x)
  signrt=matpower(var(x),-1/2)
  xc=t(t(x)-apply(x,2,mean))
  xst=xc%*%signrt
  if(ytype=="continuous") ydis=discretize(y,h)
  if(ytype=="categorical") ydis=y
  yless=ydis;ylabel=numeric()
  for(i in 1:n) {
    if(var(yless)!=0) {ylabel=c(ylabel,yless[1])
    yless=yless[yless!=yless[1]]}}
  ylabel=c(ylabel,yless[1])
  prob=numeric()
  for(i in 1:h) prob=c(prob,length(ydis[ydis==ylabel[i]])/n)
  vxy = array(0,c(p,p,h));exy=numeric()
  for(i in 1:h) {
    vxy[,,i]=var(xst[ydis==ylabel[i],])
    exy=rbind(exy,apply(xst[ydis==ylabel[i],],2,mean))}
  mat1 = matrix(0,p,p);mat2 = matrix(0,p,p)
  for(i in 1:h){
    mat1 = mat1+prob[i]*(vxy[,,i]+exy[i,]%*%t(exy[i,]))%*%
      (vxy[,,i]+exy[i,]%*%t(exy[i,]))
    mat2 = mat2+prob[i]*exy[i,]%*%t(exy[i,])}
  out = 2*mat1+2*mat2%*%mat2+2*sum(diag(mat2))*mat2-2*diag(p)
  V = eigen(out)$values[1] * eigen(out)$vectors[,1] + eigen(out)$values[2] * eigen(out)$vectors[,2]
  return(beta = signrt%*%eigen(out)$vectors[,1:r])
}

opg=function(x,y,d){
y = as.vector(y)
p=dim(x)[2];n=dim(x)[1]
c0=2.34;p0=max(p,3);rn=n^(-1/(2*(p0+6)));h=c0*n^(-(1/(p0+6)))
sig=diag(var(x));x=apply(x,2,standvec)
kmat=kern(x,h);bmat=numeric()
for(i in 1:dim(x)[1]){
wi=kmat[,i];xi=cbind(1,t(t(x)-x[i,]))
bmat=cbind(bmat,wls(xi,y,wi)$b)}
beta=eigen(bmat%*%t(bmat))$vectors[,1:d]
return(diag(sig^(-1/2))%*%beta)
}


mave=function(x,y,d,nit){
  y = as.vector(y)
  sig=diag(var(x));n=dim(x)[1];p=dim(x)[2]
  c0=2.34;p0=max(p,3);rn=n^(-1/(2*(p0+6)));h=c0*n^(-(1/(p0+6)))
  x=apply(x,2,standvec);beta=opg(x,y,d);kermat=kern(x,h)
  for(iit in 1:nit){
    b=numeric();a=numeric(); for(i in 1:n){
      wi=kermat[,i]/(apply(kermat,2,mean)[i])
      ui=cbind(1,t(t(x)-x[i,])%*%beta)
      out=wls(ui,y,wi);a=c(a,out$a);b=cbind(b,out$b)}
    out=0;out1=0; for(i in 1:n){
      xi=kronecker(t(t(x)-x[i,]),t(b[,i]))
      yi=y-a[i];wi=kermat[,i]/apply(kermat,2,mean)[i]
      out=out+apply(xi*yi*wi,2,mean)
      out1=out1+t(xi*wi)%*%xi/n}
    beta=t(matrix(solve(out1)%*%out,d,p))}
  return(diag(sig^(-1/2))%*%beta)}

rmave=function(x,y,d,nit){
  y = as.vector(y)
  sig=diag(var(x));n=dim(x)[1];p=dim(x)[2]
  x=apply(x,2,standvec)
  c0=2.34;p0=max(p,3);h=c0*n^(-(1/(p0+6)));rn=n^(-1/(2*(p0+6)))
  beta=opg(x,y,d)
  for(iit in 1:nit){
    kermat=kern(x%*%beta,h);mkermat=apply(kermat,2,mean)
    b=numeric();a=numeric()
    for(i in 1:n){
      wi=kermat[,i]/mkermat[i];ui=cbind(1,t(t(x)-x[i,])%*%beta)
      out=wls(ui,y,wi);a=c(a,out$a);b=cbind(b,out$b)}
    out=0;out1=0
    for(i in 1:n) {
      xi=kronecker(t(t(x)-x[i,]),t(b[,i]));yi=y-a[i]
      wi=kermat[,i]/mkermat[i]
      out=out+apply(xi*yi*wi,2,mean)
      out1=out1+t(xi*wi)%*%xi/n}
    beta=t(matrix(solve(out1)%*%out,d,p))
    h=max(rn*h,c0*n^((-1/(d+4))))
  }
  return(diag(sig^(-1/2))%*%beta)}



