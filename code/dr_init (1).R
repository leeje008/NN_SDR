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



dr = function(x,y,h,r,ytype){
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
return(list(beta = signrt%*%eigen(out)$vectors[,1:r], adj_pen = V, eigen_val = eigen(out)$values ))
}

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
  return(list(beta = signrt%*%eigen(sirmat)$vectors[,1:r]))
}


