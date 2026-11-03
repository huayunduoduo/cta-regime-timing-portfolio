#include <cmath>
#include <algorithm>
#include <random>
#include <vector>
#include <cstring>
using namespace std;
// Transformed Gaussian joint AR(1)-GARCH(1,1) likelihood and exact gradient.
static double eval(const double*z,const double*y,int n,double*g,double*last=nullptr){
 double rho=tanh(z[1]),dr=1-rho*rho,om=exp(z[2]),aa=exp(z[3]),bb=exp(z[4]),den=1+aa+bb;
 double a=.999*aa/den,b=.999*bb/den;
 double da[5]={0,0,0,a*(1-aa/den),-a*bb/den},db[5]={0,0,0,-b*aa/den,b*(1-bb/den)};
 double mean=0,v=0;for(int i=0;i<n;i++)mean+=y[i]/n;for(int i=0;i<n;i++)v+=(y[i]-mean)*(y[i]-mean)/n;
 double h=max(v,1e-6),dh[5]={},old=0,deold[5]={},val=0;fill(g,g+5,0.);
 for(int i=1;i<n;i++){
  if(i>1){double hn=om+a*old*old+b*h;for(int j=0;j<5;j++)dh[j]=(j==2?om:0)+da[j]*old*old+2*a*old*deold[j]+db[j]*h+b*dh[j];h=max(hn,1e-12);}
  double e=y[i]-z[0]-rho*y[i-1],de[5]={-1,-dr*y[i-1],0,0,0};
  val+=.5*(log(h)+e*e/h);
  for(int j=0;j<5;j++)g[j]+=(.5/h-.5*e*e/(h*h))*dh[j]+e/h*de[j];
  old=e;copy(de,de+5,deold);
 }
 if(last)*last=h;
 return val;
}
static int optimize(const double*y,int n,double*z,int maxit=200){
 double H[25]={};for(int j=0;j<5;j++)H[j*5+j]=1.;double g[5],f=eval(z,y,n,g);if(!isfinite(f))return 0;
 for(int it=0;it<maxit;it++){
  double gn=0;for(int j=0;j<5;j++)gn=max(gn,abs(g[j]));if(gn<1e-5)return 1;
  double p[5]={};for(int j=0;j<5;j++)for(int k=0;k<5;k++)p[j]-=H[j*5+k]*g[k];
  double dot=0;for(int j=0;j<5;j++)dot+=p[j]*g[j];
  if(!isfinite(dot)||dot>=0){dot=0;fill(H,H+25,0.);for(int j=0;j<5;j++){H[j*5+j]=1;p[j]=-g[j];dot-=g[j]*g[j];}}
  double step=1,zz[5],gg[5],ff;bool ok=false;
  for(int ls=0;ls<35;ls++){
   for(int j=0;j<5;j++){zz[j]=z[j]+step*p[j];double lo=j==0?-100:j==1?-4:j==2?-25:-20;double hi=j==0?100:j==1?4:j==2?10:20;zz[j]=max(lo,min(hi,zz[j]));}
   ff=eval(zz,y,n,gg);
   if(isfinite(ff)&&ff<=f+1e-4*step*dot){ok=true;break;}step*=.5;
  }
  if(!ok)return gn<1e-3?1:0;
  double s[5],q[5],Hq[5]={},sq=0,qHq=0;
  for(int j=0;j<5;j++){s[j]=zz[j]-z[j];q[j]=gg[j]-g[j];sq+=s[j]*q[j];}
  if(sq>1e-12){for(int j=0;j<5;j++)for(int k=0;k<5;k++)Hq[j]+=H[j*5+k]*q[k];for(int j=0;j<5;j++)qHq+=q[j]*Hq[j];
   for(int j=0;j<5;j++)for(int k=0;k<5;k++)H[j*5+k]+=(1+qHq/sq)*s[j]*s[k]/sq-(s[j]*Hq[k]+Hq[j]*s[k])/sq;
  }
  bool settled=abs(ff-f)<=1e-10*(1+abs(f));copy(zz,zz+5,z);copy(gg,gg+5,g);f=ff;if(settled)return 1;
 }
 return 0;
}
static void init(const double*y,int n,double*z){double mx=0,my=0;for(int i=1;i<n;i++){mx+=y[i-1]/(n-1);my+=y[i]/(n-1);}double xx=0,xy=0,v=0;
 for(int i=1;i<n;i++){xx+=(y[i-1]-mx)*(y[i-1]-mx);xy+=(y[i-1]-mx)*(y[i]-my);}double r=max(-.95,min(.95,xy/max(xx,1e-12)));z[0]=my-r*mx;z[1]=atanh(r);
 for(int i=1;i<n;i++)v+=pow(y[i]-z[0]-r*y[i-1],2)/(n-1);z[2]=log(max(v*.05,1e-7));z[3]=log(.05/.049);z[4]=log(.90/.049);
}
static void physical(const double*z,double*p){double aa=exp(z[3]),bb=exp(z[4]);p[0]=z[0];p[1]=tanh(z[1]);p[2]=exp(z[2]);p[3]=.999*aa/(1+aa+bb);p[4]=.999*bb/(1+aa+bb);}
extern "C" int fit_model(const double*y,int n,double*out,double*zout){
 double z[5];init(y,n,z);int ok=optimize(y,n,z);double g[5],h;eval(z,y,n,g,&h);physical(z,out);out[5]=sqrt(h);out[6]=z[0]/out[5];copy(z,z+5,zout);return ok;
}
extern "C" void gradient(const double*z,const double*y,int n,double*g){eval(z,y,n,g);}
extern "C" double objective(const double*z,const double*y,int n){double g[5];return eval(z,y,n,g);}
extern "C" int bootstrap(const double*y,int n,const double*z,int B,unsigned long long seed,double*out){
 double p[5];physical(z,p);double mean=0,v=0;for(int i=0;i<n;i++)mean+=y[i]/n;for(int i=0;i<n;i++)v+=pow(y[i]-mean,2)/n;
 vector<double>res(n-1),yb(n);double h=max(v,1e-6),old=0,rmean=0;
 for(int i=1;i<n;i++){if(i>1)h=p[2]+p[3]*old*old+p[4]*h;old=y[i]-p[0]-p[1]*y[i-1];res[i-1]=old/sqrt(max(h,1e-12));rmean+=res[i-1]/(n-1);}
 double rv=0;for(auto &e:res){e-=rmean;rv+=e*e/(n-1);}for(auto&e:res)e/=sqrt(max(rv,1e-12));
 mt19937_64 rng(seed);uniform_int_distribution<int>pick(0,n-2);int valid=0;
 for(int k=0;k<B;k++){
  yb[0]=y[0];double hh=max(v,1e-6),ee=0;
  for(int i=1;i<n;i++){if(i>1)hh=p[2]+p[3]*ee*ee+p[4]*hh;ee=sqrt(max(hh,1e-12))*res[pick(rng)];yb[i]=p[0]+p[1]*yb[i-1]+ee;}
  double zz[5];copy(z,z+5,zz);int ok=optimize(yb.data(),n,zz);if(!ok){init(yb.data(),n,zz);ok=optimize(yb.data(),n,zz,400);}
  double gg[5],last;eval(zz,yb.data(),n,gg,&last);out[k]=(ok&&isfinite(last))?zz[0]/sqrt(last):nan("");valid+=isfinite(out[k]);
 }
 return valid;
}
