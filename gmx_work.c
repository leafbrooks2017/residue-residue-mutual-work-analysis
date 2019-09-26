/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014,2015, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
#include "gmxpre.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "gromacs/commandline/pargs.h"
#include "gromacs/fileio/confio.h"
#include "gromacs/fileio/tpxio.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/fileio/xvgr.h"
#include "gromacs/gmxana/gmx_ana.h"
#include "gromacs/legacyheaders/macros.h"
#include "gromacs/legacyheaders/typedefs.h"
#include "gromacs/legacyheaders/viewit.h"
#include "gromacs/linearalgebra/nrjac.h"
#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/pbcutil/rmpbc.h"
#include "gromacs/topology/index.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/smalloc.h"

#include "gromacs/pbcutil/pbc.h"
#include "gromacs/listed-forces/bonded.h"
#include "gromacs/legacyheaders/types/forcerec.h"

/*coded by Wenjin Li on April 28th 2016*/
#define HMASS 1.1

#define snew_mat(m,N,M,i) snew(m,N);for(i=0;i<N;i++)\
        snew(m[i],M);
#define sfree_mat(m,N,i) for(i=0;i<N;i++)\
        sfree(m[i]);sfree(m);

typedef struct {
  int   ncolB;  /* the number of non-zero component in each colB */
  int   *indB;  /* the index of the non-zero component in the B-matrix */
  real  *colB; /* the non-zero column component of B-matrix */
  real  **matB; /* the non-zero matrix component of secondary B-matrix */
} t_Bmat;

static void clear_matrix(real **m,int M,int N)
{
  int i,j;
  for(i=0;i<M;i++)
    for(j=0;j<N;j++)
      m[i][j]=0.0;
}

static void sub_allowMem_Bmat(t_Bmat *Bmat)
{
  int i;
  snew(Bmat->indB,Bmat->ncolB);
  snew(Bmat->colB,Bmat->ncolB*3);
  snew_mat(Bmat->matB,Bmat->ncolB*3,Bmat->ncolB*3,i);
}

static void allowMem_Bmat(t_Bmat *Bmat,int M)
{
  int ii;

  /* sign ncolB */
  for(ii=0;ii<M;ii++){
    Bmat[3*ii].ncolB=2;
    Bmat[3*ii+1].ncolB=3;
    Bmat[3*ii+2].ncolB=4;
  }

  for(ii=0;ii<3*M;ii++){
    sub_allowMem_Bmat(&Bmat[ii]);
  }
}

static void sub_clean_Bmats(t_Bmat *Bmat)
{
  int i;
  for(i=0;i<Bmat->ncolB*3;i++) Bmat->colB[i]=0.0;
  clear_matrix(Bmat->matB,Bmat->ncolB*3,Bmat->ncolB*3);
}

static void clean_Bmats(t_Bmat *Bmat, int N)
{
  int ii;

  for(ii=0;ii<N;ii++){
    sub_clean_Bmats(&Bmat[ii]);
  }
}

static void sub_freeMem_Bmat(t_Bmat *Bmat)
{
  int i;
  sfree(Bmat->indB);
  sfree(Bmat->colB);
  sfree_mat(Bmat->matB,Bmat->ncolB*3,i);
}

static void freeMem_Bmat(t_Bmat *Bmat,int M)
{
  int ii;

  for(ii=0;ii<3*M;ii++){
    sub_freeMem_Bmat(&Bmat[ii]);
  }
}

/* void rvec_sub(const rvec a,const rvec b,rvec c)  c = a - b */
static int pbc_rvec_sub(const t_pbc *pbc,const rvec xi,const rvec xj,rvec dx)
{
  if (pbc) {
    return pbc_dx_aiuc(pbc,xi,xj,dx);
  }
  else {
    rvec_sub(xi,xj,dx);
    return 0;
  }
}

static void print_data(FILE *fp,real time,real SumWork)
{
  fprintf(fp, " %g\t%g\n", time, SumWork);
}

static void vec2real3(real *fvec, real **mat, int M, int N)
{
  int j,k=0,m;

  for(j=0;j<M;j++)
    for(m=0;m<N;m++)
      fvec[k++]=mat[j][m];
}

static void vec2real2(real *fvec, rvec *dw, int M, int N)
{
  int j,k=0,m;

  for(j=0;j<M;j++)
    for(m=0;m<N;m++)
      fvec[k++]=dw[j][m];
}

static void writeBin2_nvec(FILE *fp, rvec *dw, int M, int N, real *fvec)
{
  vec2real2(fvec,dw,M,N);
  fwrite(fvec, sizeof(real),M*N, fp);
}

static void writeBin_nvec(FILE *fp, rvec *dw, int M, int N)
{
  real *fvec;

  //fwrite(&M, sizeof(int), 1, fp);
  //fwrite(&N, sizeof(int), 1, fp);

  snew(fvec,M*N);
  vec2real2(fvec,dw,M,N);
  fwrite(fvec, sizeof(real),M*N, fp);
  sfree(fvec);
}

static void print_data_nrvec(FILE *fp,real time,rvec *vec,int nset)
{
  int i,d;

  fprintf(fp," %g",time);
  for(i=0;i<nset;i++)
    for(d=0;d<DIM;d++)
      fprintf(fp, "\t%g", vec[i][d]);
  fprintf(fp,"\n");
}

/* copy coordinate (X), force (F) and box (B) from frame information (fr)*/
static void copy_XFB(t_trxframe *fr, rvec xx[], rvec ff[], matrix box)
{   
  int i;
  for(i=0;i<fr->natoms;i++){
    copy_rvec(fr->x[i],xx[i]);
    copy_rvec(fr->f[i],ff[i]);
  }
  copy_mat(fr->box,box);
}

static void ncopy_real2vec(real *fvec,rvec *f,int n)
{
  int i,k=0,m;
  for(i=0;i<n;i++){
    for(m=0;m<DIM;m++)
      f[i][m]=fvec[k++];
  }
}

static void ncopy_vec2real(rvec *f,real *fvec,int n)
{
  int i,k=0,m;
  for(i=0;i<n;i++){
    for(m=0;m<DIM;m++)
      fvec[k++]=f[i][m];
  }
}

static void copy_nrvec(rvec *v1,rvec *v2,int n)
{
  int i;
  for(i=0;i<n;i++) copy_rvec(v1[i],v2[i]);
}

static void update_delta_q(rvec *dq, rvec xcom,rvec xcom_old,t_pbc *pbc)
{
  int t1;
  t1=pbc_rvec_sub(pbc,xcom,xcom_old,dq[0]);
}

static void copy_prevX(rvec *x,rvec *x_old,BKStree_t *tr)
{
  int i,alpha;
  int M=tr->N;
  for(i=0;i<M;i++){
    alpha=tr->nodes[i].iAtom;
    copy_rvec(x[alpha],x_old[i]);
  }
}

static void update_ext_intQ(rvec *Qvec,rvec xcom,rvec rdr,rvec *x,rvec *x_old,rvec *xp,BKStree_t *tr,t_topology *top)
{
  int ii, alpha;
  real m0;
  rvec dx,tmp;

  copy_rvec(xcom,Qvec[0]); /* update the com part */

  /* update the rotatiional part*/
  for(ii=0;ii<(tr->N);ii++){
    alpha=tr->nodes[ii].iAtom;
    m0=top->atoms.atom[alpha].m;
    rvec_sub(x[alpha],x_old[ii],dx); /* not consider periodic condition */
    cprod(xp[ii],dx,tmp);
    svmul(m0,tmp,tmp);
    rvec_inc(rdr,tmp);
  }

  Qvec[2][2]=rdr[0];
  Qvec[1][1]=rdr[1];
  Qvec[1][2]=rdr[2];
}

static void update_intQ(rvec *Qvec,rvec xcom,rvec rdr)
{
  copy_rvec(xcom,Qvec[0]);
  clear_rvec(rdr);
  Qvec[2][2]=rdr[0];
  Qvec[1][1]=rdr[1];
  Qvec[1][2]=rdr[2];
}

static void update_geneF(rvec *intQvec,rvec totF, rvec tau)
{
  copy_rvec(totF,intQvec[0]);
  intQvec[2][2]=tau[0];
  intQvec[1][1]=tau[1];
  intQvec[1][2]=tau[2];
}

static int S2T(int ii, gmx_bool bSubTree,int *subtree)
{
  if(bSubTree) return subtree[ii];
  else return ii;
}

static int T2S(int ii, gmx_bool bSubTree,int *subtree)
{

  if(bSubTree) return subtree[ii];
  else return ii;
}

static void read_bonds_from_ilist(t_functype *functype,t_ilist *ilist, int **bondMatrix)
{
    int i,j,k,type,ftype;
    t_iatom *iatoms;
    int a[2];

    a[0]=0; a[1]=0;
    if ((ilist!=NULL) && ilist->nr > 0)
    {
        iatoms=ilist->iatoms;
        for (i=j=0; i<ilist->nr;) {
          type=*(iatoms++);
          ftype=functype[type];
	  j++;
	  if(!(IS_CHEMBOND(ftype)))
	    gmx_fatal(FARGS,"Topology error: Contain non-bond type in bond type");
          for (k=0; k<interaction_function[ftype].nratoms; k++)
              a[k]=*(iatoms++);
	  bondMatrix[a[0]][a[1]]++;
          bondMatrix[a[1]][a[0]]++;
          i+=1+interaction_function[ftype].nratoms;
        }
    }
}

static void read_bonds_from_ilist_new(t_functype *functype,t_ilist *ilist, int **bondList)
{
    int i,k,type,ftype;
    t_iatom *iatoms;
    int a[2];
    int ind=0;

    a[0]=0; a[1]=0;
    if ((ilist!=NULL) && ilist->nr > 0)
    {
        iatoms=ilist->iatoms;
        for (i=0; i<ilist->nr;) {
          type=*(iatoms++);
          ftype=functype[type];
          if(!(IS_CHEMBOND(ftype)))
            gmx_fatal(FARGS,"Topology error: Contain non-bond type in bond type");
          for (k=0; k<interaction_function[ftype].nratoms; k++)
              a[k]=*(iatoms++);
	  bondList[0][ind]=a[0];
	  bondList[1][ind]=a[1];
          ind++;
          i+=1+interaction_function[ftype].nratoms;
        }
    }
}

static void read_bonds_from_index(int *bondIndex, int totBonds, int **bondList){
   int i;
   for(i=0;i<(totBonds/2);i++){
      bondList[0][i]=bondIndex[2*i];
      bondList[1][i]=bondIndex[2*i+1];
   }
}

static int count_edges_from_bondMat(int *subtree, int treesize, int **bondMatrix){
  int i,j;
  int count=0;
  for(i=0;i<treesize;i++){
    for(j=i+1;j<treesize;j++)
      if(bondMatrix[subtree[i]][subtree[j]]>=1) count++;
  }
  return count;
}

static int count_edges_from_bondList(int *sT2, int treesize, int **bondList, int nbonds){
  int i,j;
  int count=0;
  for(i=0;i<nbonds;i++){
      if((sT2[bondList[0][i]]<treesize) && (sT2[bondList[1][i]]<treesize)) count++;
  }
  return count;
}

static void get_nEdges(int *nEdges,int **bondList,int nbonds,gmx_bool bST,int *sT2,int treesize){
  int i,j;
  int a0,a1;
  for(i=0;i<nbonds;i++){
    a0=S2T(bondList[0][i],bST,sT2);
    a1=S2T(bondList[1][i],bST,sT2);
    if((a0<treesize) && (a1<treesize)){
      nEdges[a0]++; nEdges[a1]++; 
    }
  }
}

static void get_nEdges_tr(BKStree_t *tr,int **BKSedge){
  int i,j;
  int a0,a1;
  for(i=0;i<tr->N;i++){
    tr->nodes[i].nEdges=0;
  }

  for(i=0;i<tr->nEdges;i++){
    a0=BKSedge[0][i];
    a1=BKSedge[1][i];
    tr->nodes[a0].nEdges++;
    tr->nodes[a1].nEdges++;
  }
}

static void get_edgeEle(int **edgeEle,int *nEdges,int **bondList,int nbonds,gmx_bool bST,int *sT2,int treesize){
  int i,j;
  int a0,a1;
  for(i=0;i<nbonds;i++){
       a0=S2T(bondList[0][i],bST,sT2);
       a1=S2T(bondList[1][i],bST,sT2);
      if((a0<treesize) && (a1<treesize)){
        edgeEle[a0][nEdges[a0]]=a1; edgeEle[a1][nEdges[a1]]=a0; 
	nEdges[a0]++; nEdges[a1]++;
      }
  }
}

/* check whether there is a edge between ii and jj, here ii (jj) is BKS-tree index */
static gmx_bool isEdge(int ii,int jj,int **edgeEle,int *nEdges,int **edgeLabel){
   int kk;
   for(kk=0;kk<nEdges[ii];kk++){
     if((edgeEle[ii][kk]==jj) && edgeLabel[ii][kk]==1) return TRUE;
   }
   return FALSE;
}

static int delEdge(int ii,int jj,int **edgeEle,int *nEdges,int **edgeLabel){
   int kk;
   for(kk=0;kk<nEdges[ii];kk++){
     if(edgeEle[ii][kk]==jj){ edgeLabel[ii][kk]=0; return 1; }
   }
   
   gmx_fatal(FARGS,"No such edges, check code!");
   return 0;
}

static void pr_real_matrix(real **mat, int nrow, int ncol){
  int ii, jj;
  for (ii=0;ii<nrow;ii++){
    for (jj=0;jj<ncol;jj++)
      fprintf(stderr,"%g ",mat[ii][jj]);
    fprintf(stderr,"\n");
  }
}

static void pr_int_matrix(int **mat, int nrow, int ncol){
  int ii, jj;
  for (ii=0;ii<nrow;ii++){
    for (jj=0;jj<ncol;jj++)
      fprintf(stderr,"%d ",mat[ii][jj]);
    fprintf(stderr,"\n");
  }
}

static void pr_bondList(int **bondList,int nbonds){
  int ii, jj;
  fprintf(stderr,"Bond List:\n");
  for (ii=0;ii<nbonds;ii++){
    for (jj=0;jj<2;jj++)
      fprintf(stderr,"%d ",bondList[jj][ii]+1);
    fprintf(stderr,"\n");
  }
}

static real cal_kin(int start,int homenr,t_topology *top,rvec v[],gmx_bool bST, int *sT)
{
   int i,k,m;
   real m0, newekin;
   newekin=0.0;
   for (k=start; k<(start+homenr); k++){
        i=T2S(k,bST,sT);
        m0=top->atoms.atom[i].m;
      for (m=0; (m<DIM); m++) {
        newekin+=m0*v[i][m]*v[i][m];
      }
   }
   return 0.5*newekin;
}

static void matrix_transpose(real **m, int N, int M, real **minv)
{
  int i,j;
  for(i=0;i<N;i++){
    for(j=0;j<M;j++)
      minv[j][i]=m[i][j];
  }
}

static void matrix_multi(real **m1, real **m2, int N, int M, int L, real **mat)
{
  int i,j,k;
  real tmp;
  for(i=0;i<N;i++){
    for(j=0;j<L;j++){
      tmp=0;
      for(k=0;k<M;k++)
        tmp+=m1[i][k]*m2[k][j];
      mat[i][j]=tmp;
    }
  }
}

static void copy_matrix(real **m, real **m2, int N, int M)
{
  int i,j;
  for(i=0;i<N;i++)
    for(j=0;j<M;j++)
      m2[i][j]=m[i][j];
}

static void copy_Angular(real **m1,rvec v1,rvec v2,rvec v3,rvec v4,rvec v5,rvec v6,rvec v7,real **m2,rvec k1,rvec k2,rvec k3,rvec k4,rvec k5,rvec k6,rvec k7)
{
  copy_rvec(v1,k1);
  copy_rvec(v2,k2);
  copy_rvec(v3,k3);
  copy_rvec(v4,k4);
  copy_rvec(v5,k5);
  copy_rvec(v6,k6);
  copy_rvec(v7,k7);
  copy_matrix(m1,m2,3,3);
}

/* here we do not consider the periodic boundary */
static void cal_com_simple(rvec *x,t_topology *top,rvec com,int M,BKStree_t *tr)
{
  int i,j,alpha,d;
  real mass=0.0;
  for(d=0;d<DIM;d++) com[d]=0.0;
  for(i=0;i<M;i++){
    alpha=tr->nodes[i].iAtom;
    mass+=top->atoms.atom[alpha].m;
    for(d=0;d<DIM;d++)
      com[d]+=(top->atoms.atom[alpha].m*x[alpha][d]);
  }
  for(d=0;d<DIM;d++)
    com[d]=com[d]/mass;
}

static void total_F(BKStree_t *tr,rvec *f)
{
  int i,d;
  rvec totF;
  clear_rvec(totF);
  for(i=0;i<tr->N;i++){
    for(d=0;d<DIM;d++)
      totF[d]+=f[tr->nodes[i].iAtom][d];
  }
  fprintf(stderr,"Total Force: %g %g %g\n",totF[XX],totF[YY],totF[ZZ]);
}

static real det3(real **a)
{
    return ( a[XX][XX]*(a[YY][YY]*a[ZZ][ZZ]-a[ZZ][YY]*a[YY][ZZ])
             -a[YY][XX]*(a[XX][YY]*a[ZZ][ZZ]-a[ZZ][YY]*a[XX][ZZ])
             +a[ZZ][XX]*(a[XX][YY]*a[YY][ZZ]-a[YY][YY]*a[XX][ZZ]));
}

/* m3_inv() is an alternative function to get the inverse of a 3*3 matrix, e.g., tensor
   It is adapted from m_inv in vec.h
*/
static void m3_inv(real **src, real **dest)
{
    const real smallreal = (real)1.0e-24;
    const real largereal = (real)1.0e24;
    real       deter, c, fc;

    deter = det3(src);
    c     = (real)1.0/deter;
    fc    = (real)fabs(c);

    if ((fc <= smallreal) || (fc >= largereal))
    {
        gmx_fatal(FARGS, "Can not invert matrix, determinant = %e", deter);
    }

    dest[XX][XX] = c*(src[YY][YY]*src[ZZ][ZZ]-src[ZZ][YY]*src[YY][ZZ]);
    dest[XX][YY] = -c*(src[XX][YY]*src[ZZ][ZZ]-src[ZZ][YY]*src[XX][ZZ]);
    dest[XX][ZZ] = c*(src[XX][YY]*src[YY][ZZ]-src[YY][YY]*src[XX][ZZ]);
    dest[YY][XX] = -c*(src[YY][XX]*src[ZZ][ZZ]-src[ZZ][XX]*src[YY][ZZ]);
    dest[YY][YY] = c*(src[XX][XX]*src[ZZ][ZZ]-src[ZZ][XX]*src[XX][ZZ]);
    dest[YY][ZZ] = -c*(src[XX][XX]*src[YY][ZZ]-src[YY][XX]*src[XX][ZZ]);
    dest[ZZ][XX] = c*(src[YY][XX]*src[ZZ][YY]-src[ZZ][XX]*src[YY][YY]);
    dest[ZZ][YY] = -c*(src[XX][XX]*src[ZZ][YY]-src[ZZ][XX]*src[XX][YY]);
    dest[ZZ][ZZ] = c*(src[XX][XX]*src[YY][YY]-src[YY][XX]*src[XX][YY]);
}

static void outer_prod(rvec v1,rvec v2,matrix m)
{
  int d;
  for(d=0;d<DIM;d++)
    svmul(v1[d],v2,m[d]);
}

static void outer_prod2(rvec v1,rvec v2,real **m)
{
  int d;
  for(d=0;d<DIM;d++)
    svmul(v1[d],v2,m[d]);
}

static void cal_totF(BKStree_t *tr,rvec *f,rvec totF)
{
  int i,alpha;
  int M=tr->N;
  clear_rvec(totF);
  for(i=0;i<M;i++){
    alpha=tr->nodes[i].iAtom;
    rvec_inc(totF,f[alpha]);
  }
}

static void cal_totF2(BKStree_t *tr,rvec *f,rvec totF)
{
  int i;
  int M=tr->N;
  clear_rvec(totF);
  for(i=0;i<M;i++){
    rvec_inc(totF,f[i]);
  }
}

static void mat_mul_vec(real **a,rvec src,rvec dest)
{
  dest[XX]=a[XX][XX]*src[XX]+a[XX][YY]*src[YY]+a[XX][ZZ]*src[ZZ];
  dest[YY]=a[YY][XX]*src[XX]+a[YY][YY]*src[YY]+a[YY][ZZ]*src[ZZ];
  dest[ZZ]=a[ZZ][XX]*src[XX]+a[ZZ][YY]*src[YY]+a[ZZ][ZZ]*src[ZZ];
}

static void cal_torque(rvec tau,rvec *x,rvec *f,BKStree_t *tr)
{
  int i,alpha;
  int M=tr->N;
  rvec tmp;
  clear_rvec(tau);
  for(i=0;i<M;i++){
    alpha=tr->nodes[i].iAtom;
    cprod(x[i],f[alpha],tmp);
    rvec_inc(tau,tmp);
  }
}

static void cal_torque2(rvec tau,rvec *x,rvec *f,BKStree_t *tr)
{
  int i;
  int M=tr->N;
  rvec tmp;
  clear_rvec(tau);
  for(i=0;i<M;i++){
    cprod(x[i],f[i],tmp);
    rvec_inc(tau,tmp);
  }
}

/* Here, xp is the coordinate relative to COM */
static void cal_tensor_mat(real **inertia,rvec *xp,t_topology *top,BKStree_t *tr)
{
  int i,d,alpha;
  int M=tr->N;
  real m0,r2;
  rvec tmp;
  matrix mat;
  real imat[3][3]={{1,0,0},
             {0,1,0},
             {0,0,1}};
  clear_matrix(inertia,3,3); 

  for(i=0;i<M;i++){
    alpha=tr->nodes[i].iAtom;
    m0=top->atoms.atom[alpha].m;
    r2=iprod(xp[i],xp[i]);
    outer_prod(xp[i],xp[i],mat);
    for(d=0;d<DIM;d++){
      svmul(r2,imat[d],tmp);
      rvec_dec(tmp,mat[d]);
      svmul(m0,tmp,tmp);
      rvec_inc(inertia[d],tmp);
    }  
  }
}

static void cal_geneF_rdr(rvec tau,real **inv_inert)
{
  int i,d;
  rvec tvec;

  /* tau*(I^-1) */
  clear_rvec(tvec);
  for(i=0;i<DIM;i++){
    for(d=0;d<DIM;d++){
      tvec[i]+=tau[d]*inv_inert[d][i];
    }
  }

  copy_rvec(tvec,tau);

}

/* WARNING: no periodic boundary is considered */
static void remove_com_x(t_topology *top,BKStree_t *tr,rvec *x,rvec *x1,rvec xcom)
{
  int i,alpha;
  int M=tr->N;
  cal_com_simple(x,top,xcom,M,tr);
  for(i=0;i<M;i++){
    alpha=tr->nodes[i].iAtom;
    rvec_sub(x[alpha],xcom,x1[i]); /* WARNING: no periodic boundary is considered */
  }
}

static real dotp(real *vec1, real *vec2, int n)
{
  int i;
  real dotp=0.0;

  for(i=0;i<n;i++)
    dotp+=vec1[i]*vec2[i];

  return dotp;
}

static void copy_int_mat_l(int **mat, int **cmat, int nrow, int ncol, gmx_bool bST, int *sT2){
  int ii, jj;
  for (ii=0;ii<nrow;ii++){
    for (jj=0;jj<ncol;jj++)
      cmat[S2T(ii,bST,sT2)][S2T(jj,bST,sT2)]=mat[ii][jj];
  }
}

static void pr_tree(BKStree_t *tr)
{
  int i,j;
  
  fprintf(stderr,"Tree info: i, iAtom, nEdges, iChain\n");
  for(i=0;i<tr->N;i++){
    fprintf(stderr,"Node %d : %d %d %d\n",i,tr->nodes[i].iAtom,tr->nodes[i].nEdges, tr->nodes[i].iChain);
  }
  fprintf(stderr,"Tree info: i, nEle: chainEle\n");
  for(i=0;i<tr->N;i++){
    fprintf(stderr,"Node %d : %d -- ",i,tr->nodes[i].nEle);
    for(j=0;j<tr->nodes[i].nEle;j++)
      fprintf(stderr," %d",tr->nodes[i].ele[j]);
    fprintf(stderr,"\n");
  }
  fprintf(stderr,"Tree info: i, nEle: ijkl, names\n");
  for(i=0;i<tr->N;i++){
    fprintf(stderr,"Node %d : %d -- ",i,tr->nodes[i].nEle);
    for(j=0;j<4;j++)
      fprintf(stderr," %d",tr->nodes[i].ijkl[j]);
    fprintf(stderr," ::Name:: %s %s %s\n",tr->nodes[i].nameXX,tr->nodes[i].nameYY,tr->nodes[i].nameZZ);
  }
  fprintf(stderr,"Tree info: i, ndrdq: drdqEle\n");
  for(i=0;i<tr->N;i++){
    fprintf(stderr,"Node %d : %d -- ",i,tr->nodes[i].ndrdq);
    for(j=0;j<tr->nodes[i].ndrdq;j++)
      fprintf(stderr," %d",tr->nodes[i].drdqEle[j]);
    fprintf(stderr,"\n");
  }
}

static void print_coor_name(FILE *fp, BKStree_t *tr){
  int i;
  for(i=0;i<tr->N;i++){
    fprintf(fp,"%s %s %s\n",tr->nodes[i].nameXX,tr->nodes[i].nameYY,tr->nodes[i].nameZZ);
  }
}

static void sort_index(int *index, int *buf, int n)
{
  int i,j;
  int tmp;
  for(i=0;i<n;i++)
    buf[i]=index[i];
  for(i=0;i<n;i++)
    for(j=1;j<n;j++){
      if(buf[j-1]>buf[j]){
        tmp=buf[j-1];
        buf[j-1]=buf[j];
        buf[j]=tmp;
      }
    }
}

static void reverse_index(int *index, int *buf, int n)
{
  int i,j;
  int tmp;

  if(index[0]<index[n-1]){
    for(i=0;i<n;i++)
      buf[i]=index[i];
  }else{
    for(i=0;i<n;i++)
      buf[i]=index[n-1-i];
  }
}

static void get_name(BKStree_t *tr, int i)
{
  int n,ni,nj,nk,nl;
  BKSnode_t *node;
  int ijkl[4];

  node=&tr->nodes[i];
  n=node->nEle;
  if(n==1){
    sprintf(node->nameXX,"bon_%d_O0.xvg",node->ijkl[0]);
    sprintf(node->nameYY,"ang_%d_O0_z0.xvg",node->ijkl[0]);
    sprintf(node->nameZZ,"dih_%d_O0_z0_x0.xvg",node->ijkl[0]);
  }
  if(n==2){
    //sort_index(node->ijkl,ijkl,n)
    reverse_index(node->ijkl,ijkl,n);
    sprintf(node->nameXX,"bon_%d_%d.xvg",ijkl[0],ijkl[1]);
    sprintf(node->nameYY,"ang_%d_%d_O0.xvg",ijkl[0],ijkl[1]);
    sprintf(node->nameZZ,"dih_%d_%d_O0_z0.xvg",ijkl[0],ijkl[1]);
  }
  if(n==3){
    //sort_index(node->ijkl,ijkl,2);
    reverse_index(node->ijkl,ijkl,2);
    sprintf(node->nameXX,"bon_%d_%d.xvg",ijkl[0],ijkl[1]);
    //sort_index(node->ijkl,ijkl,n);
    reverse_index(node->ijkl,ijkl,n);
    sprintf(node->nameYY,"ang_%d_%d_%d.xvg",ijkl[0],ijkl[1],ijkl[2]);
    nj=node->ele[n-2];
    if((tr->nodes[nj].nEdges>2) && (node->ele[n-1] != (nj+1))){
      reverse_index(node->ijkl,ijkl,4);
      sprintf(node->nameZZ,"dih_%d_%d_%d_%d.xvg",ijkl[0],ijkl[1],ijkl[2],ijkl[3]);
    }else{
      sprintf(node->nameZZ,"dih_%d_%d_%d_O0.xvg",ijkl[0],ijkl[1],ijkl[2]);
    }
  }
  if(n>=4){
    //sort_index(node->ijkl,ijkl,2);
    reverse_index(node->ijkl,ijkl,2);
    sprintf(node->nameXX,"bon_%d_%d.xvg",ijkl[0],ijkl[1]);
    //sort_index(node->ijkl,ijkl,3);
    reverse_index(node->ijkl,ijkl,3);
    sprintf(node->nameYY,"ang_%d_%d_%d.xvg",ijkl[0],ijkl[1],ijkl[2]);
    //sort_index(node->ijkl,ijkl,4);
    reverse_index(node->ijkl,ijkl,4);
    sprintf(node->nameZZ,"dih_%d_%d_%d_%d.xvg",ijkl[0],ijkl[1],ijkl[2],ijkl[3]);
  }
}

static void get_index_ijkl(BKStree_t *tr, int i, int *index)
{
  int n,ni,nj,nk,nl;
  BKSnode_t *node;

  node=&tr->nodes[i];
  n=node->nEle;
  if(n==1){
    ni=node->ele[n-1];
    index[0]=tr->nodes[ni].iAtom;
  }
  if(n==2){
    ni=node->ele[n-1];
    nj=node->ele[n-2];
    index[0]=tr->nodes[ni].iAtom;
    index[1]=tr->nodes[nj].iAtom;
  }
  if(n==3){
    ni=node->ele[n-1];
    nj=node->ele[n-2];
    nk=node->ele[n-3];
    index[0]=tr->nodes[ni].iAtom;
    index[1]=tr->nodes[nj].iAtom;
    index[2]=tr->nodes[nk].iAtom;
    if((tr->nodes[nj].nEdges>2) && (ni != (nj+1))){
      index[3]=tr->nodes[nj+1].iAtom;
    }
  }
  if(n>=4){
    ni=node->ele[n-1];
    nj=node->ele[n-2];
    nk=node->ele[n-3];
    nl=node->ele[n-4];
    index[0]=tr->nodes[ni].iAtom;
    index[1]=tr->nodes[nj].iAtom;
    index[2]=tr->nodes[nk].iAtom;
    /*for nodes j have more than 2 connections, the dihideral is calculated by take the first dihedral as the reference.*/
    if((tr->nodes[nj].nEdges>2) && (ni != (nj+1))){
      index[3]=tr->nodes[nj+1].iAtom;
    }else{
      index[3]=tr->nodes[nl].iAtom;
    }
  }
}

/* The dihidral angle is calculated from the same jkl for an atom have more than two connections */
static void get_index_ijkl_old(BKStree_t *tr, int i, int *index)
{
  int n,ni,nj,nk,nl;
  BKSnode_t *node;

  node=&tr->nodes[i];
  n=node->nEle;
  if(n==1){
    ni=node->ele[n-1];
    index[0]=tr->nodes[ni].iAtom;
  }
  if(n==2){
    ni=node->ele[n-1];
    nj=node->ele[n-2];
    index[0]=tr->nodes[ni].iAtom;
    index[1]=tr->nodes[nj].iAtom;
  }
  if(n==3){
    ni=node->ele[n-1];
    nj=node->ele[n-2];
    nk=node->ele[n-3];
    index[0]=tr->nodes[ni].iAtom;
    index[1]=tr->nodes[nj].iAtom;
    index[2]=tr->nodes[nk].iAtom;
  }
  if(n>=4){
    ni=node->ele[n-1];
    nj=node->ele[n-2];
    nk=node->ele[n-3];
    nl=node->ele[n-4];
    index[0]=tr->nodes[ni].iAtom;
    index[1]=tr->nodes[nj].iAtom;
    index[2]=tr->nodes[nk].iAtom;
    index[3]=tr->nodes[nl].iAtom;
  }
}

static gmx_bool bInBackbone_old(int backbone[], int jj, int n)
{
  int i;

  for(i=0;i<n;i++){
    if(backbone[i]==jj)
      return TRUE;
  }
  return FALSE;
}

static void update_names(BKStree_t *tr)
{
  sprintf(tr->nodes[0].nameXX,"com_XX.xvg");
  sprintf(tr->nodes[0].nameYY,"com_YY.xvg");
  sprintf(tr->nodes[0].nameZZ,"com_ZZ.xvg");
  sprintf(tr->nodes[2].nameZZ,"rdr_XX.xvg");
  sprintf(tr->nodes[1].nameYY,"rdr_YY.xvg");
  sprintf(tr->nodes[1].nameZZ,"rdr_ZZ.xvg");
}

static void initial_BKStree(BKStree_t *tr, t_topology *top, int base2, gmx_bool bHeavy, gmx_bool bBackbone, int *backbone,
                     gmx_bool bST, int *sT, int tn, gmx_bool bBondList, int *bondIndex, int totBonds)
{
  //int **bondMatrix, **cmat, **BKSedgeMat;
  int **bondList, **BKSedge, **edgeLabel;
  int ii, jj, idepth, kk, nn;
  int *buf;
  int nTotEdges=0, chainLen;
  gmx_bool bEdge;
  int *nEdges, **edgeEle, *bEdges;
  int base,nbonds,indEdge;
  int *sT2=NULL;

  /*extract the bond information from topology and save as connection matrix*/
  if(bBondList){ nbonds=totBonds/2; }else{ nbonds=top->idef.il[0].nr/3; }
  snew(bondList,2);
  for (ii=0;ii<2;ii++){
    snew(bondList[ii], nbonds);
  }

  if(bBondList){
    read_bonds_from_index(bondIndex, totBonds, bondList);
  }else{
    read_bonds_from_ilist_new(top->idef.functype, &top->idef.il[0], bondList);
  }
  //pr_bondList(bondList,nbonds);

  /* add or delete bond to form a BKS-tree */

  /*construct the BKStree and initialize the tree*/  
  if(bST){ /* initial subtree*/
    tr->N=tn;
    /* contruct the transition between atoms in subtree and in system */
    snew(sT2,top->atoms.nr);
    for(ii=0;ii<top->atoms.nr;ii++){ sT2[ii]=top->atoms.nr+1; }
    for(ii=0;ii<tr->N;ii++){
      sT2[sT[ii]]=ii;
    }

    tr->nEdges=count_edges_from_bondList(sT2,tn,bondList,nbonds);
  }else{
    tr->N=top->atoms.nr;
    tr->nEdges=top->idef.il[0].nr/3;
  }
  if(tr->N < (tr->nEdges+1))
    gmx_fatal(FARGS,"Cyclic system detected! Only applicable to non-cyclic system: Treeszie %d Edges %d ",tr->N,tr->nEdges+1);
  if(tr->N > (tr->nEdges+1))
    gmx_fatal(FARGS,"Too few bonds. The system may have more than one molecules. Define the connection between them or define subtree: Treeszie %d Edges %d ",tr->N,tr->nEdges+1);

  snew(tr->nodes,tr->N);
  snew(tr->edges, tr->nEdges);
  base=S2T(base2,bST,sT2);
  tr->nodes[0].iAtom=base;
  //snew(tr->nodes[0].edgeEle,1);

  snew(BKSedge,2);
  for (ii=0;ii<2;ii++){
    snew(BKSedge[ii], tr->nEdges);
  }

  snew(buf,tr->N);
  snew(nEdges,tr->N);
  snew(edgeEle,tr->N);
  snew(bEdges,tr->N);

  /* label the edge in edgeEle as an edge or not */
  snew(edgeLabel,tr->N);

  for (ii=0;ii<tr->N;ii++){ bEdges[ii]=0; nEdges[ii]=0; }

  /*get the number of edeges connect to an atom and the elelent of those atoms */
  get_nEdges(nEdges,bondList,nbonds,bST,sT2,tr->N);
  for (ii=0;ii<tr->N;ii++){
    snew(edgeEle[ii],nEdges[ii]);
    snew(edgeLabel[ii],nEdges[ii]);
    for (jj=0;jj<nEdges[ii];jj++){ edgeLabel[ii][jj]=1; }
    nEdges[ii]=0;
  }

  get_edgeEle(edgeEle,nEdges,bondList,nbonds,bST,sT2,tr->N);
 
  if(nEdges[base]!=1)
    gmx_fatal(FARGS,"Base have %d bonds, Choose as base an atom that has only one bond",nEdges[base]);

  bEdges[base]=1;
  indEdge=0;
  for (idepth=ii=1;ii<tr->N;){
    bEdge=FALSE;
    /*for atoms have more than 2 edges, choose the heavy atom as the first leaf if it has one*/
    if((bHeavy || bBackbone) && (nEdges[tr->nodes[idepth-1].iAtom]>2) && (bEdges[tr->nodes[idepth-1].iAtom]==0)){
      bEdges[tr->nodes[idepth-1].iAtom]=1;
      if(bBackbone){
        for(jj=0;jj<tr->N;jj++){
          //if((cmat[tr->nodes[idepth-1].iAtom][jj]==1) && bInBackbone(backbone,T2S(jj,bST,sT),N)){
            if(isEdge(tr->nodes[idepth-1].iAtom,jj,edgeEle,nEdges,edgeLabel) && backbone[T2S(jj,bST,sT)]){
            tr->nodes[ii].iAtom=jj;
            BKSedge[0][indEdge]=idepth-1; /*sign the connection matrix with the index of nodes in BKS tree*/
            BKSedge[1][indEdge]=ii;
            indEdge++;
            delEdge(tr->nodes[idepth-1].iAtom,jj,edgeEle,nEdges,edgeLabel);
            delEdge(jj,tr->nodes[idepth-1].iAtom,edgeEle,nEdges,edgeLabel);

            ii++;
            idepth=ii;
            bEdge=TRUE;
            break;
          }
        }
      }
      if(!bEdge && bHeavy){
        for(jj=0;jj<tr->N;jj++){
          //if((cmat[tr->nodes[idepth-1].iAtom][jj]==1) && (top->atoms.atom[T2S(jj,bST,sT)].m>HMASS)){
          if(isEdge(tr->nodes[idepth-1].iAtom,jj,edgeEle,nEdges,edgeLabel) && (top->atoms.atom[T2S(jj,bST,sT)].m>HMASS)){
            tr->nodes[ii].iAtom=jj;
            BKSedge[0][indEdge]=idepth-1; /*sign the connection matrix with the index of nodes in BKS tree*/
            BKSedge[1][indEdge]=ii;
            indEdge++;
            delEdge(tr->nodes[idepth-1].iAtom,jj,edgeEle,nEdges,edgeLabel);
            delEdge(jj,tr->nodes[idepth-1].iAtom,edgeEle,nEdges,edgeLabel);

            ii++;
            idepth=ii;
            bEdge=TRUE;
            break;
          }
        }
      }
    }
    if(!bEdge){
      for(jj=0;jj<tr->N;jj++){
        //if(cmat[tr->nodes[idepth-1].iAtom][jj]==1){
        if(isEdge(tr->nodes[idepth-1].iAtom,jj,edgeEle,nEdges,edgeLabel)){
          tr->nodes[ii].iAtom=jj;
          BKSedge[0][indEdge]=idepth-1; /*sign the connection matrix with the index of nodes in BKS tree*/
          BKSedge[1][indEdge]=ii;
          indEdge++;
          delEdge(tr->nodes[idepth-1].iAtom,jj,edgeEle,nEdges,edgeLabel);
          delEdge(jj,tr->nodes[idepth-1].iAtom,edgeEle,nEdges,edgeLabel);

	  ii++;
	  idepth=ii;
	  bEdge=TRUE;
	  break;
        }
      }
      if(!bEdge){ /*if not find any edge, track back until find another edge*/
        idepth--;
      }
    }
  }

  /*calculate the number of egdes and chain*/
  get_nEdges_tr(tr,BKSedge);

  tr->nChain=0;
  for (ii=0;ii<tr->N;ii++){
    //snew(tr->nodes[ii].edgeEle,tr->nodes[ii].nEdges);
    tr->nodes[ii].iChain=tr->nChain;
    tr->nodes[ii].edgesLeft=tr->nodes[ii].nEdges;
    if(tr->nodes[ii].nEdges==1) tr->nChain++;
  }
  fprintf(stderr,"nChain is %d\n",tr->nChain);
  snew(tr->chains,tr->nChain);
  tr->nodes[0].iChain=1;

  /*Search the tree again to sign the elements of nodes from the node to the base*/
  for (ii=0;ii<tr->N;ii++){
    for (jj=0;jj<nEdges[ii];jj++){ edgeLabel[ii][jj]=1; }
  }

  tr->nodes[0].nEle=1;
  snew(tr->nodes[0].ele,tr->nodes[0].nEle);
  tr->nodes[0].ele[0]=0;
  bEdges[base]=2;
  for (idepth=ii=1;ii<tr->N;){
    bEdge=FALSE;
    /*for atoms have more than 2 edges, choose the heavy atom as the first leaf if it has one*/
    if((bHeavy || bBackbone) && (nEdges[tr->nodes[idepth-1].iAtom]>2) && (bEdges[tr->nodes[idepth-1].iAtom]==1)){
      bEdges[tr->nodes[idepth-1].iAtom]=2;
      if(bBackbone){
        for(jj=0;jj<tr->N;jj++){
          //if((cmat[tr->nodes[idepth-1].iAtom][jj]==1) && bInBackbone(backbone,T2S(jj,bST,sT),N)){
          if(isEdge(tr->nodes[idepth-1].iAtom,jj,edgeEle,nEdges,edgeLabel) && backbone[T2S(jj,bST,sT)]){
            tr->edges[ii-1].iNode1=idepth-1;
            tr->edges[ii-1].iNode2=ii;
            delEdge(tr->nodes[idepth-1].iAtom,jj,edgeEle,nEdges,edgeLabel);
            delEdge(jj,tr->nodes[idepth-1].iAtom,edgeEle,nEdges,edgeLabel);

            tr->nodes[ii].nEle=tr->nodes[idepth-1].nEle+1;
            snew(tr->nodes[ii].ele,tr->nodes[ii].nEle);
            tr->nodes[ii].ele[tr->nodes[idepth-1].nEle]=ii;
            for(kk=0;kk<tr->nodes[idepth-1].nEle;kk++)
              tr->nodes[ii].ele[kk]=tr->nodes[idepth-1].ele[kk];

            ii++;
            idepth=ii;
            bEdge=TRUE;
            break;
          }
        }
      }
      if(!bEdge && bHeavy){
        for(jj=0;jj<tr->N;jj++){
          if(isEdge(tr->nodes[idepth-1].iAtom,jj,edgeEle,nEdges,edgeLabel) && (top->atoms.atom[T2S(jj,bST,sT)].m>HMASS)){
            tr->edges[ii-1].iNode1=idepth-1;
            tr->edges[ii-1].iNode2=ii;
            delEdge(tr->nodes[idepth-1].iAtom,jj,edgeEle,nEdges,edgeLabel);
            delEdge(jj,tr->nodes[idepth-1].iAtom,edgeEle,nEdges,edgeLabel);

            tr->nodes[ii].nEle=tr->nodes[idepth-1].nEle+1;
            snew(tr->nodes[ii].ele,tr->nodes[ii].nEle);
            tr->nodes[ii].ele[tr->nodes[idepth-1].nEle]=ii;
            for(kk=0;kk<tr->nodes[idepth-1].nEle;kk++)
              tr->nodes[ii].ele[kk]=tr->nodes[idepth-1].ele[kk];

            ii++;
            idepth=ii;
            bEdge=TRUE;
            break;
          }
        }
      }
    }
    if(!bEdge){
      for(jj=0;jj<tr->N;jj++){
        if(isEdge(tr->nodes[idepth-1].iAtom,jj,edgeEle,nEdges,edgeLabel)){
          tr->edges[ii-1].iNode1=idepth-1;
          tr->edges[ii-1].iNode2=ii;
          delEdge(tr->nodes[idepth-1].iAtom,jj,edgeEle,nEdges,edgeLabel);
          delEdge(jj,tr->nodes[idepth-1].iAtom,edgeEle,nEdges,edgeLabel);

          tr->nodes[ii].nEle=tr->nodes[idepth-1].nEle+1;
          snew(tr->nodes[ii].ele,tr->nodes[ii].nEle);
          tr->nodes[ii].ele[tr->nodes[idepth-1].nEle]=ii;
          for(kk=0;kk<tr->nodes[idepth-1].nEle;kk++)
            tr->nodes[ii].ele[kk]=tr->nodes[idepth-1].ele[kk];

          ii++;
          idepth=ii;
          bEdge=TRUE;
          break;
        }
      }
      if(!bEdge){ /*if not find any edge, track back until find another edge*/
        idepth--;
      }
    }
  } 

  /*Search for nodes drdq is not zero*/
  for(ii=0;ii<tr->N;ii++){
    tr->nodes[ii].ndrdq=0;
    for(jj=ii;jj<tr->N;jj++){
      for(kk=0;kk<tr->nodes[jj].nEle;kk++){
        if(tr->nodes[jj].ele[kk]==ii){
	  buf[tr->nodes[ii].ndrdq]=jj;
	  tr->nodes[ii].ndrdq++;
	  break;
	}
      }
    }
    snew(tr->nodes[ii].drdqEle,tr->nodes[ii].ndrdq);
    for(jj=0;jj<tr->nodes[ii].ndrdq;jj++)
      tr->nodes[ii].drdqEle[jj]=T2S(buf[jj],bST,sT);
  }

  /*resign the iAtom with index from system */
  if(bST){
    for(ii=0;ii<tr->N;ii++) 
      tr->nodes[ii].iAtom=sT[tr->nodes[ii].iAtom];
  }

  for(ii=0;ii<tr->N;ii++){
    get_index_ijkl(tr, ii, tr->nodes[ii].ijkl);
    get_name(tr,ii);
  }
  update_names(tr); /* update the name of external six coordinates */

  pr_tree(tr);

  sfree(buf);
  for(ii=0;ii<2;ii++){
    sfree(BKSedge[ii]);
    sfree(bondList[ii]);
  }
  sfree(BKSedge);
  sfree(bondList);
  for (ii=0;ii<tr->N;ii++){
    sfree(edgeEle[ii]);
    sfree(edgeLabel[ii]);
  }
  sfree(edgeEle);
  sfree(edgeLabel);
  sfree(nEdges);
  sfree(bEdges);

  /* add the free of matrices (not done April 29th 2016) */
  if(bST) sfree(sT2);
  
}

static real cal_dih_angle(const rvec xi,const rvec xj,const rvec xk,const rvec xl,
               const t_pbc *pbc)
{
  int  t1,t2,t3;
  rvec r_ij,r_kj,r_kl,m,n;
  real sign,aaa;

  aaa=dih_angle(xi,xj,xk,xl,pbc,r_ij,r_kj,r_kl,m,n,
                  &sign,&t1,&t2,&t3);

  return aaa;
}

static real cal_bond_angle(const rvec xi,const rvec xj,const rvec xk,const t_pbc *pbc)
{
  int t1,t2;
  rvec r_ij,r_kj;
  real cos_theta, theta;

  theta=bond_angle(xi,xj,xk,pbc,r_ij,r_kj,&cos_theta,&t1,&t2);
 
  return theta;
}

static real cal_bond_length(const t_pbc *pbc, const rvec xi,const rvec xj)
{
  int t1;
  rvec tmp;
  real vnorm;

  t1=pbc_rvec_sub(pbc,xi,xj,tmp);
  vnorm=norm(tmp);

  return vnorm;
}

static void get_rvec_xj(rvec *xx, BKStree_t *tr, int i, rvec xj)
{
  int n,nj;
  BKSnode_t *node;

  node=&tr->nodes[i];
  n=node->nEle;
  if(n==1){
    copy_rvec(tr->O0,xj);
  }
  if(n>=2){
    nj=node->ijkl[1];
    copy_rvec(xx[nj],xj);
  }
}

static void get_rvec_ijkl(rvec *xx, BKStree_t *tr, int i, rvec xi, rvec xj, rvec xk,rvec xl)
{
  real O0[]={0.0,0.0,0.0}, x0[]={0.1,0.0,0.0}, y0[]={0.0,0.1,0.0}, z0[]={0.0,0.0,0.1};
  int n,ni,nj,nk,nl;
  BKSnode_t *node;
 
  rvec_inc(O0,tr->O0);  /* set the origin of the lab-fixed frame */
  rvec_inc(x0,tr->O0);  
  rvec_inc(y0,tr->O0);
  rvec_inc(z0,tr->O0);
  
  node=&tr->nodes[i];
  n=node->nEle;  
  if(n==1){
    ni=node->ijkl[0];    
    copy_rvec(xx[ni],xi);
    copy_rvec(O0,xj);
    copy_rvec(z0,xk);
    copy_rvec(x0,xl);
  }
  if(n==2){
    ni=node->ijkl[0];
    nj=node->ijkl[1];
    copy_rvec(xx[ni],xi);
    copy_rvec(xx[nj],xj);
    copy_rvec(O0,xk);
    copy_rvec(z0,xl);
  }
  if(n==3){
    ni=node->ijkl[0];
    nj=node->ijkl[1];
    nk=node->ijkl[2];
    copy_rvec(xx[ni],xi);
    copy_rvec(xx[nj],xj);
    copy_rvec(xx[nk],xk);
    nj=node->ele[n-2];
    if((tr->nodes[nj].nEdges>2) && (node->ele[n-1] != (nj+1))){
      nl=node->ijkl[3];
      copy_rvec(xx[nl],xl);
    }else{
      copy_rvec(O0,xl);
    }
  }
  if(n>=4){
    ni=node->ijkl[0];
    nj=node->ijkl[1];
    nk=node->ijkl[2];
    nl=node->ijkl[3];
    copy_rvec(xx[ni],xi);
    copy_rvec(xx[nj],xj);
    copy_rvec(xx[nk],xk);
    copy_rvec(xx[nl],xl);
  }
}

static void pr_nrvec(rvec *xx, int N)
{
  int i;
  for(i=0;i<N;i++)
    fprintf(stderr,"%d: %g %g %g\n",i,xx[i][XX],xx[i][YY],xx[i][ZZ]);
}

static void set_frame_origin(rvec *xx, BKStree_t *tr, t_pbc *pbc)
{
  int a1,a2,a3,t1,d;
  rvec dx;
  real tol=1.0e-6;

  a1=tr->nodes[0].iAtom;
  a2=tr->nodes[1].iAtom;
  a3=tr->nodes[2].iAtom;
  t1=pbc_rvec_sub(pbc,xx[a3],xx[a2],dx);
  
  for(d=0;d<DIM;d++){
    tr->O0[d]=xx[a1][d]-5*dx[d];
  }
}

static void check_frame_origin(rvec *xx, BKStree_t *tr, t_pbc *pbc)
{
  int i,a1,t1;
  rvec dx1,dx2;
  real dr,tol=1.0e-6;
  //real dr1,dr2;

  for(i=0;i<tr->N;i++){
    a1=tr->nodes[i].iAtom;
    t1=pbc_rvec_sub(pbc,tr->O0,xx[a1],dx1);
    rvec_sub(tr->O0,xx[a1],dx2);
    //dr1=norm(dx1); dr2=norm(dx2);
    rvec_sub(dx1,dx2,dx1);
    dr=norm(dx1);
    //fprintf(stderr,"For node %d: With pbc dr= %f , Without pbc dr= %f\n",i,dr1,dr2);
    //if(dr>tol) fprintf(stderr,"WARNING: The origin is set too far away and the pbc condition makes a difference of %f\n",dr);
    if(dr>tol) gmx_fatal(FARGS,"The origin is set too far away and the pbc condition makes a difference of %f\n",dr);
  }
}

static void calUnitVecE(rvec *xx, rvec *ex, rvec *ey, rvec *ez, BKStree_t *tr, t_pbc *pbc)
{
  int i,j,M,t1;
  rvec xi,xj,xk,xl;
  real vnorm;
  rvec tmp;

  M=tr->N;

  i=0;
  get_rvec_ijkl(xx,tr,i,xi,xj,xk,xl);
  rvec_sub(xi,xj,ex[i]);
  rvec_sub(xj,xk,ey[i]);
  cprod(ex[i],ey[i],ez[i]);

  for(i=1;i<M;i++){
    get_rvec_ijkl(xx,tr,i,xi,xj,xk,xl);
    t1=pbc_rvec_sub(pbc,xi,xj,ex[i]);
    t1=pbc_rvec_sub(pbc,xj,xk,ey[i]);
    cprod(ex[i],ey[i],ez[i]);
  }

  /*normalize to unit vector*/
  for(i=0;i<M;i++){
    unitv(ex[i],ex[i]);
    unitv(ey[i],ey[i]);
    unitv(ez[i],ez[i]);
  }
}

static void calInternalQ(rvec *xx, rvec *q, BKStree_t *tr, t_pbc *pbc)
{
  int i,j,M,t1;
  rvec xi,xj,xk,xl;
  rvec tmp;

  M=tr->N;
 
  for(i=0;i<M;i++){
    get_rvec_ijkl(xx,tr,i,xi,xj,xk,xl);
    q[i][XX]=cal_bond_length(pbc,xi,xj);
    q[i][YY]=cal_bond_angle(xi,xj,xk,pbc);
    q[i][ZZ]=cal_dih_angle(xi,xj,xk,xl,pbc);
  }
} 

static void unit_e(t_pbc *pbc, rvec xi, rvec xj, rvec ex)
{
  int t1;
  t1=pbc_rvec_sub(pbc,xi,xj,ex);
  unitv(ex,ex);
}

static real unit_cprod(rvec eji, rvec ekj)
{
  rvec v1;
  cprod(eji,ekj,v1);
  return norm(v1);
}

/* Note that A1_eta is transposed */
static void calA1_eta(rvec *xx, BKStree_t *tr, rvec *ex, rvec *ey, rvec *ez, real **A_mat, t_pbc *pbc, gmx_bool bST, int *sT2)
{
  int i,j,M,t1,d,k;
  int alpha;
  rvec xi,xj,xk,xl;
  rvec buf,buf2,buf3;
  int n,nj;

  M=tr->N;
  /* clean A_mat */
  clear_matrix(A_mat,9,3*M);

  j=0;
  get_rvec_xj(xx,tr,j,xj);
  for(i=0;i<tr->nodes[j].ndrdq;i++){
    alpha=tr->nodes[j].drdqEle[i];
    k=S2T(alpha,bST,sT2);
    alpha=tr->nodes[k].iAtom;
    for(d=0;d<DIM;d++) A_mat[j*3][k*3+d]=ex[j][d];
    rvec_sub(xx[alpha],xj,buf2);
    cprod(ez[j],buf2,buf3);
    for(d=0;d<DIM;d++) A_mat[j*3+1][k*3+d]=buf3[d];
    cprod(ey[j],buf2,buf3);
    for(d=0;d<DIM;d++) A_mat[j*3+2][k*3+d]=buf3[d];
  }

  for(j=1;j<3;j++){
    get_rvec_xj(xx,tr,j,xj);
    for(i=0;i<tr->nodes[j].ndrdq;i++){
      alpha=tr->nodes[j].drdqEle[i];
      k=S2T(alpha,bST,sT2);
      alpha=tr->nodes[k].iAtom;
      for(d=0;d<DIM;d++) A_mat[j*3][k*3+d]=ex[j][d];
      t1=pbc_rvec_sub(pbc,xx[alpha],xj,buf2);
      cprod(ez[j],buf2,buf3);
      for(d=0;d<DIM;d++) A_mat[j*3+1][k*3+d]=buf3[d];
      cprod(ey[j],buf2,buf3);
      for(d=0;d<DIM;d++) A_mat[j*3+2][k*3+d]=buf3[d];
    }
    n=tr->nodes[j].nEle;
    if(n>=3){  /*for proper dihedral, we count all the nodes after it, that is take the ndrdqEle of the previous node */
      nj=tr->nodes[j].ele[n-2];
      if((tr->nodes[nj].nEdges>2) && (tr->nodes[j].ele[n-1] == (nj+1))){
        for(i=1;i<tr->nodes[nj].ndrdq;i++){
          alpha=tr->nodes[nj].drdqEle[i];
          k=S2T(alpha,bST,sT2);
          alpha=tr->nodes[k].iAtom;
          t1=pbc_rvec_sub(pbc,xx[alpha],xj,buf2);
          cprod(ey[j],buf2,buf3);
          for(d=0;d<DIM;d++) A_mat[j*3+2][k*3+d]=buf3[d];
        }
      }
    }
  }

  /* swith column 4 and 9 of A1_eta */
  for(k=0;k<3*M;k++){
    buf2[0]=A_mat[3][k];
    A_mat[3][k]=A_mat[8][k];
    A_mat[8][k]=buf2[0];
  }
}

/* Note that Ai_eta is transposed */
static void calAi_eta(int j,rvec *xx, BKStree_t *tr, rvec *ex, rvec *ey, rvec *ez, real **A_mat, t_pbc *pbc, gmx_bool bST, int *sT2)
{
  int i,M,t1,d,k;
  int alpha;
  rvec xi,xj,xk,xl;
  rvec buf,buf2,buf3;
  int n,nj;

  M=tr->N;
  /* clean A_mat */
  clear_matrix(A_mat,3,3*M);

  get_rvec_xj(xx,tr,j,xj); /* j>3 */
  for(i=0;i<tr->nodes[j].ndrdq;i++){
    alpha=tr->nodes[j].drdqEle[i];
    k=S2T(alpha,bST,sT2);
    alpha=tr->nodes[k].iAtom;
    for(d=0;d<DIM;d++) A_mat[0][k*3+d]=ex[j][d];
    t1=pbc_rvec_sub(pbc,xx[alpha],xj,buf2);
    cprod(ez[j],buf2,buf3);
    for(d=0;d<DIM;d++) A_mat[1][k*3+d]=buf3[d];
    cprod(ey[j],buf2,buf3);
    for(d=0;d<DIM;d++) A_mat[2][k*3+d]=buf3[d];
  }
  n=tr->nodes[j].nEle;
  if(n>=3){  /*for proper dihedral, we count all the nodes after it, that is take the ndrdqEle of the previous node */
    nj=tr->nodes[j].ele[n-2];
    if((tr->nodes[nj].nEdges>2) && (tr->nodes[j].ele[n-1] == (nj+1))){
      for(i=1;i<tr->nodes[nj].ndrdq;i++){
        alpha=tr->nodes[nj].drdqEle[i];
        k=S2T(alpha,bST,sT2);
        alpha=tr->nodes[k].iAtom;
        t1=pbc_rvec_sub(pbc,xx[alpha],xj,buf2);
        cprod(ey[j],buf2,buf3);
        for(d=0;d<DIM;d++) A_mat[2][k*3+d]=buf3[d];
      }
    }
  }
}

/* not used */
static void calA_matrix(rvec *xx, BKStree_t *tr, rvec *ex, rvec *ey, rvec *ez, real **A_mat, t_pbc *pbc, gmx_bool bST, int *sT2)
{
  int i,j,M,t1,d,k;
  int alpha;
  rvec xi,xj,xk,xl;
  rvec buf,buf2,buf3;
  int n,nj;

  M=tr->N;
  /* clean A_mat */
  clear_matrix(A_mat,3*M,3*M);

  for(j=0;j<M;j++){
    get_rvec_xj(xx,tr,j,xj);
    for(i=0;i<tr->nodes[j].ndrdq;i++){ 
      alpha=tr->nodes[j].drdqEle[i];
      k=S2T(alpha,bST,sT2);
      alpha=tr->nodes[k].iAtom;
      for(d=0;d<DIM;d++) A_mat[k*3+d][j*3]=ex[j][d];
      t1=pbc_rvec_sub(pbc,xx[alpha],xj,buf2);
      cprod(ez[j],buf2,buf3);
      for(d=0;d<DIM;d++) A_mat[k*3+d][j*3+1]=buf3[d];
      cprod(ey[j],buf2,buf3);
      for(d=0;d<DIM;d++) A_mat[k*3+d][j*3+2]=buf3[d];
    }
    n=tr->nodes[j].nEle;
    if(n>=3){  /*for proper dihedral, we count all the nodes after it, that is take the ndrdqEle of the previous node */
      nj=tr->nodes[j].ele[n-2];
      if((tr->nodes[nj].nEdges>2) && (tr->nodes[j].ele[n-1] == (nj+1))){
        for(i=1;i<tr->nodes[nj].ndrdq;i++){
          alpha=tr->nodes[nj].drdqEle[i];
	  k=S2T(alpha,bST,sT2);
          alpha=tr->nodes[k].iAtom;
          t1=pbc_rvec_sub(pbc,xx[alpha],xj,buf2);
          cprod(ey[j],buf2,buf3);
	  for(d=0;d<DIM;d++) A_mat[k*3+d][j*3+2]=buf3[d];
        }
      }
    }
  }
}

static void calGeneF_BKS(rvec *xx, rvec *ff, rvec *extf, BKStree_t *tr, rvec *ex, rvec *ey, rvec *ez, rvec *intQ, t_pbc *pbc, gmx_bool bST, int *sT2)
{
  int i,j,M,t1;
  int alpha;
  rvec xi,xj,xk,xl;
  rvec buf,buf2,buf3,tbuf;
  int n,nj,nk;

  M=tr->N;

  real zero[]={0.0,0.0,0.0};

  for(j=0;j<M;j++){
    copy_rvec(zero,intQ[j]);
    get_rvec_xj(xx,tr,j,xj);
    for(i=0;i<tr->nodes[j].ndrdq;i++){ /*How about try to sum the force or torque first and then do the dotproduct with the unit vector? */
      alpha=tr->nodes[j].drdqEle[i];
      nk=S2T(alpha,bST,sT2);
      alpha=tr->nodes[nk].iAtom;
      rvec_sub(ff[alpha],extf[nk],tbuf);
      intQ[j][XX]+=iprod(tbuf,ex[j]);
      t1=pbc_rvec_sub(pbc,xx[alpha],xj,buf2);
      cprod(ez[j],buf2,buf3);
      intQ[j][YY]+=iprod(tbuf,buf3);
      cprod(ey[j],buf2,buf3);
      intQ[j][ZZ]+=iprod(tbuf,buf3);
    }
    n=tr->nodes[j].nEle;
    if(n>=3){  /*for proper dihedral, we count all the nodes after it, that is take the ndrdqEle of the previous node */
      nj=tr->nodes[j].ele[n-2];
      if((tr->nodes[nj].nEdges>2) && (tr->nodes[j].ele[n-1] == (nj+1))){
//        fprintf(stderr,"Node %d, from node %d, drdqEle: ",j,nj);
	intQ[j][ZZ]=0.0;
        for(i=1;i<tr->nodes[nj].ndrdq;i++){
          alpha=tr->nodes[nj].drdqEle[i];
//          fprintf(stderr," %d",alpha);
          nk=S2T(alpha,bST,sT2);
          alpha=tr->nodes[nk].iAtom;
          rvec_sub(ff[alpha],extf[nk],tbuf);
          t1=pbc_rvec_sub(pbc,xx[alpha],xj,buf2);
          cprod(ey[j],buf2,buf3);
          intQ[j][ZZ]+=iprod(tbuf,buf3);
        }
//	fprintf(stderr,"\n");
      }
    }  
  }
}

/* Here the column order in D1_mat is the defaut one from the BKStree */
static void calD1_mat(rvec *xx, real **B1_mat, real **D1_mat, BKStree_t *tr, rvec *ex, rvec *ey, rvec *ez, t_pbc *pbc, gmx_bool bST, int *sT2)
{
  int i,j,M,t1,next;
  int alpha;
  rvec xi,xj,xk,xl;
  rvec buf,buf2,buf3;
  int n,nj,nk;
  rvec buf_B1;

  M=tr->N;
  clear_matrix(D1_mat,6,3*M);

for(next=0;next<6;next++){
  j=0;
  get_rvec_xj(xx,tr,j,xj);
  for(i=0;i<tr->nodes[j].ndrdq;i++){ /*How about try to sum the force or torque first and then do the dotproduct with the unit vector? */
    alpha=tr->nodes[j].drdqEle[i];
    nk=S2T(alpha,bST,sT2);
    buf_B1[XX]=B1_mat[next][3*nk+XX]; buf_B1[YY]=B1_mat[next][3*nk+YY]; buf_B1[ZZ]=B1_mat[next][3*nk+ZZ];
    alpha=tr->nodes[nk].iAtom;
    D1_mat[next][3*j+XX]+=iprod(buf_B1,ex[j]);
    rvec_sub(xx[alpha],xj,buf2);
    cprod(ez[j],buf2,buf3);
    D1_mat[next][3*j+YY]+=iprod(buf_B1,buf3);
    cprod(ey[j],buf2,buf3);
    D1_mat[next][3*j+ZZ]+=iprod(buf_B1,buf3);
  }

  for(j=1;j<M;j++){
    get_rvec_xj(xx,tr,j,xj);
    for(i=0;i<tr->nodes[j].ndrdq;i++){ /*How about try to sum the force or torque first and then do the dotproduct with the unit vector? */
      alpha=tr->nodes[j].drdqEle[i];
      nk=S2T(alpha,bST,sT2);
      buf_B1[XX]=B1_mat[next][3*nk+XX]; buf_B1[YY]=B1_mat[next][3*nk+YY]; buf_B1[ZZ]=B1_mat[next][3*nk+ZZ];
      alpha=tr->nodes[nk].iAtom;
      D1_mat[next][3*j+XX]+=iprod(buf_B1,ex[j]);
      t1=pbc_rvec_sub(pbc,xx[alpha],xj,buf2);
      cprod(ez[j],buf2,buf3);
      D1_mat[next][3*j+YY]+=iprod(buf_B1,buf3);
      cprod(ey[j],buf2,buf3);
      D1_mat[next][3*j+ZZ]+=iprod(buf_B1,buf3);
    }
    n=tr->nodes[j].nEle;
    if(n>=3){  /*for proper dihedral, we count all the nodes after it, that is take the ndrdqEle of the previous node */
      nj=tr->nodes[j].ele[n-2];
      if((tr->nodes[nj].nEdges>2) && (tr->nodes[j].ele[n-1] == (nj+1))){
//        fprintf(stderr,"Node %d, from node %d, drdqEle: ",j,nj);
        D1_mat[next][3*j+ZZ]=0.0;
        for(i=1;i<tr->nodes[nj].ndrdq;i++){
          alpha=tr->nodes[nj].drdqEle[i];
//          fprintf(stderr," %d",alpha);
          nk=S2T(alpha,bST,sT2);
          buf_B1[XX]=B1_mat[next][3*nk+XX]; buf_B1[YY]=B1_mat[next][3*nk+YY]; buf_B1[ZZ]=B1_mat[next][3*nk+ZZ];
          alpha=tr->nodes[nk].iAtom;
          t1=pbc_rvec_sub(pbc,xx[alpha],xj,buf2);
          cprod(ey[j],buf2,buf3);
          D1_mat[next][3*j+ZZ]+=iprod(buf_B1,buf3);
        }
//      fprintf(stderr,"\n");
      }
    }
  }
}

}

static void B_dot_A_mat_new(real **B_mat,real **A_mat, int M, int N)
{
  int i,j,k;
  real tol=1.0e-6;
  real tmp;
  for(i=0;i<M;i++){
    for(j=0;j<M;j++){
      tmp=0;
      for(k=0;k<N;k++){
        tmp+=B_mat[i][k]*A_mat[k][j];
      }
      if(i==j) tmp-=1.0;
      if(fabs(tmp)>tol)
        fprintf(stderr,"ERROR: %.10lf \n",tmp);
    }
  }
}

static int get_sign(int n){
  if(n%2 == 0) return 1;
  else return -1;
}

/*
   Recursive definition of determinate using expansion by minors.
*/
static real Determinant(real **a,int n)
{
   int i,j,j1,j2;
   real det = 0;
   real **m = NULL;

   if (n < 1) { /* Error */
     gmx_fatal(FARGS,"Not a matrix\n");
   } else if (n == 1) { /* Shouldn't get used */
      det = a[0][0];
   } else if (n == 2) {
      det = a[0][0] * a[1][1] - a[1][0] * a[0][1];
   } else {
      det = 0;
      for (j1=0;j1<n;j1++) {
         snew_mat(m,n-1,n-1,i);
         for (i=1;i<n;i++) {
            j2 = 0;
            for (j=0;j<n;j++) {
               if (j == j1)
                  continue;
               m[i-1][j2] = a[i][j];
               j2++;
            }
         }
         det += get_sign(j1) * a[0][j1] * Determinant(m,n-1);
         sfree_mat(m,n-1,i);
      }
   }
   return(det);
}

/*
   Find the cofactor matrix of a square matrix
*/
static void Adjoint(real **a,int n,real **b)
{
   int i,j,ii,jj,i1,j1;
   real det;
   real **c;

   snew_mat(c,n-1,n-1,i);

   for (j=0;j<n;j++) {
      for (i=0;i<n;i++) {

         /* Form the adjoint a_ij */
         i1 = 0;
         for (ii=0;ii<n;ii++) {
            if (ii == i)
               continue;
            j1 = 0;
            for (jj=0;jj<n;jj++) {
               if (jj == j)
                  continue;
               c[i1][j1] = a[ii][jj];
               j1++;
            }
            i1++;
         }

         /* Calculate the determinate */
         det = Determinant(c,n-1);

         /* Fill in the elements of the cofactor */
         b[j][i] = get_sign(i+j) * det;
      }
   }
   sfree_mat(c,n-1,i);
}

static void do_inverse_mat(real **m, int N, real **minv)
{
  const real smallreal = (real)1.0e-24;
  const real largereal = (real)1.0e24;
  int i,j;
  real det,tmp;

  det=Determinant(m,N);
  fprintf(stderr,"Determinant = %e\n",det);

  tmp = (real)fabs(det);
  if((tmp <= smallreal) || (tmp >= largereal))
  {
    fprintf(stderr,"WARNING: Determinant = %e\n",det); 
    //gmx_fatal(FARGS, "Can not invert matrix, determinant = %e", det);
  }
 
  Adjoint(m,N,minv); 

  /* m^-1 = adj(m)/det(m) */ 
  tmp=(real)1.0/det; 
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      minv[i][j]*=tmp;

}

/* input a rectangular matrix B (N*M), return B+ with BB+=I if N<M B+=BT(BBT)^-1 
   When N=M, moore-penrose inverse is the inverse
*/
static void moore_penrose_inverse(real **m, int N, int M, real **minv)
{
  int i,j,t1;
  real **mat, **matInv;
  real **NM;

  if(N>M)
    gmx_fatal(FARGS,"Can not do pseudo inverse for a matrix with more rows than columns\n");

  snew_mat(mat,M,M,i);
  snew_mat(matInv,M,M,i);
  snew_mat(NM,M,M,i);

  matrix_transpose(m,N,M,NM);
  matrix_multi(m,NM,N,M,N,mat);
  t1=m_inv_gen(mat,N,matInv);
  if(t1>0)
    fprintf(stderr,"WARNING: Invert a matrix which is singular with %d zero eigenvalues\n",t1);
    //gmx_fatal(FARGS,"Invert a matrix which is singular with %d zero eigenvalues\n",t1);
  matrix_multi(NM,matInv,M,N,N,minv);
//  fprintf(stderr,"check pseudo inverse B2*B2+\n");
//  B_dot_A_mat_new(m,minv,N,M);

  sfree_mat(mat,M,i);
  sfree_mat(matInv,M,i);
  sfree_mat(NM,M,i);
}

/* here E12 is actually the minus of the true E12 */
static void calE12_mat(real **D1,real **E12,int M)
{
  real **tmp,**tmp2;
  int i,j,t1;

  snew_mat(tmp,6,6,i);
  snew_mat(tmp2,6,6,i);

  /* shuffle the column order in D1 matrix as ext and internal by switch the postion of rot-ZZ ([2][ZZ]) and the first bond coordinate ([1][XX]) */
  for(i=0;i<6;i++){
    tmp[i][0]=D1[i][2*3+ZZ];
    D1[i][2*3+ZZ]=D1[i][1*3+XX];
    D1[i][1*3+XX]=tmp[i][0];
  }

  /* copy D11 (6 \times 6) matrix into tmp2 */
  for(i=0;i<6;i++){
    for(j=0;j<6;j++) tmp2[i][j]=D1[i][j];
  } 

  /* inverse D11 (6 \times 6) matrix and saved in tmp */
  //moore_penrose_inverse(tmp2,6,6,tmp);
  do_inverse_mat(tmp2,6,tmp);

  /* move D12 into the front of D1 matrix */
  for(i=0;i<6;i++){
    for(j=6;j<M;j++) D1[i][j-6]=D1[i][j];
  }

  /*  -E12 = D11^-1 * D12 */
  matrix_multi(tmp,D1,6,6,M-6,E12); 

  sfree_mat(tmp,6,i);
  sfree_mat(tmp2,6,i);
}

static void writeBin_data_prefix(rvec *dw, BKStree_t *tr, gmx_bool flag, char *prefix){
  FILE *out=NULL;
  int i;
  char fname[80];
  
  if(prefix==NULL)
      sprintf(fname,"%s","all_coordinate.xvg");
    else
      sprintf(fname,"%s_%s",prefix,"all_coordinate.xvg");
    if(flag)
      out=fopen(fname,"w");
    else
      out=fopen(fname,"a");
    if(out==NULL){
      gmx_fatal(FARGS,"Can't open file %s\n",fname);
    }else{
      writeBin_nvec(out,dw,tr->N,DIM);
    }
    fclose(out);
}

static void print_data_prefix(real time, rvec *dw, BKStree_t *tr, gmx_bool flag, char *prefix)
{
  FILE *out=NULL;
  int i;
  char fname[80];

  for(i=0;i<tr->N;i++){
    if(prefix==NULL)
      sprintf(fname,"%s",tr->nodes[i].nameXX);
    else
      sprintf(fname,"%s_%s",prefix,tr->nodes[i].nameXX);
    if(flag) 
      out=fopen(fname,"w");
    else
      out=fopen(fname,"a");
    if(out==NULL){
      gmx_fatal(FARGS,"Can't open file %s\n",fname);
    }else{
      fprintf(out,"%g\t%g\n",time,dw[i][XX]);
    }
    fclose(out);

    if(prefix==NULL)
      sprintf(fname,"%s",tr->nodes[i].nameYY);
    else
      sprintf(fname,"%s_%s",prefix,tr->nodes[i].nameYY);
    if(flag)
      out=fopen(fname,"w");
    else
      out=fopen(fname,"a");
    if(out==NULL){
      gmx_fatal(FARGS,"Can't open file %s\n",fname);
    }else{
      fprintf(out,"%g\t%g\n",time,dw[i][YY]);
    }
    fclose(out);

    if(prefix==NULL)
      sprintf(fname,"%s",tr->nodes[i].nameZZ);
    else
      sprintf(fname,"%s_%s",prefix,tr->nodes[i].nameZZ);
    if(flag)
      out=fopen(fname,"w");
    else
      out=fopen(fname,"a");
    if(out==NULL){
      gmx_fatal(FARGS,"Can't open file %s\n",fname);
    }else{
      fprintf(out,"%g\t%g\n",time,dw[i][ZZ]);
    }
    fclose(out);
  }
}

static void print_nlines(FILE *out,real time,real **dw, int i,int nset, int m)
{
  int j;

  fprintf(out,"%g",time);
  for(j=0;j<nset;j++)
    fprintf(out,"\t%g",dw[j][i*DIM+m]);
  fprintf(out,"\n");
}

static void print_data_intQ(rvec *Q, BKStree_t *tr, gmx_bool flag)
{
  writeBin_data_prefix(Q,tr,flag,"intQ");
}

static void print_data_geneF(rvec *F, BKStree_t *tr, gmx_bool flag)
{
  writeBin_data_prefix(F,tr,flag,"geneF");
}

static void print_data_geneF_FM(rvec *dw, BKStree_t *tr, gmx_bool flag, int m)
{
  FILE *out=NULL;
  int i;
  char fname[80];

  sprintf(fname,"%s_%d.%s","geneF_all_coordinate",m,"xvg");
  if(flag)
    out=fopen(fname,"w");
  else
    out=fopen(fname,"a");
  if(out==NULL){
    gmx_fatal(FARGS,"Can't open file %s\n",fname);
  }else{
    writeBin_nvec(out,dw,tr->N,DIM);
  }
  fclose(out);
}

static void print_data_dotQ(rvec *F, BKStree_t *tr, gmx_bool flag)
{
  writeBin_data_prefix(F,tr,flag,"dotQ");
}

static void print_data_dlnBdqi(rvec *F, BKStree_t *tr, gmx_bool flag)
{
  writeBin_data_prefix(F,tr,flag,"dlnBdqi");
}

/* Here, xp is the coordinate relative to COM */
static void cal_extF(t_topology *top,BKStree_t *tr,rvec *xp,real **inertia,real **inv_inert,rvec *extF, rvec totF, rvec tau)
{
  int i,j,d,alpha,t1;
  real m0,tm=0.0;
  int  M=tr->N;
  rvec tvec;

  /* get the total mass */
  for(i=0;i<M;i++){
    alpha=tr->nodes[i].iAtom;
    tm+=top->atoms.atom[alpha].m;
  }

  /* inertia */
  cal_tensor_mat(inertia,xp,top,tr);
  m3_inv(inertia,inv_inert);
  //t1=m_inv_gen(inertia,3,inv_inert); /* m_inv_gen can only get the inverse of symmetric matrix */
  //if(t1>0)
  //  gmx_fatal(FARGS,"Invert a matrix which is singular with %d zero eigenvalues\n",t1);

  /* tau*(I^-1) */
  clear_rvec(tvec);
  for(i=0;i<DIM;i++){ 
    for(d=0;d<DIM;d++){
      tvec[i]+=tau[d]*inv_inert[d][i];
    }
  }

  for(i=0;i<M;i++){
    alpha=tr->nodes[i].iAtom;
    m0=top->atoms.atom[alpha].m;
    
    cprod(tvec,xp[i],extF[i]); /* rotational part */
    svmul(m0,extF[i],extF[i]);
    
    for(d=0;d<DIM;d++){
      extF[i][d]+=totF[d]*m0/tm; /* translational part */
    }
  }
}

static int delta_fun(int d,int m)
{
  if(m==d) return 1;
  return 0;
}

static int delta_fun2(int ki,int kj,int km)
{
  if(km==ki) return 1;
  if(km==kj) return -1;
  return 0;
}

static void ang_st1(real cos_a, rvec eji, rvec ekj, real rij, rvec st1)
{
  real a;
  svmul(cos_a,eji,st1);
  rvec_inc(st1,ekj);
  a=1.0/(rij*sqrt(1-cos_a*cos_a)); /* here, we did not consider the case cos_a=1 */
  svmul(a,st1,st1);
}

static void ang_st2(real cos_a, rvec eji, rvec ekj, real rij, real rjk, rvec st2)
{
  real a1,a2;
  rvec v1;
  a1=rij-rjk*cos_a;
  a2=rij*cos_a-rjk;
  svmul(a1,eji,v1);
  svmul(a2,ekj,st2);
  rvec_inc(st2,v1);
  a1=1.0/(rij*rjk*sqrt(1-cos_a*cos_a));
  svmul(a1,st2,st2);

/*  a1=sqrt(1-cos_a*cos_a);
  a2=unit_cprod(eji,ekj);
  fprintf(stderr,"sin_alpha is %g or %g\n",a1,a2);
*/
}

static void ang_st3(real cos_a, rvec eji, rvec ekj, real rjk, rvec st3)
{
  ang_st1(cos_a,ekj,eji,rjk,st3);
  svmul(-1.0,st3,st3);
}

static void dih_st1(real cos_a, rvec eji, rvec ekj, real rij, rvec st1)
{
  real a;
  a=-1.0/(rij*(1-cos_a*cos_a)); /* here, we did not consider the case cos_a=1 */
  cprod(eji,ekj,st1);
  svmul(a,st1,st1);
}

static void dih_st2(real cos_a2, real cos_a3, rvec eji, rvec ekj, rvec elk, real rij, real rjk, rvec st2)
{
 real a1,a2;
 rvec v1;
 a1=(rjk-rij*cos_a2)/(rjk*rij*(1-cos_a2*cos_a2));
 a2=cos_a3/(rjk*(1-cos_a3*cos_a3));
 cprod(eji,ekj,v1);
 svmul(a1,v1,v1);
 cprod(elk,ekj,st2);
 svmul(a2,st2,st2);
 rvec_inc(st2,v1);
}

static void ang_rirm(real cos_a,rvec eji,rvec ejk,real rij,real rjk,real *colB,int ki,int kj,int kk,real ***rirm)
{
   real sin_a, res, tmp;
   int ii,d,m;

   sin_a=sqrt(1-cos_a*cos_a);
   for(ii=0;ii<DIM;ii++){
     for(d=0;d<DIM;d++){
      for(m=0;m<DIM;m++){
       res=delta_fun2(ki,kj,ii)*(delta_fun(d,m)-eji[d]*eji[m])/rij; /* partial_eji/partial_rm */
       res=res*cos_a-sin_a*eji[d]*colB[ii*3+m];
       res=res-delta_fun2(kk,kj,ii)*(delta_fun(d,m)-ejk[d]*ejk[m])/rjk;
       tmp=sin_a*delta_fun2(ki,kj,ii)*eji[m]+rij*cos_a*colB[ii*3+m];
       rirm[ii][d][m]=(res-colB[ki*3+d]*tmp)/(rij*sin_a);
      }
     }
   }
}

static void ang_rjrm(real cos_a,rvec eji,rvec ejk,real rij,real rjk,real *colB,int ki,int kj,int kk,real ***rirm)
{
   real sin_a, res, tmp;
   int ii,d,m;

   sin_a=sqrt(1-cos_a*cos_a);
   for(ii=0;ii<DIM;ii++){
     for(d=0;d<DIM;d++){
      for(m=0;m<DIM;m++){
       res=delta_fun2(ki,kj,ii)*eji[m]-cos_a*delta_fun2(kk,kj,ii)*ejk[m]+rjk*sin_a*colB[ii*3+m];
       res=eji[d]*res+(rij-rjk*cos_a)*delta_fun2(ki,kj,ii)*(delta_fun(d,m)-eji[d]*eji[m])/rij;
       res=res+ejk[d]*(delta_fun2(kk,kj,ii)*ejk[m]-cos_a*delta_fun2(ki,kj,ii)*eji[m]+rij*sin_a*colB[ii*3+m]);
       res=res+(rjk-rij*cos_a)*delta_fun2(kk,kj,ii)*(delta_fun(d,m)-ejk[d]*ejk[m])/rjk;
       tmp=sin_a*(rjk*delta_fun2(ki,kj,ii)*eji[m]+rij*delta_fun2(kk,kj,ii)*ejk[m])+cos_a*rij*rjk*colB[ii*3+m];
       rirm[ii][d][m]=(res-colB[kj*3+d]*tmp)/(rij*rjk*sin_a);
      }
     }
   }
}

static void dih_rirm(real cos_a,rvec eji,rvec ekj,real rij,real rjk,t_Bmat *B_mat,int ki,int kj,int kk,int i,real ***rirm)
{
   real sin_a;
   int ii,d,m;
   rvec res, tmp, tmp2;

   sin_a=sqrt(1-cos_a*cos_a);
   for(ii=0;ii<3;ii++){
     for(m=0;m<DIM;m++){
      for(d=0;d<DIM;d++) tmp2[d]=delta_fun2(ki,kj,ii)*(delta_fun(d,m)-eji[d]*eji[m])/rij;
      cprod(tmp2,ekj,res);
      for(d=0;d<DIM;d++) tmp2[d]=delta_fun2(kj,kk,ii)*(delta_fun(d,m)-ekj[d]*ekj[m])/rjk;
      cprod(eji,tmp2,tmp);
      for(d=0;d<DIM;d++) res[d]=res[d]+tmp[d];
      tmp[m]=delta_fun2(ki,kj,ii)*eji[m]*sin_a*sin_a+B_mat[i*3+1].colB[ii*3+m]*rij*2*sin_a*cos_a;
      for(d=0;d<DIM;d++) rirm[ii][d][m]=-(res[d]+B_mat[i*3+2].colB[ki*3+d]*tmp[m])/(rij*sin_a*sin_a);
     }
   }
}

static void dih_rlrm(real cos_a,rvec elk,rvec ekj,real rkl,real rjk,t_Bmat *B_mat,real **mat,int kj,int kk,int kl,int i,real ***rirm)
{
   real sin_a;
   int ii,d,m;
   rvec res, tmp, tmp2;

   sin_a=sqrt(1-cos_a*cos_a);
   for(ii=1;ii<4;ii++){
     for(m=0;m<DIM;m++){
      for(d=0;d<DIM;d++) tmp2[d]=delta_fun2(kk,kl,ii)*(delta_fun(d,m)-elk[d]*elk[m])/rkl;
      cprod(tmp2,ekj,res);
      for(d=0;d<DIM;d++) tmp2[d]=delta_fun2(kj,kk,ii)*(delta_fun(d,m)-ekj[d]*ekj[m])/rjk;
      cprod(elk,tmp2,tmp);
      for(d=0;d<DIM;d++) res[d]=res[d]+tmp[d];
      tmp[m]=delta_fun2(kk,kl,ii)*elk[m]*sin_a*sin_a+mat[ii][m]*rkl*2*sin_a*cos_a;
      for(d=0;d<DIM;d++) rirm[ii][d][m]=-(res[d]+B_mat[i*3+2].colB[kl*3+d]*tmp[m])/(rkl*sin_a*sin_a);
     }
   }
}

static void dih_rjrm(real cos_a2,real cos_a3,rvec eji,rvec ekj,rvec elk,real rij,real rjk,real rkl,t_Bmat *B_mat,real **mat,int ki,int kj,int kk,int kl,int i,real ***rirm)
{
   real sin_a2, sin_a3, v1;
   int ii,d,m;
   rvec res, tmp, tmp2, tmp3;

   sin_a2=sqrt(1-cos_a2*cos_a2);
   sin_a3=sqrt(1-cos_a3*cos_a3);
   for(ii=0;ii<4;ii++){
     for(m=0;m<DIM;m++){
      cprod(eji,ekj,tmp2);
      v1=delta_fun2(kj,kk,ii)*ekj[m]-cos_a2*delta_fun2(ki,kj,ii)*eji[m]+rij*sin_a2*B_mat[i*3+1].colB[ii*3+m];
      for(d=0;d<DIM;d++) tmp3[d]=delta_fun2(ki,kj,ii)*(delta_fun(d,m)-eji[d]*eji[m])/rij;
      cprod(tmp3,ekj,res);
      for(d=0;d<DIM;d++) tmp3[d]=delta_fun2(kj,kk,ii)*(delta_fun(d,m)-ekj[d]*ekj[m])/rjk;
      cprod(eji,tmp3,tmp);
      for(d=0;d<DIM;d++) res[d]=(tmp2[d]*v1+(rjk-rij*cos_a2)*(res[d]+tmp[d]))/(rij*rjk*sin_a2*sin_a2);
      v1=(delta_fun2(ki,kj,ii)*eji[m]*rjk+rij*delta_fun2(kj,kk,ii)*ekj[m])*sin_a2*sin_a2;
      v1=v1+rij*rjk*2*sin_a2*cos_a2*B_mat[i*3+1].colB[ii*3+m];
      v1=(v1*(rjk-rij*cos_a2)/(rij*rjk*sin_a2*sin_a2))/(rij*rjk*sin_a2*sin_a2);
      for(d=0;d<DIM;d++) res[d]=res[d]-tmp2[d]*v1;
      for(d=0;d<DIM;d++) tmp3[d]=delta_fun2(kj,kk,ii)*(delta_fun(d,m)-ekj[d]*ekj[m])/rjk;
      cprod(elk,tmp3,tmp);
      for(d=0;d<DIM;d++) tmp3[d]=delta_fun2(kk,kl,ii)*(delta_fun(d,m)-elk[d]*elk[m])/rkl;
      cprod(tmp3,ekj,tmp2);
      for(d=0;d<DIM;d++) tmp[d]=cos_a3*(tmp[d]+tmp2[d]);
      cprod(elk,ekj,tmp2);
      for(d=0;d<DIM;d++) res[d]=res[d]+(tmp[d]-sin_a3*mat[ii][m]*tmp2[d])/(rjk*sin_a3*sin_a3);
      v1=sin_a3*sin_a3*delta_fun2(kj,kk,ii)*ekj[m]+rjk*2*sin_a3*cos_a3*mat[ii][m];
      v1=(v1*cos_a3/(rjk*sin_a3*sin_a3))/(rjk*sin_a3*sin_a3);
      for(d=0;d<DIM;d++) rirm[ii][d][m]=res[d]-v1*tmp2[d];
     }
   }
}

static void dih_rkrm(real cos_a2,real cos_a3,rvec eji,rvec ekj,rvec elk,real rij,real rjk,real rkl,t_Bmat *B_mat,real **mat,int ki,int kj,int kk,int kl,int i,real ***rirm)
{
   real sin_a2, sin_a3, v1;
   int ii,d,m;
   rvec res, tmp, tmp2, tmp3;

   sin_a2=sqrt(1-cos_a2*cos_a2);
   sin_a3=sqrt(1-cos_a3*cos_a3);
   for(ii=0;ii<4;ii++){
     for(m=0;m<DIM;m++){
      cprod(elk,ekj,tmp2);
      v1=delta_fun2(kj,kk,ii)*ekj[m]-cos_a3*delta_fun2(kk,kl,ii)*elk[m]+rkl*sin_a3*mat[ii][m];
      for(d=0;d<DIM;d++) tmp3[d]=delta_fun2(kk,kl,ii)*(delta_fun(d,m)-elk[d]*elk[m])/rkl;
      cprod(tmp3,ekj,res);
      for(d=0;d<DIM;d++) tmp3[d]=delta_fun2(kj,kk,ii)*(delta_fun(d,m)-ekj[d]*ekj[m])/rjk;
      cprod(elk,tmp3,tmp);
      for(d=0;d<DIM;d++) res[d]=(tmp2[d]*v1+(rjk-rkl*cos_a3)*(res[d]+tmp[d]))/(rkl*rjk*sin_a3*sin_a3);
      v1=(delta_fun2(kk,kl,ii)*elk[m]*rjk+rkl*delta_fun2(kj,kk,ii)*ekj[m])*sin_a3*sin_a3;
      v1=v1+rkl*rjk*2*sin_a3*cos_a3*mat[ii][m];
      v1=(v1*(rjk-rkl*cos_a3)/(rkl*rjk*sin_a3*sin_a3))/(rkl*rjk*sin_a3*sin_a3);
      for(d=0;d<DIM;d++) res[d]=res[d]-tmp2[d]*v1;
      for(d=0;d<DIM;d++) tmp3[d]=delta_fun2(kj,kk,ii)*(delta_fun(d,m)-ekj[d]*ekj[m])/rjk;
      cprod(eji,tmp3,tmp);
      for(d=0;d<DIM;d++) tmp3[d]=delta_fun2(ki,kj,ii)*(delta_fun(d,m)-eji[d]*eji[m])/rij;
      cprod(tmp3,ekj,tmp2);
      for(d=0;d<DIM;d++) tmp[d]=cos_a2*(tmp[d]+tmp2[d]);
      cprod(eji,ekj,tmp2);
      for(d=0;d<DIM;d++) res[d]=res[d]+(tmp[d]-sin_a2*B_mat[i*3+1].colB[ii*3+m]*tmp2[d])/(rjk*sin_a2*sin_a2);
      v1=sin_a2*sin_a2*delta_fun2(kj,kk,ii)*ekj[m]+rjk*2*sin_a2*cos_a2*B_mat[i*3+1].colB[ii*3+m];
      v1=(v1*cos_a2/(rjk*sin_a2*sin_a2))/(rjk*sin_a2*sin_a2);
      for(d=0;d<DIM;d++) rirm[ii][d][m]=res[d]-v1*tmp2[d];
     }
   }
}

static void sub_calB_matrix(rvec *xx, BKStree_t *tr, int i, t_Bmat *B_mat, t_pbc *pbc)
{
  real O0[]={0.0,0.0,0.0}, x0[]={0.1,0.0,0.0}, y0[]={0.0,0.1,0.0}, z0[]={0.0,0.0,0.1};
  int n,ni,nj,nk,nl,ii;
  BKSnode_t *node;
  rvec xi,xj,xk,xl;
  real rij,rjk,rkl;
  rvec eji,ekj,ejk,elk;
  int m,d,ki,kj,kk,kl=0;
  rvec st1,st2,st3,st4;
  real cos_a2,cos_a3;
  gmx_bool bImp=FALSE;
  //int ind[4];
  real ***rirm;
  real **lmat;
  snew_mat(lmat,4,3,d);

  snew_mat(rirm,4,3,d); 
  for(d=0;d<4;d++)
    for(m=0;m<DIM;m++)
      snew(rirm[d][m],3);

  rvec_inc(O0,tr->O0);  /* set the origin of the lab-fixed frame */
  rvec_inc(x0,tr->O0);
  rvec_inc(y0,tr->O0);
  rvec_inc(z0,tr->O0);

  node=&tr->nodes[i];
  n=node->nEle;
    /* Here, bond angle and dihedral are external coordinates and thus the first and second-order B-matrix elements are not used */
/*  if(n==1){
    ni=node->ijkl[0];
    copy_rvec(xx[ni],xi);
    copy_rvec(O0,xj);
    copy_rvec(z0,xk);
    copy_rvec(x0,xl);
    ki=node->ele[n-1];
    unit_e(pbc,xi,xj,eji);
    unit_e(pbc,xj,xk,ekj);
    rij=cal_bond_length(pbc,xi,xj);
    cos_a2=-1.0*iprod(eji,ekj);
    for(d=0;d<DIM;d++)  B_mat[i*3][ki*3+d]=eji[d];
    ang_st1(cos_a2,eji,ekj,rij,st1);
    for(d=0;d<DIM;d++)  B_mat[i*3+1][ki*3+d]=st1[d];
    dih_st1(cos_a2,eji,ekj,rij,st1);
    for(d=0;d<DIM;d++)  B_mat[i*3+2][ki*3+d]=st1[d];
  }*/
  if(n==2){
    ni=node->ijkl[0];
    nj=node->ijkl[1];
    copy_rvec(xx[ni],xi);
    copy_rvec(xx[nj],xj);
    copy_rvec(O0,xk);
    copy_rvec(z0,xl);
    ki=node->ele[n-1];
    kj=node->ele[n-2];
    unit_e(pbc,xi,xj,eji);
    unit_e(pbc,xj,xk,ekj);
    unit_e(pbc,xk,xl,elk);
    rij=cal_bond_length(pbc,xi,xj);
    rjk=cal_bond_length(pbc,xj,xk);
    cos_a2=-1.0*iprod(eji,ekj);
    cos_a3=-1.0*iprod(ekj,elk);
    B_mat[i*3].indB[0]=ki;
    B_mat[i*3].indB[1]=kj;
    for(d=0;d<DIM;d++){
      B_mat[i*3].colB[d]=eji[d];
      B_mat[i*3].colB[3+d]=-eji[d];
      //B_mat[i*3][ki*3+d]=eji[d];
      //B_mat[i*3][kj*3+d]=-eji[d];
    }
    /* Calculate the non-zero elements of the second order B-matrix for bonds */
    for(d=0;d<DIM;d++){
      for(m=0;m<DIM;m++){
        B_mat[i*3].matB[d][m]=(delta_fun(d,m)-eji[d]*eji[m])/rij;
        B_mat[i*3].matB[d][3+m]=-B_mat[i*3].matB[d][m];
        B_mat[i*3].matB[3+d][m]=B_mat[i*3].matB[d][3+m];
        B_mat[i*3].matB[3+d][3+m]=B_mat[i*3].matB[d][m];
        //BB_mat[i*3][ki*3+d][ki*3+m]=(delta_fun(d,m)-eji[d]*eji[m])/rij;
        //BB_mat[i*3][ki*3+d][kj*3+m]=-BB_mat[i*3][ki*3+d][ki*3+m];
        //BB_mat[i*3][kj*3+d][ki*3+m]=BB_mat[i*3][ki*3+d][kj*3+m];
        //BB_mat[i*3][kj*3+d][kj*3+m]=BB_mat[i*3][ki*3+d][ki*3+m];
      }
    }
    /* Here, angle and dihedral are external coordinates and thus the first and second-order B-matrix elements are not used */
/*    ang_st1(cos_a2,eji,ekj,rij,st1);
    ang_st2(cos_a2,eji,ekj,rij,rjk,st2);
    for(d=0;d<DIM;d++){
      B_mat[i*3+1][ki*3+d]=st1[d];
      B_mat[i*3+1][kj*3+d]=st2[d];
    }
    dih_st1(cos_a2,eji,ekj,rij,st1);
    dih_st2(cos_a2,cos_a3,eji,ekj,elk,rij,rjk,st2);
    for(d=0;d<DIM;d++){
      B_mat[i*3+2][ki*3+d]=st1[d];
      B_mat[i*3+2][kj*3+d]=st2[d];
    }
*/
  }
  if(n==3){
    ni=node->ijkl[0];
    nj=node->ijkl[1];
    nk=node->ijkl[2];
    copy_rvec(xx[ni],xi);
    copy_rvec(xx[nj],xj);
    copy_rvec(xx[nk],xk);
    ki=node->ele[n-1];
    kj=node->ele[n-2];
    kk=node->ele[n-3];
    if((tr->nodes[kj].nEdges>2) && (ki != (kj+1))){
      bImp=TRUE;
      nl=node->ijkl[3];
      copy_rvec(xx[nl],xl);
    }else{
      copy_rvec(O0,xl);
    }
    if(bImp) kl=kj+1;
    unit_e(pbc,xi,xj,eji);
    unit_e(pbc,xj,xk,ekj);
    unit_e(pbc,xk,xl,elk);
    rij=cal_bond_length(pbc,xi,xj);
    rjk=cal_bond_length(pbc,xj,xk);
    rkl=cal_bond_length(pbc,xk,xl);
    cos_a2=-1.0*iprod(eji,ekj);
    cos_a3=-1.0*iprod(ekj,elk);
    B_mat[i*3].indB[0]=ki;
    B_mat[i*3].indB[1]=kj;
    for(d=0;d<DIM;d++){
      B_mat[i*3].colB[d]=eji[d];
      B_mat[i*3].colB[3+d]=-eji[d];
      //B_mat[i*3][ki*3+d]=eji[d];
      //B_mat[i*3][kj*3+d]=-eji[d];
    }
    /* Calculate the non-zero elements of the second order B-matrix for bonds */
    for(d=0;d<DIM;d++){
      for(m=0;m<DIM;m++){
        B_mat[i*3].matB[d][m]=(delta_fun(d,m)-eji[d]*eji[m])/rij;
        B_mat[i*3].matB[d][3+m]=-B_mat[i*3].matB[d][m];
        B_mat[i*3].matB[3+d][m]=B_mat[i*3].matB[d][3+m];
        B_mat[i*3].matB[3+d][3+m]=B_mat[i*3].matB[d][m];
        //BB_mat[i*3][ki*3+d][ki*3+m]=(delta_fun(d,m)-eji[d]*eji[m])/rij;
        //BB_mat[i*3][ki*3+d][kj*3+m]=-BB_mat[i*3][ki*3+d][ki*3+m];
        //BB_mat[i*3][kj*3+d][ki*3+m]=BB_mat[i*3][ki*3+d][kj*3+m];
        //BB_mat[i*3][kj*3+d][kj*3+m]=BB_mat[i*3][ki*3+d][ki*3+m];
      }
    }
    ang_st1(cos_a2,eji,ekj,rij,st1);
    ang_st2(cos_a2,eji,ekj,rij,rjk,st2);
    ang_st3(cos_a2,eji,ekj,rjk,st3);
    B_mat[i*3+1].indB[0]=ki;
    B_mat[i*3+1].indB[1]=kj;
    B_mat[i*3+1].indB[2]=kk;
    for(d=0;d<DIM;d++){
      B_mat[i*3+1].colB[d]=st1[d];
      B_mat[i*3+1].colB[3+d]=st2[d];
      B_mat[i*3+1].colB[6+d]=st3[d];
      //B_mat[i*3+1][ki*3+d]=st1[d];
      //B_mat[i*3+1][kj*3+d]=st2[d];
      //B_mat[i*3+1][kk*3+d]=st3[d];
    }
    /* Calculate the non-zero elements of the second order B-matrix for angles */
    //ind[0]=ki; ind[1]=kj; ind[2]=kk;
    svmul(-1.0,ekj,ejk);
    ang_rirm(cos_a2,eji,ejk,rij,rjk,B_mat[i*3+1].colB,0,1,2,rirm);
    for(ii=0;ii<DIM;ii++){
     //km=ind[ii];
     for(d=0;d<DIM;d++){
      for(m=0;m<DIM;m++){
        B_mat[i*3+1].matB[d][ii*3+m]=rirm[ii][d][m];
        //BB_mat[i*3+1][ki*3+d][km*3+m]=rirm[ii][d][m];
      }
     }
    }
    /* partial2_q[alpha]/partial_rk*partial_rm */
    ang_rirm(cos_a2,ejk,eji,rjk,rij,B_mat[i*3+1].colB,2,1,0,rirm);
    for(ii=0;ii<DIM;ii++){
     //km=ind[ii];
     for(d=0;d<DIM;d++){
      for(m=0;m<DIM;m++){
        B_mat[i*3+1].matB[6+d][ii*3+m]=rirm[ii][d][m];
        //BB_mat[i*3+1][kk*3+d][km*3+m]=rirm[ii][d][m];
      }
     }
    }

    /* partial2_q[alpha]/partial_rj*partial_rm */
    ang_rjrm(cos_a2,eji,ejk,rij,rjk,B_mat[i*3+1].colB,0,1,2,rirm);
    for(ii=0;ii<DIM;ii++){
     //km=ind[ii];
     for(d=0;d<DIM;d++){
      for(m=0;m<DIM;m++){
        B_mat[i*3+1].matB[3+d][ii*3+m]=rirm[ii][d][m];
        //BB_mat[i*3+1][kj*3+d][km*3+m]=rirm[ii][d][m];
      }
     }
    }
  if(bImp){ /* Otherwise, dihedral is an external coordinate and thus the first and second-order B-matrix elements are not used */
    dih_st1(cos_a2,eji,ekj,rij,st1);
    dih_st2(cos_a2,cos_a3,eji,ekj,elk,rij,rjk,st2);
    dih_st2(cos_a3,cos_a2,elk,ekj,eji,rkl,rjk,st3);
    dih_st1(cos_a3,elk,ekj,rkl,st4);
    B_mat[i*3+2].indB[0]=ki;
    B_mat[i*3+2].indB[1]=kj;
    B_mat[i*3+2].indB[2]=kk;
    B_mat[i*3+2].indB[3]=kl;
    for(d=0;d<DIM;d++){
      B_mat[i*3+2].colB[d]=st1[d];
      B_mat[i*3+2].colB[3+d]=st2[d];
      B_mat[i*3+2].colB[6+d]=st3[d];
      B_mat[i*3+2].colB[9+d]=st4[d];
      //B_mat[i*3+2][ki*3+d]=st1[d];
      //B_mat[i*3+2][kj*3+d]=st2[d];
      //B_mat[i*3+2][kk*3+d]=st3[d];
      //B_mat[i*3+2][kl*3+d]=st4[d];
    }

    /* Calculate the non-zero elements of the second order B-matrix for dihedrals */
    ang_st1(cos_a3,ekj,elk,rjk,st1);
    ang_st2(cos_a3,ekj,elk,rjk,rkl,st2);
    ang_st3(cos_a3,ekj,elk,rkl,st3);
    for(d=0;d<DIM;d++){
      lmat[0][d]=0.0;
      lmat[1][d]=st1[d];
      lmat[2][d]=st2[d];
      lmat[3][d]=st3[d];
    }
    //ind[0]=ki; ind[1]=kj; ind[2]=kk; ind[3]=kl;
    dih_rirm(cos_a2,eji,ekj,rij,rjk,B_mat,0,1,2,i,rirm);
    for(ii=0;ii<DIM;ii++){
     //km=ind[ii];
     for(d=0;d<DIM;d++){
      for(m=0;m<DIM;m++){
        B_mat[i*3+2].matB[d][ii*3+m]=rirm[ii][d][m];
        //BB_mat[i*3+2][ki*3+d][km*3+m]=rirm[ii][d][m];
      }
     }
    }

    dih_rlrm(cos_a3,elk,ekj,rkl,rjk,B_mat,lmat,1,2,3,i,rirm);
    for(ii=1;ii<4;ii++){
     //km=ind[ii];
     for(d=0;d<DIM;d++){
      for(m=0;m<DIM;m++){
        B_mat[i*3+2].matB[9+d][ii*3+m]=rirm[ii][d][m];
        //BB_mat[i*3+2][kl*3+d][km*3+m]=rirm[ii][d][m];
      }
     }
    }

    dih_rjrm(cos_a2,cos_a3,eji,ekj,elk,rij,rjk,rkl,B_mat,lmat,0,1,2,3,i,rirm);
    for(ii=0;ii<4;ii++){
     //km=ind[ii];
     for(d=0;d<DIM;d++){
      for(m=0;m<DIM;m++){
        B_mat[i*3+2].matB[3+d][ii*3+m]=rirm[ii][d][m];
        //BB_mat[i*3+2][kj*3+d][km*3+m]=rirm[ii][d][m];
      }
     }
    }

    dih_rkrm(cos_a2,cos_a3,eji,ekj,elk,rij,rjk,rkl,B_mat,lmat,0,1,2,3,i,rirm);
    for(ii=0;ii<4;ii++){
     //km=ind[ii];
     for(d=0;d<DIM;d++){
      for(m=0;m<DIM;m++){
        B_mat[i*3+2].matB[6+d][ii*3+m]=rirm[ii][d][m];
        //BB_mat[i*3+2][kk*3+d][km*3+m]=rirm[ii][d][m];
      }
     }
    }
   } /* if(bImp) */
  }
  if(n>=4){
    ni=node->ijkl[0];
    nj=node->ijkl[1];
    nk=node->ijkl[2];
    nl=node->ijkl[3];
    copy_rvec(xx[ni],xi);
    copy_rvec(xx[nj],xj);
    copy_rvec(xx[nk],xk);
    copy_rvec(xx[nl],xl);
    ki=node->ele[n-1];
    kj=node->ele[n-2];
    kk=node->ele[n-3];
    if((tr->nodes[kj].nEdges>2) && (ki != (kj+1)))
      kl=kj+1;
    else kl=node->ele[n-4];
    unit_e(pbc,xi,xj,eji);
    unit_e(pbc,xj,xk,ekj);
    unit_e(pbc,xk,xl,elk);
    rij=cal_bond_length(pbc,xi,xj);
    rjk=cal_bond_length(pbc,xj,xk);
    rkl=cal_bond_length(pbc,xk,xl);
    cos_a2=-1.0*iprod(eji,ekj);
    cos_a3=-1.0*iprod(ekj,elk);
    B_mat[i*3].indB[0]=ki;
    B_mat[i*3].indB[1]=kj;
    for(d=0;d<DIM;d++){
      B_mat[i*3].colB[d]=eji[d];
      B_mat[i*3].colB[3+d]=-eji[d];
      //B_mat[i*3][ki*3+d]=eji[d];
      //B_mat[i*3][kj*3+d]=-eji[d];
    }
    /* Calculate the non-zero elements of the second order B-matrix for bonds */
    for(d=0;d<DIM;d++){
      for(m=0;m<DIM;m++){
        B_mat[i*3].matB[d][m]=(delta_fun(d,m)-eji[d]*eji[m])/rij;
        B_mat[i*3].matB[d][3+m]=-B_mat[i*3].matB[d][m];
        B_mat[i*3].matB[3+d][m]=B_mat[i*3].matB[d][3+m];
        B_mat[i*3].matB[3+d][3+m]=B_mat[i*3].matB[d][m];
        //BB_mat[i*3][ki*3+d][ki*3+m]=(delta_fun(d,m)-eji[d]*eji[m])/rij;
        //BB_mat[i*3][ki*3+d][kj*3+m]=-BB_mat[i*3][ki*3+d][ki*3+m];
        //BB_mat[i*3][kj*3+d][ki*3+m]=BB_mat[i*3][ki*3+d][kj*3+m];
        //BB_mat[i*3][kj*3+d][kj*3+m]=BB_mat[i*3][ki*3+d][ki*3+m];
      }
    }
    ang_st1(cos_a2,eji,ekj,rij,st1);
    ang_st2(cos_a2,eji,ekj,rij,rjk,st2);
    ang_st3(cos_a2,eji,ekj,rjk,st3);
    B_mat[i*3+1].indB[0]=ki;
    B_mat[i*3+1].indB[1]=kj;
    B_mat[i*3+1].indB[2]=kk;
    for(d=0;d<DIM;d++){
      B_mat[i*3+1].colB[d]=st1[d];
      B_mat[i*3+1].colB[3+d]=st2[d];
      B_mat[i*3+1].colB[6+d]=st3[d];
      //B_mat[i*3+1][ki*3+d]=st1[d];
      //B_mat[i*3+1][kj*3+d]=st2[d];
      //B_mat[i*3+1][kk*3+d]=st3[d];
    }
    ang_st1(cos_a3,ekj,elk,rjk,st1);
    ang_st2(cos_a3,ekj,elk,rjk,rkl,st2);
    ang_st3(cos_a3,ekj,elk,rkl,st3);
    for(d=0;d<DIM;d++){
      lmat[0][d]=0.0;
      lmat[1][d]=st1[d];
      lmat[2][d]=st2[d];
      lmat[3][d]=st3[d];
    }
    /* Calculate the non-zero elements of the second order B-matrix for angles */
    //ind[0]=ki; ind[1]=kj; ind[2]=kk;
    svmul(-1.0,ekj,ejk);
    ang_rirm(cos_a2,eji,ejk,rij,rjk,B_mat[i*3+1].colB,0,1,2,rirm);
    for(ii=0;ii<DIM;ii++){
     //km=ind[ii];
     for(d=0;d<DIM;d++){
      for(m=0;m<DIM;m++){
        B_mat[i*3+1].matB[d][ii*3+m]=rirm[ii][d][m];
        //BB_mat[i*3+1][ki*3+d][km*3+m]=rirm[ii][d][m];
      }
     }
    }
    /* partial2_q[alpha]/partial_rk*partial_rm */
    ang_rirm(cos_a2,ejk,eji,rjk,rij,B_mat[i*3+1].colB,2,1,0,rirm);
    for(ii=0;ii<DIM;ii++){
     //km=ind[ii];
     for(d=0;d<DIM;d++){
      for(m=0;m<DIM;m++){
        B_mat[i*3+1].matB[6+d][ii*3+m]=rirm[ii][d][m];
        //BB_mat[i*3+1][kk*3+d][km*3+m]=rirm[ii][d][m];
      }
     }
    }

    /* partial2_q[alpha]/partial_rj*partial_rm */
    ang_rjrm(cos_a2,eji,ejk,rij,rjk,B_mat[i*3+1].colB,0,1,2,rirm);
    for(ii=0;ii<DIM;ii++){
     //km=ind[ii];
     for(d=0;d<DIM;d++){
      for(m=0;m<DIM;m++){
        B_mat[i*3+1].matB[3+d][ii*3+m]=rirm[ii][d][m];
        //BB_mat[i*3+1][kj*3+d][km*3+m]=rirm[ii][d][m];
      }
     }
    }

    dih_st1(cos_a2,eji,ekj,rij,st1);
    dih_st2(cos_a2,cos_a3,eji,ekj,elk,rij,rjk,st2);
    dih_st2(cos_a3,cos_a2,elk,ekj,eji,rkl,rjk,st3);
    dih_st1(cos_a3,elk,ekj,rkl,st4);
    B_mat[i*3+2].indB[0]=ki;
    B_mat[i*3+2].indB[1]=kj;
    B_mat[i*3+2].indB[2]=kk;
    B_mat[i*3+2].indB[3]=kl;
    for(d=0;d<DIM;d++){
      B_mat[i*3+2].colB[d]=st1[d];
      B_mat[i*3+2].colB[3+d]=st2[d];
      B_mat[i*3+2].colB[6+d]=st3[d];
      B_mat[i*3+2].colB[9+d]=st4[d];
      //B_mat[i*3+2][ki*3+d]=st1[d];
      //B_mat[i*3+2][kj*3+d]=st2[d];
      //B_mat[i*3+2][kk*3+d]=st3[d];
      //B_mat[i*3+2][kl*3+d]=st4[d];
    }

    /* Calculate the non-zero elements of the second order B-matrix for dihedrals */
    //ind[0]=ki; ind[1]=kj; ind[2]=kk; ind[3]=kl;
    dih_rirm(cos_a2,eji,ekj,rij,rjk,B_mat,0,1,2,i,rirm);
    for(ii=0;ii<DIM;ii++){
     //km=ind[ii];
     for(d=0;d<DIM;d++){
      for(m=0;m<DIM;m++){
        B_mat[i*3+2].matB[d][ii*3+m]=rirm[ii][d][m];
        //BB_mat[i*3+2][ki*3+d][km*3+m]=rirm[ii][d][m];
      }
     }
    }

    dih_rlrm(cos_a3,elk,ekj,rkl,rjk,B_mat,lmat,1,2,3,i,rirm);
    for(ii=1;ii<4;ii++){
     //km=ind[ii];
     for(d=0;d<DIM;d++){
      for(m=0;m<DIM;m++){
        B_mat[i*3+2].matB[9+d][ii*3+m]=rirm[ii][d][m];
        //BB_mat[i*3+2][kl*3+d][km*3+m]=rirm[ii][d][m];
      }
     }
    }

    dih_rjrm(cos_a2,cos_a3,eji,ekj,elk,rij,rjk,rkl,B_mat,lmat,0,1,2,3,i,rirm);
    for(ii=0;ii<4;ii++){
     //km=ind[ii];
     for(d=0;d<DIM;d++){
      for(m=0;m<DIM;m++){
        B_mat[i*3+2].matB[3+d][ii*3+m]=rirm[ii][d][m];
        //BB_mat[i*3+2][kj*3+d][km*3+m]=rirm[ii][d][m];
      }
     }
    }

    dih_rkrm(cos_a2,cos_a3,eji,ekj,elk,rij,rjk,rkl,B_mat,lmat,0,1,2,3,i,rirm);
    for(ii=0;ii<4;ii++){
     //km=ind[ii];
     for(d=0;d<DIM;d++){
      for(m=0;m<DIM;m++){
        B_mat[i*3+2].matB[6+d][ii*3+m]=rirm[ii][d][m];
        //BB_mat[i*3+2][kk*3+d][km*3+m]=rirm[ii][d][m];
      }
     }
    }
  }

  for(d=0;d<4;d++)
    for(m=0;m<DIM;m++)
      sfree(rirm[d][m]);
  sfree_mat(rirm,4,d);
  sfree_mat(lmat,4,d);
}

static void calB_matrix(rvec *xx, BKStree_t *tr, t_Bmat *B_mat, t_pbc *pbc)
{
  int i,j,k,M;

  M=tr->N;

  /* clean B_mat & BB_mat */
  clean_Bmats(B_mat,3*M);

  for(j=0;j<M;j++){
    sub_calB_matrix(xx,tr,j,B_mat,pbc);
  }
}

static void sub_cal_geneVel(rvec *xx, BKStree_t *tr, int i, rvec *dotQ, t_pbc *pbc, rvec *vv)
{
  real O0[]={0.0,0.0,0.0}, x0[]={0.1,0.0,0.0}, y0[]={0.0,0.1,0.0}, z0[]={0.0,0.0,0.1};
  int n,ni,nj,nk,nl,ii;
  BKSnode_t *node;
  rvec xi,xj,xk,xl;
  real rij,rjk,rkl;
  rvec eji,ekj,ejk,elk;
  int m,d,ki,kj,kk,kl=0,km;
  rvec st1,st2,st3,st4;
  real cos_a2,cos_a3;
  gmx_bool bImp=FALSE;

  rvec_inc(O0,tr->O0);  /* set the origin of the lab-fixed frame */
  rvec_inc(x0,tr->O0);
  rvec_inc(y0,tr->O0);
  rvec_inc(z0,tr->O0);

  node=&tr->nodes[i];
  n=node->nEle;
/* when n==1, they are external coordinates and their dotQ were estimated separately */
/*  if(n==1){
    ni=node->ijkl[0];
    copy_rvec(xx[ni],xi);
    copy_rvec(O0,xj);
    copy_rvec(z0,xk);
    copy_rvec(x0,xl);
    ki=node->ele[n-1];
    unit_e(pbc,xi,xj,eji);
    unit_e(pbc,xj,xk,ekj);
    rij=cal_bond_length(pbc,xi,xj);
    cos_a2=-1.0*iprod(eji,ekj);
    for(d=0;d<DIM;d++)  B_mat[i*3][ki*3+d]=eji[d];
    ang_st1(cos_a2,eji,ekj,rij,st1);
    for(d=0;d<DIM;d++)  B_mat[i*3+1][ki*3+d]=st1[d];
    dih_st1(cos_a2,eji,ekj,rij,st1);
    for(d=0;d<DIM;d++)  B_mat[i*3+2][ki*3+d]=st1[d];
  }*/
  if(n==2){
    ni=node->ijkl[0];
    nj=node->ijkl[1];
    copy_rvec(xx[ni],xi);
    copy_rvec(xx[nj],xj);
    copy_rvec(O0,xk);
    copy_rvec(z0,xl);
    ki=node->ele[n-1];
    kj=node->ele[n-2];
    unit_e(pbc,xi,xj,eji);
    unit_e(pbc,xj,xk,ekj);
    unit_e(pbc,xk,xl,elk);
    rij=cal_bond_length(pbc,xi,xj);
    rjk=cal_bond_length(pbc,xj,xk);
    cos_a2=-1.0*iprod(eji,ekj);
    cos_a3=-1.0*iprod(ekj,elk);
/*    for(d=0;d<DIM;d++){
      B_mat[i*3][ki*3+d]=eji[d];
      B_mat[i*3][kj*3+d]=-eji[d];
    }
*/
    /* estimate the generalized velocity of the bond */
    dotQ[i][XX]=iprod(vv[ni],eji)-iprod(vv[nj],eji);

/* the angle and dihedral corresponds to the external coordinates */
/*    ang_st1(cos_a2,eji,ekj,rij,st1);
    ang_st2(cos_a2,eji,ekj,rij,rjk,st2);
    for(d=0;d<DIM;d++){
      B_mat[i*3+1][ki*3+d]=st1[d];
      B_mat[i*3+1][kj*3+d]=st2[d];
    }
    dih_st1(cos_a2,eji,ekj,rij,st1);
    dih_st2(cos_a2,cos_a3,eji,ekj,elk,rij,rjk,st2);
    for(d=0;d<DIM;d++){
      B_mat[i*3+2][ki*3+d]=st1[d];
      B_mat[i*3+2][kj*3+d]=st2[d];
    }
*/
  }
  if(n==3){
    ni=node->ijkl[0];
    nj=node->ijkl[1];
    nk=node->ijkl[2];
    copy_rvec(xx[ni],xi);
    copy_rvec(xx[nj],xj);
    copy_rvec(xx[nk],xk);
    ki=node->ele[n-1];
    kj=node->ele[n-2];
    kk=node->ele[n-3];
    if((tr->nodes[kj].nEdges>2) && (ki != (kj+1))){
      bImp=TRUE;
      nl=node->ijkl[3];
      copy_rvec(xx[nl],xl);
    }else{
      copy_rvec(O0,xl);
    }
    if(bImp) kl=kj+1;
    unit_e(pbc,xi,xj,eji);
    unit_e(pbc,xj,xk,ekj);
    unit_e(pbc,xk,xl,elk);
    rij=cal_bond_length(pbc,xi,xj);
    rjk=cal_bond_length(pbc,xj,xk);
    rkl=cal_bond_length(pbc,xk,xl);
    cos_a2=-1.0*iprod(eji,ekj);
    cos_a3=-1.0*iprod(ekj,elk);
/*    for(d=0;d<DIM;d++){
      B_mat[i*3][ki*3+d]=eji[d];
      B_mat[i*3][kj*3+d]=-eji[d];
    }
*/
    /* estimate the generalized velocity of the bond */
    dotQ[i][XX]=iprod(vv[ni],eji)-iprod(vv[nj],eji);

    ang_st1(cos_a2,eji,ekj,rij,st1);
    ang_st2(cos_a2,eji,ekj,rij,rjk,st2);
    ang_st3(cos_a2,eji,ekj,rjk,st3);
/*    for(d=0;d<DIM;d++){
      B_mat[i*3+1][ki*3+d]=st1[d];
      B_mat[i*3+1][kj*3+d]=st2[d];
      B_mat[i*3+1][kk*3+d]=st3[d];
    }
*/
    /* estimate the generalized velocity of the angle */
    dotQ[i][YY]=iprod(vv[ni],st1)+iprod(vv[nj],st2)+iprod(vv[nk],st3);

    dih_st1(cos_a2,eji,ekj,rij,st1);
    dih_st2(cos_a2,cos_a3,eji,ekj,elk,rij,rjk,st2);
    dih_st2(cos_a3,cos_a2,elk,ekj,eji,rkl,rjk,st3);
    if(bImp) dih_st1(cos_a3,elk,ekj,rkl,st4);
/*    for(d=0;d<DIM;d++){
      B_mat[i*3+2][ki*3+d]=st1[d];
      B_mat[i*3+2][kj*3+d]=st2[d];
      B_mat[i*3+2][kk*3+d]=st3[d];
      if(bImp) B_mat[i*3+2][kl*3+d]=st4[d];
    }
*/
    /* estimate the generalized velocity of the improper dihedral */
    if(bImp) dotQ[i][ZZ]=iprod(vv[ni],st1)+iprod(vv[nj],st2)+iprod(vv[nk],st3)+iprod(vv[nl],st4);
  }
  if(n>=4){
    ni=node->ijkl[0];
    nj=node->ijkl[1];
    nk=node->ijkl[2];
    nl=node->ijkl[3];
    copy_rvec(xx[ni],xi);
    copy_rvec(xx[nj],xj);
    copy_rvec(xx[nk],xk);
    copy_rvec(xx[nl],xl);
    ki=node->ele[n-1];
    kj=node->ele[n-2];
    kk=node->ele[n-3];
    if((tr->nodes[kj].nEdges>2) && (ki != (kj+1)))
      kl=kj+1;
    else kl=node->ele[n-4];
    unit_e(pbc,xi,xj,eji);
    unit_e(pbc,xj,xk,ekj);
    unit_e(pbc,xk,xl,elk);
    rij=cal_bond_length(pbc,xi,xj);
    rjk=cal_bond_length(pbc,xj,xk);
    rkl=cal_bond_length(pbc,xk,xl);
    cos_a2=-1.0*iprod(eji,ekj);
    cos_a3=-1.0*iprod(ekj,elk);
/*    for(d=0;d<DIM;d++){
      B_mat[i*3][ki*3+d]=eji[d];
      B_mat[i*3][kj*3+d]=-eji[d];
    }
*/
    /* estimate the generalized velocity of the bond */
    dotQ[i][XX]=iprod(vv[ni],eji)-iprod(vv[nj],eji);

    ang_st1(cos_a2,eji,ekj,rij,st1);
    ang_st2(cos_a2,eji,ekj,rij,rjk,st2);
    ang_st3(cos_a2,eji,ekj,rjk,st3);
/*    for(d=0;d<DIM;d++){
      B_mat[i*3+1][ki*3+d]=st1[d];
      B_mat[i*3+1][kj*3+d]=st2[d];
      B_mat[i*3+1][kk*3+d]=st3[d];
    }
*/
    /* estimate the generalized velocity of the angle */
    dotQ[i][YY]=iprod(vv[ni],st1)+iprod(vv[nj],st2)+iprod(vv[nk],st3);

    dih_st1(cos_a2,eji,ekj,rij,st1);
    dih_st2(cos_a2,cos_a3,eji,ekj,elk,rij,rjk,st2);
    dih_st2(cos_a3,cos_a2,elk,ekj,eji,rkl,rjk,st3);
    dih_st1(cos_a3,elk,ekj,rkl,st4);
/*    for(d=0;d<DIM;d++){
      B_mat[i*3+2][ki*3+d]=st1[d];
      B_mat[i*3+2][kj*3+d]=st2[d];
      B_mat[i*3+2][kk*3+d]=st3[d];
      B_mat[i*3+2][kl*3+d]=st4[d];
    }
*/
    /* estimate the generalized velocity of the dihedral */
    dotQ[i][ZZ]=iprod(vv[ni],st1)+iprod(vv[nj],st2)+iprod(vv[nk],st3)+iprod(vv[nl],st4);
  }
}

static void cal_ext_dotQ(t_topology *top,rvec *dotQ,rvec *vv,rvec *xp,BKStree_t *tr)
{
  int i,j,d,alpha;
  int M=tr->N;
  real m0,tm=0.0;
  rvec tmp,tmp2;

  /* get the total mass */
  for(i=0;i<M;i++){
    alpha=tr->nodes[i].iAtom;
    tm+=top->atoms.atom[alpha].m;
  }

  /* translational part */
  clear_rvec(tmp);
  for(i=0;i<M;i++){
    alpha=tr->nodes[i].iAtom;
    m0=top->atoms.atom[alpha].m;
    svmul(m0,vv[alpha],tmp2);
    rvec_inc(tmp,tmp2);
  } 
  svmul(1/tm,tmp,tmp);
  copy_rvec(tmp,dotQ[0]);

  /* rotational part */
  clear_rvec(tmp);
  for(i=0;i<M;i++){
    alpha=tr->nodes[i].iAtom;
    m0=top->atoms.atom[alpha].m;
    cprod(xp[i],vv[alpha],tmp2);
    svmul(m0,tmp2,tmp2);
    rvec_inc(tmp,tmp2);
  }

  /* (I^-1)*tmp */
/*  clear_rvec(tmp2);
  for(j=0;j<DIM;j++){
    for(d=0;d<DIM;d++){
      tmp2[j]+=inv_inert[j][d]*tmp[d];
    }
  }
  dotQ[2][2]=tmp2[0];
  dotQ[1][1]=tmp2[1];
  dotQ[1][2]=tmp2[2];
*/

  /* use rdr as the rotational part coordinate */
  dotQ[2][2]=tmp[0];
  dotQ[1][1]=tmp[1];
  dotQ[1][2]=tmp[2];
}

static void cal_geneVel(rvec *xx,t_topology *top, BKStree_t *tr, rvec *dotQ, t_pbc *pbc, rvec *vv, rvec *xp)
{
  int i,j,M;

  /* clean dotQ */
  M=tr->N;
  for(i=0;i<M;i++){
    for(j=0;j<DIM;j++){
      dotQ[i][j]=0;
    }
  }

  for(i=0;i<M;i++){
    sub_cal_geneVel(xx,tr,i,dotQ,pbc,vv);
  }

  cal_ext_dotQ(top,dotQ,vv,xp,tr);
  
}

/* Here, x is the coordinate relative to COM */
static void calB1_matrix_Euler(t_topology *top,BKStree_t *tr,real **mat_B1,rvec *x,real **inv_inert)
{
  int i,j,k,d,alpha;
  real m0,tm=0.0;
  rvec tmp;
  int  M=tr->N,NR=3*tr->N;
  rvec tvec;

  clear_matrix(mat_B1,6,NR);
  /* get the total mass */
  for(i=0;i<M;i++){
    alpha=tr->nodes[i].iAtom;
    tm+=top->atoms.atom[alpha].m;
  }

  for(i=0;i<M;i++){
    alpha=tr->nodes[i].iAtom;
    m0=top->atoms.atom[alpha].m;
    for(d=0;d<DIM;d++){
      mat_B1[d][3*i+d]=m0/tm; /* the center of mass part of B1-matrix */
    }
    /* Angular momentum part of B1-matrix */
    mat_B1[3][3*i+YY]=-m0*x[i][ZZ];    mat_B1[3][3*i+ZZ]=m0*x[i][YY];   /* Lx */
    mat_B1[4][3*i+XX]=m0*x[i][ZZ];     mat_B1[4][3*i+ZZ]=-m0*x[i][XX];  /* Ly */
    mat_B1[5][3*i+XX]=-m0*x[i][YY];    mat_B1[5][3*i+YY]=m0*x[i][XX];   /* Lz */
  }

  for(i=0;i<NR;i++){
    for(d=0;d<DIM;d++){
      tvec[d]=0;
      for(k=0;k<DIM;k++){ tvec[d]=tvec[d]+inv_inert[d][k]*mat_B1[3+k][i]; }
    }
    for(d=0;d<DIM;d++){ mat_B1[3+d][i]=tvec[d];}
  }
}

/* Here, x is the coordinate relative to COM; in mat_B1, the rotational part use rdr (not Euler-analogue) as the external coordinates */
static void calB1_matrix(t_topology *top,BKStree_t *tr,real **mat_B1,rvec *x)
{
  int i,j,k,d,alpha;
  real m0,tm=0.0;
  rvec tmp;
  int  M=tr->N,NR=3*tr->N;

  clear_matrix(mat_B1,6,NR);
  /* get the total mass */
  for(i=0;i<M;i++){
    alpha=tr->nodes[i].iAtom;
    tm+=top->atoms.atom[alpha].m;
  }

  for(i=0;i<M;i++){
    alpha=tr->nodes[i].iAtom;
    m0=top->atoms.atom[alpha].m;
    for(d=0;d<DIM;d++){
      mat_B1[d][3*i+d]=m0/tm; /* the center of mass part of B1-matrix */
    }
    /* Angular momentum part of B1-matrix */
    mat_B1[3][3*i+YY]=-m0*x[i][ZZ];    mat_B1[3][3*i+ZZ]=m0*x[i][YY];   /* Lx */
    mat_B1[4][3*i+XX]=m0*x[i][ZZ];     mat_B1[4][3*i+ZZ]=-m0*x[i][XX];  /* Ly */
    mat_B1[5][3*i+XX]=-m0*x[i][YY];    mat_B1[5][3*i+YY]=m0*x[i][XX];   /* Lz */
  }
}

/* Here, x is the coordinate relative to COM; in mat_B1, the rotational part use rdr (not Euler-analogue) as the external coordinates */
static void calA1_matrix(BKStree_t *tr,real **mat_A1,rvec *x, real **inv_inert)
{
  int i,j,d;
  int  M=tr->N,NR=3*tr->N;
  real **tmp,**tmp2;

  snew_mat(tmp,3,3,i);
  snew_mat(tmp2,3,3,i);
  clear_matrix(mat_A1,6,NR);

  for(i=0;i<M;i++){
    //alpha=tr->nodes[i].iAtom;
    for(d=0;d<DIM;d++){
      mat_A1[d][3*i+d]=1; /* the center of mass part of A1-matrix */
    }
    /* Angular momentum part of A1-matrix */
    mat_A1[3][3*i+YY]=-x[i][ZZ];    mat_A1[3][3*i+ZZ]=x[i][YY];   /* Lx */
    mat_A1[4][3*i+XX]=x[i][ZZ];     mat_A1[4][3*i+ZZ]=-x[i][XX];  /* Ly */
    mat_A1[5][3*i+XX]=-x[i][YY];    mat_A1[5][3*i+YY]=x[i][XX];   /* Lz */

    for(j=0;j<DIM;j++)
      for(d=0;d<DIM;d++)
        tmp[j][d]=mat_A1[3+j][3*i+d];

    matrix_multi(tmp,inv_inert,3,3,3,tmp2);

    for(j=0;j<DIM;j++)
      for(d=0;d<DIM;d++)
        mat_A1[3+j][3*i+d]=tmp2[j][d];
  }

  sfree_mat(tmp,3,i);
  sfree_mat(tmp2,3,i);
}

static void sub_calAi_mat(real **A1_eta,real **E12,int i,rvec *xx, BKStree_t *tr, rvec *ex, rvec *ey, rvec *ez, real **Akm, t_pbc *pbc, gmx_bool bST, int *sT2)
{
  int j,k,l;
  real tmp;

  if(i<3){
    for(j=0;j<DIM;j++)
      for(k=0;k<3*tr->N;k++)
         Akm[j][k]=A1_eta[i*3+j][k];
  }else{
    calAi_eta(i,xx,tr,ex,ey,ez,Akm,pbc,bST,sT2);
  }

  /* A_(i+6) = A_(i+6) + A_1^eta * E12_i */
  for(j=0;j<DIM;j++){
    for(k=0;k<3*tr->N;k++){
      tmp=0; for(l=0;l<6;l++){ tmp+=A1_eta[l][k]*E12[l][3*i+j-6]; } /* A_1^eta * E12_i */
      Akm[j][k]-=tmp; /* note E12 is the negative of true, thus take minus */
    }
  }
}

static void calAi_mat(real **A1_mat,real **A1_eta,real **E12,int i,rvec *xx, BKStree_t *tr, rvec *ex, rvec *ey, rvec *ez, real **Akm, t_pbc *pbc, gmx_bool bST, int *sT2)
{
  int j,k;

  /* get partial x_k / partial xi */
  if(i<2){
    for(j=0;j<DIM;j++)
      for(k=0;k<3*tr->N;k++)
         Akm[j][k]=A1_mat[i*3+j][k];
  }else{
    sub_calAi_mat(A1_eta,E12,i,xx,tr,ex,ey,ez,Akm,pbc,bST,sT2);
  }
}

static int get_index_BBmat(t_Bmat *Bmat,int j, int i)
{
  int kk;

  for(kk=0;kk<Bmat[j].ncolB;kk++){
    if(i==Bmat[j].indB[kk]) return kk;
  }

  return 200;
}

static real get_ddq_x_qi(t_Bmat *Bmat,real *dxk_qi,int ii,int id)
{
  int kk,dd;
  real res;

  /* summation over only the non-zero elements in secondary B-matrix */
  res=0.0;
  for(kk=0;kk<Bmat->ncolB;kk++){
    for(dd=0;dd<DIM;dd++){
      res+=Bmat->matB[ii*DIM+id][kk*DIM+dd]*dxk_qi[3*Bmat->indB[kk]+dd];
    }
  }

  return res;
}

static real sub_cal_dlnBdqi(t_Bmat *Bmat,real *dxk_qi,real *dxj_qi)
{
  int ii,id; 
  real ddq_x_qi,res;

  /* summation over only the non-zero elements in B-matrix */
  res=0;
  for(ii=0;ii<Bmat->ncolB;ii++){
     for(id=0;id<DIM;id++){
        ddq_x_qi=get_ddq_x_qi(Bmat,dxk_qi,ii,id);
        res+=dxj_qi[3*Bmat->indB[ii]+id]*ddq_x_qi;
     }
  }

  return res;
}

static void cal_dlnBdqi(real **Akm,rvec *dlndq,real **A1_eta,real **E12,t_Bmat *Bmat,rvec *xx, BKStree_t *tr,
                        rvec *ex, rvec *ey, rvec *ez, real **Aim, t_pbc *pbc, gmx_bool bST, int *sT2)
{
  int i,j,d,dd,jj;

  for(i=0;i<tr->N;i++) clear_rvec(dlndq[i]);

  /* estimate dlnBdqi for the internal coordinates */
  //fprintf(stderr,"A-matrix ouput \n");
  for(i=2;i<tr->N;i++){
    /* get partial x_k / partial qi */
    sub_calAi_mat(A1_eta,E12,i,xx,tr,ex,ey,ez,Akm,pbc,bST,sT2);
    //fprintf(stderr,"%f %f %f %f %f %f\n",Akm[0][0],Akm[0][1],Akm[0][2],Akm[0][3],Akm[0][4],Akm[0][5]);
    /* estimate dlnBdqi */
    for(d=0;d<DIM;d++){
      for(j=2;j<tr->N;j++){
        /* get partial x_ii / partial q_j */
        sub_calAi_mat(A1_eta,E12,j,xx,tr,ex,ey,ez,Aim,pbc,bST,sT2);
        for(dd=0;dd<DIM;dd++){
          /* shuffle coordinates, as Bmat using the coordinate order from the BKS-tree */
          jj=3*j+dd; if(jj==8) jj=3;
          dlndq[i][d]+=sub_cal_dlnBdqi(&Bmat[jj],Akm[d],Aim[dd]);
        }
      }
    }
  }

  /* leave dlnBdqi for the external coordinate as zero. For the rotational part, dlnBdqi may not vanish and thus not taken into account yet */
}

static void out_matrix_transpose(real **A1_eta,int N,int M)
{
  int i,j;

  fprintf(stderr,"Output Matrix after transposed\n");
  for(i=0;i<M;i++){
    fprintf(stderr,"%f",A1_eta[0][i]);
    for(j=1;j<N;j++){
       fprintf(stderr," %f",A1_eta[j][i]);
    }
    fprintf(stderr,"\n");
  }
}

static void do_real2vec(real *fvec, t_forcerec *fr, int i)
{
  int j,k=0,m;

  for(j=0;j<fr->totGrps;j++)
    for(m=0;m<DIM;m++)
      fr->fmat[i].fcomp[j][m]=fvec[k++];
}
static void do_fmat2_read(FILE *fp,t_forcerec *fr,real *fvec)
{
  int i,j;
  size_t buf=0;

  buf=fread(&fr->fmNatoms, sizeof(int), 1, fp);
  buf=fread(&fr->totGrps, sizeof(int), 1, fp);

  for(i=0;i<fr->fmNatoms;i++){
    buf=fread(&fr->fmat[i].iatom, sizeof(int), 1, fp);
    buf=fread(fvec, sizeof(real),DIM*fr->totGrps, fp);
    do_real2vec(fvec,fr,i);
  }
}

static void do_fmat_read(FILE *fp,gmx_bool bFirst,t_forcerec *fr)
{
  int i,j;
  real *fvec;
  size_t buf=0;

  buf=fread(&fr->fmNatoms, sizeof(int), 1, fp);
  buf=fread(&fr->totGrps, sizeof(int), 1, fp);

  if(bFirst){
    snew(fr->fmat,fr->fmNatoms);
    snew(fr->ig2iloc,fr->fmNatoms);
    for(i=0;i<fr->fmNatoms;i++){
      snew(fr->fmat[i].fcomp,fr->totGrps);
      fr->fmat[i].iatom=i;
      fr->ig2iloc[i]=i;
    }
  } 

  snew(fvec,fr->totGrps*DIM);
  for(i=0;i<fr->fmNatoms;i++){
    buf=fread(&fr->fmat[i].iatom, sizeof(int), 1, fp);
    buf=fread(fvec, sizeof(real),DIM*fr->totGrps, fp);
    do_real2vec(fvec,fr,i);
  }
  sfree(fvec);
}

static void updateForce_fmat(t_forcerec *fr,rvec *f, int n, int nsel)
{
  int i,j;

  for(i=0;i<n;i++){ clear_rvec(f[i]); }

  for(i=0;i<fr->fmNatoms;i++){
    j=fr->fmat[i].iatom;
    copy_rvec(fr->fmat[i].fcomp[nsel],f[j]);
  }
}

int gmx_work(int argc, char *argv[])
{
  const char *desc[] = {
    "g_work :: description is comming :: normal work decomposition",
    "No components of forces, e.g., VDW, Elect. are required."
  };
  static gmx_bool bHeavy=FALSE, bBackbone=TRUE, bSubTree=FALSE, bBondList=TRUE;
  static gmx_bool bOrigin=FALSE, bDlnBdqi=FALSE;
  static rvec vec0={0.0,0.0,0.0};
  static int  base=0, ngroups=1, selFcomp=0;
  t_pargs pa[] = {
    { "-origin",    FALSE, etRVEC, {vec0},
      "set the origin of the lab-fixed frame" },
    { "-backbone", FALSE, etBOOL, {&bBackbone},
      "Construct the BKS tree by taking the backbone atoms first" },
    { "-subtree", FALSE, etBOOL, {&bSubTree},
      "Specify the group of atoms over which we construct the BKStree" },
    { "-bondlist", FALSE, etBOOL, {&bBondList},
      "Specify the bond list of the BKStree" },
    { "-heavy", FALSE, etBOOL, {&bHeavy},
      "Construct the BKS tree by taking the heavy atoms first" },
    { "-dlnBdqi", FALSE, etBOOL, {&bDlnBdqi},
      "Perform the estimation of dlnBdqi" },
    { "-ng",       FALSE, etINT, {&ngroups},
      "Number of groups to be read from the index file, select the first one by default" },
    { "-selFcomp",       FALSE, etINT, {&selFcomp},
      "The force component and the ones before it will be output, used in a different way as in version 8" },
    { "-base",       FALSE, etINT, {&base},
      "Set the index of the atom as the base of BKS tree" }
  };
  t_topology top;
  int        ePBC,flags;
  real       *mass,time;
  char       title[STRLEN];
  const char *indexfn;
  t_trxframe fr;
  t_forcerec *fmpar=NULL;
  rvec       *xtop,*xp=NULL;
  matrix     topbox;
  char       **grpname;
  int        *isize0,*isize;
  atom_id    **index0,**index;
  int        i,j,k,n,m,t1;
  gmx_bool   bTop,bPFM;

  real       **inertia,**inv_inert;
  rvec       *xprime=NULL,*prevX=NULL,xcom,rdr,totF,tau;
  real       **A1_eta=NULL,**B1_mat=NULL,**E12_mat=NULL;
  real       **Ak_tmp=NULL,**Ai_tmp=NULL;
  //real       **A1_mat=NULL;
  t_Bmat      *B_mat=NULL;

  rvec       *EvecXX, *EvecYY, *EvecZZ, *Qvec, *intQvec, *extF, *dotQ, *dlnBdqi=NULL;
  BKStree_t  Btr;
  t_pbc      *pbc;
  int        *backbone=NULL;
  int        *subTree=NULL, treesize=0, *subTree2=NULL;
  int        *bondIndex=NULL, totBonds=0;

  t_trxstatus    *status;
  gmx_rmpbc_t     gpbc       = NULL;
  output_env_t    oenv;

//  FILE       *out_name=NULL;
  FILE       *fmfile=NULL;
  FILE       *fm_intQ=NULL;
  FILE       *fm_dotQ=NULL;
  FILE       *fm_geneF=NULL;
  FILE       *fm_dlnBdqi=NULL;
  real       *outFvec=NULL;
  real       *readFvec=NULL;

  t_filenm fnm[] = {
    { efTRX, "-f", NULL, ffREAD },
    { efPFM, "-fm", "forcemat", ffOPTRD },
    { efTPS, NULL, NULL, ffREAD },
    { efNDX, NULL, NULL, ffOPTRD },
  };
#define NFILE asize(fnm)

  if (!parse_common_args(&argc, argv,
                         PCA_CAN_TIME | PCA_TIME_UNIT | PCA_CAN_VIEW,
                         NFILE, fnm, asize(pa), pa, asize(desc), desc, 0, NULL, &oenv))
  {
      return 0;
  }

  bPFM = opt2bSet("-fm",NFILE,fnm);
  bOrigin = opt2parg_bSet("-origin",asize(pa),pa);

  bTop = read_tps_conf(ftp2fn(efTPS,NFILE,fnm),title,&top,&ePBC,
                       &xtop,NULL,topbox,
                       TRUE);
  sfree(xtop);

  indexfn = ftp2fn_null(efNDX,NFILE,fnm);
  snew(grpname,ngroups);
  snew(isize0,ngroups);
  snew(index0,ngroups);
  get_index(&(top.atoms),indexfn,ngroups,isize0,index0,grpname);

  /* extract the index from the index obtained with get_index */
  if(bBackbone){
    snew(backbone,top.atoms.nr);
    flags=0;
    for (i=0; i<ngroups; i++) {
      if(strcmp(grpname[i],"Backbone")==0){
        //size=isize0[i]; /*size: the size of "backbone" group*/
        //snew(backbone,isize0[i]);
        for(j=0;j<isize0[i];j++)
          backbone[index0[i][j]]=1;
        fprintf(stderr,"\nGroup name: %s ,",grpname[i]);
        fprintf(stderr,"Group atom number: %d\n",isize0[i]);
        for(j=0;j<isize0[i];j++)
          fprintf(stderr,"%d ",index0[i][j]);
        fprintf(stderr,"\n");
        flags=1;
        break;
      }
    }
    if(flags==0){
      gmx_fatal(FARGS,"Group name of 'Backbone' is not found in the index file");
    }
  }

  if(bSubTree){
    flags=0;
    for (i=0; i<ngroups; i++) {
      if(strcmp(grpname[i],"SubTree")==0){
        treesize=isize0[i]; /*treesize: the size of "SubTree" group */
        snew(subTree,isize0[i]);
        for(j=0;j<isize0[i];j++)
          subTree[j]=index0[i][j];
        fprintf(stderr,"\nGroup name: %s ,",grpname[i]);
        fprintf(stderr,"Group atom number: %d\n",isize0[i]);
        for(j=0;j<isize0[i];j++)
          fprintf(stderr,"%d ",index0[i][j]);
        fprintf(stderr,"\n");
        flags=1;
        break;
      }
    }
    if(flags==0){
      gmx_fatal(FARGS,"Group name of 'SubTree' is not found in the index file");
    }
  }

  if(bBondList){
    flags=0;
    for (i=0; i<ngroups; i++) {
      if(strcmp(grpname[i],"Bondlist")==0){
        totBonds=isize0[i]; /*treesize: the size of "SubTree" group */
        if(totBonds%2 == 1) gmx_fatal(FARGS,"Not a pair list of bonds: totBonds= %d is not an even number",totBonds);
        snew(bondIndex,isize0[i]);
        for(j=0;j<isize0[i];j++)
          bondIndex[j]=index0[i][j];
        fprintf(stderr,"\nGroup name: %s ,",grpname[i]);
        fprintf(stderr,"Group atom number: %d\n",isize0[i]);
        for(j=0;j<isize0[i];j++)
          fprintf(stderr,"%d ",index0[i][j]);
        fprintf(stderr,"\n");
        flags=1;
        break;
      }
    }
    if(flags==0){
      gmx_fatal(FARGS,"Group name of 'Bondlist' is not found in the index file");
    }
  }

  /*Construct and Initialize the BKS tree*/
  initial_BKStree(&Btr, &top, base,bHeavy,bBackbone,backbone,bSubTree,subTree,treesize,bBondList,bondIndex,totBonds);
  if(bSubTree){
    snew(subTree2,top.atoms.nr);
    for(i=0;i<top.atoms.nr;i++){ subTree2[i]=top.atoms.nr+1; }
    for(i=0;i<treesize;i++)
      subTree2[subTree[i]]=i;
  }

  flags = 0;
  flags = flags | TRX_READ_X | TRX_READ_F | TRX_READ_V;

  read_first_frame(oenv, &status, ftp2fn(efTRX, NFILE, fnm), &fr, flags);
  fprintf(stderr,"\n");

  if (fn2ftp(ftp2fn(efTRX, NFILE, fnm)) == efXTC)
  {
      gmx_fatal(FARGS, "Cannot extract velocities or forces since your input XTC file does not contain them.");
  }

  if(!bPFM && !(fr.bX && fr.bF && fr.bV))
    gmx_fatal(FARGS,"Incomplete trajectory: no coordinate or force or velocity");

  if(bPFM && !fr.bX && !fr.bV)
    gmx_fatal(FARGS,"Incomplete trajectory: no coordinate or velocity");

  if(top.atoms.nr != fr.natoms)
    gmx_fatal(FARGS,"Topology (%d atoms) does not match trajectory (%d atoms)",
                top.atoms.nr,fr.natoms);
  if(ePBC != fr.ePBC)
//    gmx_fatal(FARGS,"ePBCs in .top  (%d) and .trr (%d) are not consistent.",ePBC,fr.ePBC);
    fprintf(stderr,"ePBCs in .top  (%d) and .trr (%d) are not consistent.\n",ePBC,fr.ePBC);

  if(ePBC != epbcNONE){
    snew(pbc,1);
  }else{
    pbc = NULL;
    fprintf(stderr,"No periodic boundary condition\n");
  }

  if(!bPFM){ selFcomp=0; }
  snew(outFvec,Btr.N*DIM);

  snew(EvecXX,Btr.N);
  snew(EvecYY,Btr.N);
  snew(EvecZZ,Btr.N);
  snew(extF,Btr.N);
  snew(Qvec,Btr.N);
  snew(intQvec,Btr.N);
  snew(dotQ,Btr.N);

  snew(inertia,3);
  snew(inv_inert,3);
  for(i=0;i<3;i++){
    snew(inertia[i],3);
    snew(inv_inert[i],3);
  }
  snew(xprime,Btr.N);
  snew(prevX,Btr.N);

  if(bDlnBdqi){
    snew(dlnBdqi,Btr.N);
    snew_mat(B1_mat,6,3*Btr.N,i);
    //snew_mat(A1_mat,6,3*Btr.N,i);
    snew_mat(A1_eta,9,3*Btr.N,i);
    snew_mat(E12_mat,6,3*Btr.N-6,i);
    snew_mat(Ak_tmp,3,3*Btr.N,i);
    snew_mat(Ai_tmp,3,3*Btr.N,i);

    snew(B_mat,3*Btr.N);
    allowMem_Bmat(B_mat,Btr.N);
    fm_dlnBdqi=fopen("dlnBdqi_all_coordinate.xvg","w");
    if(fm_dlnBdqi==NULL){ gmx_fatal(FARGS,"Can't open file dlnBdqi_all_coordinate.xvg\n"); }
  }
  //out_name=fopen("COORDINATE_NAME_LIST.dat","w");
  //if(out_name==NULL){ gmx_fatal(FARGS,"Can't open file COORDINATE_NAME_LIST.dat\n"); }
  //print_coor_name(out_name,&Btr); /* output the name of coordinates */
  //fclose(out_name);

  fm_intQ=fopen("intQ_all_coordinate.xvg","w");
  if(fm_intQ==NULL){ gmx_fatal(FARGS,"Can't open file intQ_all_coordinate.xvg\n"); }
  fm_dotQ=fopen("dotQ_all_coordinate.xvg","w");
  if(fm_dotQ==NULL){ gmx_fatal(FARGS,"Can't open file dotQ_all_coordinate.xvg\n"); }
  fm_geneF=fopen("geneF_all_coordinate.xvg","w");
  if(fm_geneF==NULL){ gmx_fatal(FARGS,"Can't open file geneF_all_coordinate.xvg\n"); }

  time = output_env_conv_time(oenv, fr.time);

  if(bPFM){
    fmfile=fopen(opt2fn("-fm",NFILE,fnm),"r");
    snew(fmpar,1);
    if(fmfile==NULL){
      gmx_fatal(FARGS,"Can't open file %s\n",opt2fn("-fm",NFILE,fnm));
    }else{
      do_fmat_read(fmfile,TRUE,fmpar);
      snew(readFvec,fmpar->totGrps*DIM);
      fmpar->bSubTree=FALSE;
      //output_fmat_ascii(NULL,fmpar,TRUE);
      if(!bSubTree && fmpar->fmNatoms!=Btr.N)
        gmx_fatal(FARGS,"Atom number do not consistent in force component file %d and in structural file %d\n",fmpar->fmNatoms,Btr.N);
      if(bSubTree && (treesize > fmpar->fmNatoms))
        gmx_fatal(FARGS,"The subTree have more atoms (%d) that the atoms in force component file (%d)\n",treesize,fmpar->fmNatoms);
      if(selFcomp>=fmpar->totGrps)
        gmx_fatal(FARGS,"The selected force component (%d) exceeds the total number of components (%d)\n",selFcomp,fmpar->totGrps);
      /* make sure the subtree is a subset in the set of atoms whose force components are saved */
      if(bSubTree){
        for(i=0;i<treesize;i++){
          k=0;
          for(j=0;j<fmpar->fmNatoms;j++){
            if(subTree[i]==fmpar->fmat[j].iatom) k++;
          }
          if(k!=1) gmx_fatal(FARGS,"The force component of subTree atom %d is not saved\n",i+1);
        }
      }
    }
  }

  if (pbc){
    set_pbc(pbc,ePBC,fr.box);
    fprintf(stderr,"set_pbc\n");
    /* remove periodic boundary of the coordinate */
    gpbc = gmx_rmpbc_init(&top.idef, ePBC, fr.natoms);
    gmx_rmpbc_trxfr(gpbc, &fr);
  }

  remove_com_x(&top,&Btr,fr.x,xprime,xcom);
  if(bOrigin){ 
    copy_rvec(vec0,Btr.O0);
  }else{
    copy_rvec(xcom,Btr.O0); /* set xcom as the origin of lab-fixed frame */
    check_frame_origin(fr.x,&Btr,pbc);
  }
  fprintf(stderr,"The origin of lab-fixed frame: %f %f %f\n",Btr.O0[0],Btr.O0[1],Btr.O0[2]);

  calUnitVecE(fr.x,EvecXX,EvecYY,EvecZZ,&Btr,pbc);
  calInternalQ(fr.x,Qvec,&Btr,pbc);

/**********************************************************************/
/*** NOTE: extF,xprime saves only the values for the BKS-tree      ***/
/********************************************************************/

for(m=0;m<=selFcomp;m++){
  /* remove the force on each coordinate, which contribute to the generalized force on the six external coordinates */
  if(bPFM){ updateForce_fmat(fmpar,fr.f,fr.natoms,m); }
  cal_totF(&Btr,fr.f,totF);
  cal_torque(tau,xprime,fr.f,&Btr);
  cal_extF(&top,&Btr,xprime,inertia,inv_inert,extF,totF,tau);

  /* estimate the generalized force for the BKS-tree coordinates, via the way of BKS-tree */
  calGeneF_BKS(fr.x,fr.f,extF,&Btr,EvecXX,EvecYY,EvecZZ,intQvec,pbc,bSubTree,subTree2);
  cal_geneF_rdr(tau,inv_inert); /* here tau saves the generalized force on rdr */
  update_geneF(intQvec,totF,tau);    /* update geneF on external coordinate */
  writeBin2_nvec(fm_geneF,intQvec,Btr.N,DIM,outFvec); 
  /*if(bPFM){
    print_data_geneF_FM(intQvec,&Btr,TRUE,m);
  }else{
    print_data_geneF(intQvec,&Btr,TRUE);
  }*/
}
/**************************************/
/*** estimate d ln|B| / d q_i      ***/ 
/************************************/
  if(bDlnBdqi){
    /* estimate B1 matrix, A1 matrix, D1 matrix (D1 is temperarily saved in A1_eta matrix) and then E12 matrix */
    calB1_matrix(&top,&Btr,B1_mat,xprime);
    //out_matrix_transpose(B1_mat,6,3*Btr.N);
    //calB1_matrix_Euler(&top,&Btr,B1_mat,xprime,inv_inert);
    //calA1_matrix(&Btr,A1_mat,xprime,inv_inert);
    calD1_mat(fr.x,B1_mat,A1_eta,&Btr,EvecXX,EvecYY,EvecZZ,pbc,bSubTree,subTree2);
    //out_matrix_transpose(A1_eta,6,3*Btr.N);
    calE12_mat(A1_eta,E12_mat,3*Btr.N);
    //out_matrix_transpose(E12_mat,6,3*Btr.N-6);

    /* calculate A1_eta matrix */
    calA1_eta(fr.x,&Btr,EvecXX,EvecYY,EvecZZ,A1_eta,pbc,bSubTree,subTree2);
    /* estimate the first order and the second order B-matrix and save it in a sparse matrix structure */
    calB_matrix(fr.x,&Btr,B_mat,pbc); 
    /* d ln|B| / d q_i */
    cal_dlnBdqi(Ak_tmp,dlnBdqi,A1_eta,E12_mat,B_mat,fr.x,&Btr,EvecXX,EvecYY,EvecZZ,Ai_tmp,pbc,bSubTree,subTree2);
  }
/**************************************/
/*** end of d ln|B| / d q_i code   ***/
/************************************/

  cal_geneVel(fr.x,&top,&Btr,dotQ,pbc,fr.v,xprime);
  /* update intQ on external coordinate */
  update_intQ(Qvec,xcom,rdr);
  /* output stuff */
  writeBin2_nvec(fm_intQ,Qvec,Btr.N,DIM,outFvec);
  writeBin2_nvec(fm_dotQ,dotQ,Btr.N,DIM,outFvec);
  if(bDlnBdqi) writeBin2_nvec(fm_dlnBdqi,dlnBdqi,Btr.N,DIM,outFvec);
  //print_data_intQ(Qvec,&Btr,TRUE);
  //print_data_dotQ(dotQ,&Btr,TRUE);
  //if(bDlnBdqi) print_data_dlnBdqi(dlnBdqi,&Btr,TRUE);

  copy_prevX(fr.x,prevX,&Btr);

  while(read_next_frame(oenv, status, &fr)) {
    if(bPFM){ 
      do_fmat2_read(fmfile,fmpar,readFvec); 
      //output_fmat_ascii(NULL,fmpar,FALSE);
    }

    if (pbc){
      set_pbc(pbc,ePBC,fr.box);
      gmx_rmpbc_trxfr(gpbc, &fr);
    }

    remove_com_x(&top,&Btr,fr.x,xprime,xcom);
    if(!bOrigin){ 
      copy_rvec(xcom,Btr.O0); /* set xcom as the origin of lab-fixed frame */
      check_frame_origin(fr.x,&Btr,pbc);
    }
    //fprintf(stderr,"The origin of lab-fixed frame: %f %f %f\n",Btr.O0[0],Btr.O0[1],Btr.O0[2]);

    calUnitVecE(fr.x,EvecXX,EvecYY,EvecZZ,&Btr,pbc);
    calInternalQ(fr.x,Qvec,&Btr,pbc);

  for(m=0;m<=selFcomp;m++){
    if(bPFM){ updateForce_fmat(fmpar,fr.f,fr.natoms,m); }
    cal_totF(&Btr,fr.f,totF);
    cal_torque(tau,xprime,fr.f,&Btr);
    cal_extF(&top,&Btr,xprime,inertia,inv_inert,extF,totF,tau);
    calGeneF_BKS(fr.x,fr.f,extF,&Btr,EvecXX,EvecYY,EvecZZ,intQvec,pbc,bSubTree,subTree2);
    cal_geneF_rdr(tau,inv_inert); /* here tau saves the generalized force on rdr */
    update_geneF(intQvec,totF,tau);
    writeBin2_nvec(fm_geneF,intQvec,Btr.N,DIM,outFvec);
    /*if(bPFM){
      print_data_geneF_FM(intQvec,&Btr,FALSE,m);
    }else{
      print_data_geneF(intQvec,&Btr,FALSE);
    }*/
  }
/**************************************/
/*** estimate d ln|B| / d q_i      ***/
/************************************/
    if(bDlnBdqi){
      /* estimate B1 matrix, A1 matrix, D1 matrix (D1 is temperarily saved in A1_eta matrix) and then E12 matrix */
      calB1_matrix(&top,&Btr,B1_mat,xprime);
      //calB1_matrix_Euler(&top,&Btr,B1_mat,xprime,inv_inert);
      //calA1_matrix(&Btr,A1_mat,xprime,inv_inert);
      calD1_mat(fr.x,B1_mat,A1_eta,&Btr,EvecXX,EvecYY,EvecZZ,pbc,bSubTree,subTree2);
      calE12_mat(A1_eta,E12_mat,3*Btr.N);

      /* calculate A1_eta matrix */
      calA1_eta(fr.x,&Btr,EvecXX,EvecYY,EvecZZ,A1_eta,pbc,bSubTree,subTree2);
      /* estimate the first order and the second order B-matrix and save it in a sparse matrix structure */
      calB_matrix(fr.x,&Btr,B_mat,pbc);
      /* d ln|B| / d q_i */
      cal_dlnBdqi(Ak_tmp,dlnBdqi,A1_eta,E12_mat,B_mat,fr.x,&Btr,EvecXX,EvecYY,EvecZZ,Ai_tmp,pbc,bSubTree,subTree2);
    }
/**************************************/
/*** end of d ln|B| / d q_i code   ***/
/************************************/

    cal_geneVel(fr.x,&top,&Btr,dotQ,pbc,fr.v,xprime);

    /*calculate the external coordinate*/
    update_ext_intQ(Qvec,xcom,rdr,fr.x,prevX,xprime,&Btr,&top);

    time = output_env_conv_time(oenv, fr.time);

    writeBin2_nvec(fm_intQ,Qvec,Btr.N,DIM,outFvec);
    writeBin2_nvec(fm_dotQ,dotQ,Btr.N,DIM,outFvec);
    if(bDlnBdqi) writeBin2_nvec(fm_dlnBdqi,dlnBdqi,Btr.N,DIM,outFvec);
    //print_data_intQ(Qvec,&Btr,FALSE);
    //print_data_dotQ(dotQ,&Btr,FALSE);
    //if(bDlnBdqi) print_data_dlnBdqi(dlnBdqi,&Btr,FALSE);

    copy_prevX(fr.x,prevX,&Btr);
  }

  if (gpbc != NULL)
  {
      gmx_rmpbc_done(gpbc);
  }

  /* clean up a bit */
  close_trj(status);

  for(i=0;i<3;i++){
    sfree(inertia[i]);
    sfree(inv_inert[i]);
  }
  sfree(inertia);
  sfree(inv_inert);

  if(bPFM){
    fclose(fmfile);
    fmat_destory(fmpar);
    sfree(fmpar);
    sfree(readFvec);
  }

  fclose(fm_intQ);
  fclose(fm_dotQ);
  fclose(fm_geneF); 

  sfree(outFvec);
  sfree(extF);
  sfree(xprime);
  sfree(prevX);
  sfree(EvecXX);
  sfree(EvecYY);
  sfree(EvecZZ);
  sfree(Qvec);
  sfree(intQvec);
  sfree(dotQ);
  if(bSubTree){ sfree(subTree); sfree(subTree2); }
  if(bBackbone){ sfree(backbone); }
  if(bBondList){ sfree(bondIndex); }

  if(bDlnBdqi){
    fclose(fm_dlnBdqi);
    sfree(dlnBdqi);
    sfree_mat(B1_mat,6,i);
    //sfree_mat(A1_mat,6,i);
    sfree_mat(A1_eta,9,i);
    sfree_mat(E12_mat,6,i);
    sfree_mat(Ak_tmp,3,i);
    sfree_mat(Ai_tmp,3,i);

    freeMem_Bmat(B_mat,Btr.N);
    sfree(B_mat);
  }
  /* view it */
  view_all(oenv, NFILE, fnm);

  return 0;
}

