


/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */


#include <cstdio>
#include <cstring>
//#include "fix_qeq_reax.h"
#include "fix_nve.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "update.h"
#include "respa.h"
#include "error.h"
#include "memory.h"
#include <cmath>
using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixNVE::FixNVE(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (strcmp(style,"nve/sphere") != 0 && narg < 11)
    error->all(FLERR,"Illegal fix nve command");
  int nlocal = atom->nlocal;
  t_s_flag     =force->numeric(FLERR,arg[3]); //t_s_flag=1:solve for t and s t_s_flag=0 q solve for q (s only)
  iEL_Scf_flag =force->numeric(FLERR,arg[4]);
  thermo_flag  =force->numeric(FLERR,arg[5]);
  tautemp_aux  =force->numeric(FLERR,arg[6]); //100000.0; //t_S //1000000    //100000
  kelvin_aux_t =force->numeric(FLERR,arg[7]);  //0.000001; //t_S//0.00000001
  kelvin_aux_s =force->numeric(FLERR,arg[8]); //0.00000001; //t_s//0.0000001  //0.0000001
  Omega_t        =force->numeric(FLERR,arg[9]);
  Omega_s        =force->numeric(FLERR,arg[10]);
    
  if(comm->me == 0)
    printf("\niEL_Scf params:\nt_s_flag= %i iEL_Scf_flag= %i  thermo_flag= %i tau= %.4f T0_t= %.10f T0_s= %.10f Omega_t= %.5f Omega_s=%.5f\n\n",t_s_flag,iEL_Scf_flag,thermo_flag,tautemp_aux,kelvin_aux_t,kelvin_aux_s,Omega_t,Omega_s);
  
  for (int i=0;i < 5; i++)
    {
      tgnhaux[i] = 0.0;
      tvnhaux[i] = 0.0;
      tnhaux[i] = 0.0;
      sgnhaux[i] = 0.0;
      svnhaux[i] = 0.0;
      snhaux[i] = 0.0;
    }

  dynamic_group_allow = 1;
  time_integrate = 1;
}

/* ---------------------------------------------------------------------- */

int FixNVE::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNVE::init()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;

  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  double qterm_aux_t;
  double qterm_aux_s;
  
  double *vt_EL_Scf;
  double *at_EL_Scf;
  double *vs_EL_Scf;
  double *as_EL_Scf;

  get_names("vt_EL_Scf",vt_EL_Scf); 
  get_names("at_EL_Scf",at_EL_Scf); 
  get_names("vs_EL_Scf",vs_EL_Scf); 
  get_names("as_EL_Scf",as_EL_Scf); 
    
  atom->vChi_eq_iEL_Scf = 0.0;
  atom->aChi_eq_iEL_Scf =0.0; // add by Itai for the iEL_ScF

  for (int i = 0; i < nlocal; i++){
    if (mask[i] & groupbit) {
      vs_EL_Scf[i] = 0.0;
      as_EL_Scf[i] = 0.0; // add by Itai for the iEL_ScF
      vt_EL_Scf[i] = 0.0;
      at_EL_Scf[i] = 0.0; // add by Itai for the iEL_ScF
    }
  }
  
  qterm_aux_t = kelvin_aux_t * tautemp_aux * tautemp_aux;
  qterm_aux_s = kelvin_aux_s * tautemp_aux * tautemp_aux;

  for (int i=0;i < 5; i++)
    {
      if (tgnhaux[i]==0.0)tgnhaux[i] = qterm_aux_t;
      tvnhaux[i] = 0.0;
      if (tnhaux[i]==0.0)tnhaux[i]  = qterm_aux_t;
      if (sgnhaux[i]==0.0)sgnhaux[i] = qterm_aux_s;
      svnhaux[i] = 0.0;
      if (snhaux[i]==0.0)snhaux[i]  = qterm_aux_s;
    }

  if (strstr(update->integrate_style,"respa"))
    step_respa = ((Respa *) update->integrate)->step;
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixNVE::initial_integrate(int vflag)
{
  double dtfm;

  // update v and x of atoms in group
  double *vt_EL_Scf;
  double *at_EL_Scf;
  double *t_EL_Scf;
  double *vs_EL_Scf;
  double *as_EL_Scf;
  double *s_EL_Scf;
  double *s;
  double *t;
  
  get_names("t_EL_Scf",t_EL_Scf); 
  get_names("vt_EL_Scf",vt_EL_Scf); 
  get_names("at_EL_Scf",at_EL_Scf); 
  get_names("s_EL_Scf",s_EL_Scf); 
  get_names("vs_EL_Scf",vs_EL_Scf); 
  get_names("as_EL_Scf",as_EL_Scf); 
  get_names("s",s); 
  get_names("t",t); 
  
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  
  double term=2.0/(dtv*dtv);
  double dt=dtv;
  double dt_2=0.5*dtv;
  
  //printf("nlocal=%.i \n",nlocal);
  if (rmass) {
    if(iEL_Scf_flag){
      for (int i = 0; i < nlocal; i++)
	if (mask[i] & groupbit) {
	  dtfm = dtf / rmass[i];
	  v[i][0] += dtfm * f[i][0];
	  v[i][1] += dtfm * f[i][1];
	  v[i][2] += dtfm * f[i][2];
	  x[i][0] += dtv * v[i][0];
	  x[i][1] += dtv * v[i][1];
	  x[i][2] += dtv * v[i][2];
	  
	  vt_EL_Scf[i] += at_EL_Scf[i]*dt_2*Omega_t; // add by Itai for the iEL_ScF
	  t_EL_Scf[i]  += vt_EL_Scf[i]*dtv*Omega_t;     // add by Itai for the iEL_ScF
	  vs_EL_Scf[i] += as_EL_Scf[i]*dt_2*Omega_s; // add by Itai for the iEL_ScF
	  s_EL_Scf[i]  += vs_EL_Scf[i]*dtv*Omega_s;     // add by Itai for the iEL_ScF
	}
    }
    else
      {
	for (int i = 0; i < nlocal; i++)
	  if (mask[i] & groupbit) {
	    dtfm = dtf / rmass[i];
	    v[i][0] += dtfm * f[i][0];
	    v[i][1] += dtfm * f[i][1];
	    v[i][2] += dtfm * f[i][2];
	    x[i][0] += dtv * v[i][0];
	    x[i][1] += dtv * v[i][1];
	    x[i][2] += dtv * v[i][2];
	  }
      }
  }
  else {
    if(iEL_Scf_flag){
      for (int i = 0; i < nlocal; i++)
	if (mask[i] & groupbit) {
	  dtfm = dtf / mass[type[i]];
	  
	  v[i][0] += dtfm * f[i][0];
	  v[i][1] += dtfm * f[i][1];
	  v[i][2] += dtfm * f[i][2];
	  x[i][0] += dtv * v[i][0];
	  x[i][1] += dtv * v[i][1];
	  x[i][2] += dtv * v[i][2];
	  
	  vt_EL_Scf[i] += at_EL_Scf[i]*dt_2*Omega_t; // add by Itai for the iEL_ScF
	  t_EL_Scf[i]  += vt_EL_Scf[i]*dtv*Omega_t;     // add by Itai for the iEL_ScF
	  vs_EL_Scf[i] += as_EL_Scf[i]*dt_2*Omega_s; // add by Itai for the iEL_ScF
	  s_EL_Scf[i]  += vs_EL_Scf[i]*dtv*Omega_s;     // add by Itai for the iEL_ScF
	}
    }
    else
      {
	for (int i = 0; i < nlocal; i++)
	  if (mask[i] & groupbit) {
	    dtfm = dtf / mass[type[i]];
	    
	    v[i][0] += dtfm * f[i][0];
	    v[i][1] += dtfm * f[i][1];
	    v[i][2] += dtfm * f[i][2];
	    x[i][0] += dtv * v[i][0];
	    x[i][1] += dtv * v[i][1];
	    x[i][2] += dtv * v[i][2];
	  }
      }
  }
  
  if(t_s_flag==0 and iEL_Scf_flag){
    atom->vChi_eq_iEL_Scf = atom->vChi_eq_iEL_Scf + atom->aChi_eq_iEL_Scf*dt_2; // add by Itai for the iEL_ScF
    atom->Chi_eq_iEL_Scf  = atom->Chi_eq_iEL_Scf + atom->vChi_eq_iEL_Scf*dt;     // add by Itai for the iEL_ScF
  }
    
  if(thermo_flag==1 and iEL_Scf_flag)
    Nose_Hoover();
  else if(thermo_flag==2 and iEL_Scf_flag)
    Berendersen();
    //********Berendersen scaling for the iEL_ScF velocities******
  //comm->forward_comm_fix(this);

  //printf("timestep dtf= %.16f timestep dtv = %.16f\n",dtf,dtv);
 
  //************************************
  
  
  
  //**********************************************************
}

/* ---------------------------------------------------------------------- */

void FixNVE::final_integrate()
{
  double dtfm;

  // update v of atoms in group

  double *vt_EL_Scf;
  double *at_EL_Scf;
  double *t_EL_Scf;
  double *vs_EL_Scf;
  double *as_EL_Scf;
  double *s_EL_Scf;
  double *s;
  double *t;
  
  get_names("t_EL_Scf",t_EL_Scf); 
  get_names("vt_EL_Scf",vt_EL_Scf); 
  get_names("at_EL_Scf",at_EL_Scf); 
  get_names("s_EL_Scf",s_EL_Scf); 
  get_names("vs_EL_Scf",vs_EL_Scf); 
  get_names("as_EL_Scf",as_EL_Scf); 
  get_names("s",s); 
  get_names("t",t); 

  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  //double term=2.0/(dtf*dtf);
  //double dt=dtf;
  //double dt_2=0.5*dtf;
  double term_t,term_s;
  double term=2.0/(dtv*dtv);
  double dt=dtv;
  double dt_2=0.5*dtv;
  //term_t=Omega_t*term;
  //term_s=Omega_s*term;
  term_t=term;
  term_s=term;
  tagint *tag = atom->tag;
  int n_atoms =atom->natoms;
  if (rmass) {
    if(iEL_Scf_flag){
      for (int i = 0; i < nlocal; i++){
	if (mask[i] & groupbit) {
	  dtfm = dtf / rmass[i];
	  
	  v[i][0] += dtfm * f[i][0];
	  v[i][1] += dtfm * f[i][1];
	  v[i][2] += dtfm * f[i][2];
	  
	  //at_EL_Scf[i] = term_t * (t[i]-t_EL_Scf[i]);          // add by Itai for the iEL_ScF
	  at_EL_Scf[i] = term_t * (t_EL_Scf[i]-t[i]);          // add by Itai for the iEL_ScF
	  vt_EL_Scf[i] = vt_EL_Scf[i] + at_EL_Scf[i]*dt_2*Omega_t;   // add by Itai for the iEL_ScF
	  //as_EL_Scf[i] = term_s * (s[i]-s_EL_Scf[i]);          // add by Itai for the iEL_ScF
	  as_EL_Scf[i] = term_s * (s_EL_Scf[i]-s[i]);          // add by Itai for the iEL_ScF
	  vs_EL_Scf[i] = vs_EL_Scf[i] + as_EL_Scf[i]*dt_2*Omega_s;   // add by Itai for the iEL_ScF
	  //printf("at_EL_Scf=%.16f as_EL_Scf=%.16f vt_EL_Scf=%.16f vs_EL_Scf=%.16f\n",at_EL_Scf[i],as_EL_Scf[i],vt_EL_Scf[i],vs_EL_Scf[i]);
	}
      }
    }
    else 
      {
	for (int i = 0; i < nlocal; i++)
	  if (mask[i] & groupbit) {
	    dtfm = dtf / rmass[i];
	    
	    v[i][0] += dtfm * f[i][0];
	    v[i][1] += dtfm * f[i][1];
	    v[i][2] += dtfm * f[i][2];
	    //printf("at_EL_Scf=%.16f as_EL_Scf=%.16f vt_EL_Scf=%.16f vs_EL_Scf=%.16f\n",at_EL_Scf[i],as_EL_Scf[i],vt_EL_Scf[i],vs_EL_Scf[i]);
	  }
      }
  }
  else {
    if(iEL_Scf_flag){
      for (int i = 0; i < nlocal; i++)
	if (mask[i] & groupbit) {
	  dtfm = dtf / mass[type[i]];
	  
	  v[i][0] += dtfm * f[i][0];
	  v[i][1] += dtfm * f[i][1];
	  v[i][2] += dtfm * f[i][2];
	  
	  at_EL_Scf[i] = term_t * (t_EL_Scf[i]-t[i]);          // add by Itai for the iEL_ScF
	  vt_EL_Scf[i] = vt_EL_Scf[i] + at_EL_Scf[i]*dt_2*Omega_t;   // add by Itai for the iEL_ScF
	  as_EL_Scf[i] = term_s * (s_EL_Scf[i]-s[i]);          // add by Itai for the iEL_ScF
	  vs_EL_Scf[i] = vs_EL_Scf[i] + as_EL_Scf[i]*dt_2*Omega_s;   // add by Itai for the iEL_Sc

	}
    }
    else{
      for (int i = 0; i < nlocal; i++)
	if (mask[i] & groupbit) {
	  dtfm = dtf / mass[type[i]];
	  
	  v[i][0] += dtfm * f[i][0];
	  v[i][1] += dtfm * f[i][1];
	  v[i][2] += dtfm * f[i][2];
	  
	  //printf("t_EL_Scf=%.16f s_EL_Scf=%.16f t=%.16f s=%.16f\n",t_EL_Scf[i],s_EL_Scf[i],t[i],s[i]);
	}
    }
  }
  
  
  if(thermo_flag==1 and iEL_Scf_flag)
    Nose_Hoover();
  
  
}

/* ---------------------------------------------------------------------- */

void FixNVE::initial_integrate_respa(int vflag, int ilevel, int iloop)
{
  dtv = step_respa[ilevel];
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;

  // innermost level - NVE update of v and x
  // all other levels - NVE update of v

  if (ilevel == 0) initial_integrate(vflag);
  else final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVE::final_integrate_respa(int ilevel, int iloop)
{
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVE::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}

void FixNVE::kinaux (double &temp_aux_t,double &temp_aux_s)
{

  double *vt_EL_Scf;
  double *vs_EL_Scf;
  get_names("vt_EL_Scf",vt_EL_Scf); 
  get_names("vs_EL_Scf",vs_EL_Scf); 

  int i;
  double term;
  double eksum_aux;
  double ekaux_t=0.0;
  double ekaux_s=0.0;
  int nfree_aux =atom->natoms;
  int nlocal = atom->nlocal;
  //zero out the temperature and kinetic energy components
  double ekaux_t_mpi = 0.0;
  double ekaux_s_mpi = 0.0;

  temp_aux_t = 0.0;
  temp_aux_s = 0.0;
  
  //get the kinetic energy tensor for auxiliary variables
  term = 0.5;
  
  if(t_s_flag==1)
    {
      for (i = 0; i < nlocal ;i++){
	ekaux_t = ekaux_t + term*vt_EL_Scf[i]*vt_EL_Scf[i];
	ekaux_s = ekaux_s + term*vs_EL_Scf[i]*vs_EL_Scf[i];
      }
      MPI_Allreduce( &ekaux_t, &ekaux_t_mpi, 1, MPI_DOUBLE, MPI_SUM, world);
      temp_aux_t = 2.0 * ekaux_t_mpi / nfree_aux;
    }
  else 
    {
      for (i = 0; i < nlocal ;i++){
	ekaux_s = ekaux_s + term*vs_EL_Scf[i]*vs_EL_Scf[i];
      }
      ekaux_t = term*atom->vChi_eq_iEL_Scf*atom->vChi_eq_iEL_Scf;
      temp_aux_t = 2.0 * ekaux_t ;
    }
  
  MPI_Allreduce( &ekaux_s, &ekaux_s_mpi, 1, MPI_DOUBLE, MPI_SUM, world);
  
  // find the total kinetic energy and auxiliary temperatures
  
  //eksum_aux = ekaux;
  
  //if (nfree_aux =! 0){
  temp_aux_s = 2.0 * ekaux_s_mpi / (nfree_aux);
  
}

void FixNVE::get_names(char *c,double *&ptr)
{
  int index,flag;
  index = atom->find_custom(c,flag);
  
  if(index!=-1) ptr = atom->dvector[index];
  else error->all(FLERR,"fix iEL-Scf requires fix property/atom ?? command");
}

int FixNVE::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int i,j,m;
  
  double *vt_EL_Scf;
  double *at_EL_Scf;
  double *t_EL_Scf;
  double *vs_EL_Scf;
  double *as_EL_Scf;
  double *s_EL_Scf;
  
  get_names("t_EL_Scf",t_EL_Scf); 
  get_names("vt_EL_Scf",vt_EL_Scf); 
  get_names("at_EL_Scf",at_EL_Scf); 
  get_names("s_EL_Scf",s_EL_Scf); 
  get_names("vs_EL_Scf",vs_EL_Scf); 
  get_names("as_EL_Scf",as_EL_Scf); 
  
  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = t_EL_Scf[j];
    buf[m++] = vt_EL_Scf[j];
    buf[m++] = at_EL_Scf[j];
    buf[m++] = s_EL_Scf[j];
    buf[m++] = vs_EL_Scf[j];
    buf[m++] = as_EL_Scf[j];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixNVE::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  double *vt_EL_Scf;
  double *at_EL_Scf;
  double *t_EL_Scf;
  double *vs_EL_Scf;
  double *as_EL_Scf;
  double *s_EL_Scf;

  get_names("t_EL_Scf",t_EL_Scf); 
  get_names("vt_EL_Scf",vt_EL_Scf); 
  get_names("at_EL_Scf",at_EL_Scf); 
  get_names("s_EL_Scf",s_EL_Scf); 
  get_names("vs_EL_Scf",vs_EL_Scf); 
  get_names("as_EL_Scf",as_EL_Scf); 

  m = 0;
  last = first + n;
  for (i = first; i < last; i++){
    t_EL_Scf[i] = buf[m++];
    vt_EL_Scf[i] = buf[m++];
    at_EL_Scf[i] = buf[m++];
    s_EL_Scf[i] = buf[m++];
    vs_EL_Scf[i] = buf[m++];
    as_EL_Scf[i] = buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

void FixNVE::Berendersen()
{
  
  int nlocal = atom->nlocal;
  double *vt_EL_Scf;
  double *vs_EL_Scf;
  
  get_names("vt_EL_Scf",vt_EL_Scf); 
  get_names("vs_EL_Scf",vs_EL_Scf);   

  double scale_t = 1.0;
  double scale_s = 1.0;
  double temp_aux_t = 0.0;
  double temp_aux_s = 0.0;
  kinaux (temp_aux_t, temp_aux_s);
  
  if(temp_aux_s != 0)
    scale_s = sqrt(1.0 + (dtv/tautemp_aux)*(kelvin_aux_s/temp_aux_s-1.0));
  
  if(temp_aux_t != 0)
    scale_t = sqrt(1.0 + (dtv/tautemp_aux)*(kelvin_aux_t/temp_aux_t-1.0));
  
  if(t_s_flag){
    for (int i = 0; i < nlocal; i++){
      vt_EL_Scf[i] = scale_t * vt_EL_Scf[i];
      vs_EL_Scf[i] = scale_s * vs_EL_Scf[i];
      if(abs(vt_EL_Scf[i]) > 0.1)vt_EL_Scf[i]*=0.1;
      if(abs(vs_EL_Scf[i]) > 0.1)vs_EL_Scf[i]*=0.1;
    }
  }
  else{
    for (int i = 0; i < nlocal; i++){
      vs_EL_Scf[i] = scale_s * vs_EL_Scf[i];
      if(abs(vs_EL_Scf[i]) > 1.0)vs_EL_Scf[i]*=0.1;
    }
    atom->vChi_eq_iEL_Scf=atom->vChi_eq_iEL_Scf*scale_t;
  }

}
  
void FixNVE::Nose_Hoover()
{
  
  int nlocal = atom->nlocal;
  double *vt_EL_Scf;
  double *vs_EL_Scf;
  
  get_names("vt_EL_Scf",vt_EL_Scf); 
  get_names("vs_EL_Scf",vs_EL_Scf); 
  
  double scale_t = 1.0;
  double scale_s = 1.0;
  double temp_aux_t = 0.0;
  double temp_aux_s = 0.0;
  kinaux (temp_aux_t, temp_aux_s);
  
  int nc = 5;
  int ns = 3;
  double dtc = dtv / nc;
  double w[3];
  w[0] = 1.0 / (2.0-pow(2.0,(1.0/3.0)));
  w[1] = 1.0 - 2.0*w[0];
  w[2] = w[0];
   
  double expterm;
  double dts,dt2,dt4,dt8;
  int nfree_aux =atom->natoms;
  
  for (int i = 0;i < nc; i++){
    for (int j = 0; j < ns; j++){
      dts = w[j] * dtc;
      dt2 = 0.5 * dts;
      dt4 = 0.25 * dts;
      dt8 = 0.125 * dts;
      
      if(t_s_flag)
	{
	  // t aux calculation
	  tgnhaux[4]=(tnhaux[3]*tvnhaux[3]*tvnhaux[3]-kelvin_aux_t)/tnhaux[4];
	  //printf("tgnhaux[4]= %.16f tnhaux[3]= %.16f tvnhaux[3]=%.16f tnhaux[4]=%.16f kelvin_aux=%.16f\n",tgnhaux[4],tnhaux[3],tvnhaux[3],tnhaux[4],kelvin_aux);
	  tvnhaux[4]=tvnhaux[4]+tgnhaux[4]*dt4;
	  tgnhaux[3]=(tnhaux[2]*tvnhaux[2]*tvnhaux[2]-kelvin_aux_t)/tnhaux[3];
	  expterm=exp(-tvnhaux[4]*dt8);
	  tvnhaux[3]=expterm*(tvnhaux[3]*expterm+tgnhaux[3]*dt4);
	  tgnhaux[2]=(tnhaux[1]*tvnhaux[1]*tvnhaux[1]-kelvin_aux_t)/tnhaux[2];
	  expterm=exp(-tvnhaux[3]*dt8);
	  tvnhaux[2]=expterm*(tvnhaux[2]*expterm+tgnhaux[2]*dt4);
	  tgnhaux[1]=((nfree_aux)*temp_aux_t-(nfree_aux)*kelvin_aux_t)/tnhaux[1];
	  expterm=exp(-tvnhaux[2]*dt8);
	  tvnhaux[1]=expterm*(tvnhaux[1]*expterm+tgnhaux[1]*dt4);
	  scale_t=scale_t*exp(-tvnhaux[1]*dt2);
	  
	  temp_aux_t=temp_aux_t*exp(-tvnhaux[1]*dt2)*exp(-tvnhaux[1]*dt2);
	  tgnhaux[1]=((nfree_aux)*temp_aux_t-(nfree_aux)*kelvin_aux_t)/tnhaux[1];
	  expterm=exp(-tvnhaux[2]*dt8);
	  tvnhaux[1]=expterm*(tvnhaux[1]*expterm+tgnhaux[1]*dt4);
	  tgnhaux[2]=(tnhaux[1]*tvnhaux[1]*tvnhaux[1]-kelvin_aux_t)/tnhaux[2];
	  expterm=exp(-tvnhaux[3]*dt8);
	  tvnhaux[2]=expterm*(tvnhaux[2]*expterm+tgnhaux[2]*dt4);
	  tgnhaux[3]=(tnhaux[2]*tvnhaux[2]*tvnhaux[2]-kelvin_aux_t)/tnhaux[3];
	  expterm=exp(-tvnhaux[4]*dt8);
	  tvnhaux[3]=expterm*(tvnhaux[3]*expterm+tgnhaux[3]*dt4);
	  tgnhaux[4]=(tnhaux[3]*tvnhaux[3]*tvnhaux[3]-kelvin_aux_t)/tnhaux[4];
	  tvnhaux[4]=tvnhaux[4]+tgnhaux[4]*dt4;
	} 
      //printf("scale_t= %.16f exp(tvn)= %.16f dt2=%.16f tvnhaux[1]=%.16f \n",scale_t,exp(-tvnhaux[1]*dt2),dt2,tvnhaux[1]);
      // s aux calculation
      sgnhaux[4]=(snhaux[3]*svnhaux[3]*svnhaux[3]-kelvin_aux_s)/snhaux[4];
      svnhaux[4]=svnhaux[4]+sgnhaux[4]*dt4;
      sgnhaux[3]=(snhaux[2]*svnhaux[2]*svnhaux[2]-kelvin_aux_s)/snhaux[3];
      expterm=exp(-svnhaux[4]*dt8);
      svnhaux[3]=expterm*(svnhaux[3]*expterm+sgnhaux[3]*dt4);
      sgnhaux[2]=(snhaux[1]*svnhaux[1]*svnhaux[1]-kelvin_aux_s)/snhaux[2];
      expterm=exp(-svnhaux[3]*dt8);
      svnhaux[2]=expterm*(svnhaux[2]*expterm+sgnhaux[2]*dt4);
      sgnhaux[1]=((nfree_aux)*temp_aux_s-(nfree_aux)*kelvin_aux_s)/snhaux[1];
      expterm=exp(-svnhaux[2]*dt8);
      svnhaux[1]=expterm*(svnhaux[1]*expterm+sgnhaux[1]*dt4);
      scale_s=scale_s*exp(-svnhaux[1]*dt2);
      temp_aux_s=temp_aux_s*exp(-svnhaux[1]*dt2)*exp(-svnhaux[1]*dt2);
      sgnhaux[1]=((nfree_aux)*temp_aux_s-(nfree_aux)*kelvin_aux_s)/snhaux[1];
      expterm=exp(-svnhaux[2]*dt8);
      svnhaux[1]=expterm*(svnhaux[1]*expterm+sgnhaux[1]*dt4);
      sgnhaux[2]=(snhaux[1]*svnhaux[1]*svnhaux[1]-kelvin_aux_s)/snhaux[2];
      expterm=exp(-svnhaux[3]*dt8);
      svnhaux[2]=expterm*(svnhaux[2]*expterm+sgnhaux[2]*dt4);
      sgnhaux[3]=(snhaux[2]*svnhaux[2]*svnhaux[2]-kelvin_aux_s)/snhaux[3];
      expterm=exp(-svnhaux[4]*dt8);
      svnhaux[3]=expterm*(svnhaux[3]*expterm+sgnhaux[3]*dt4);
      sgnhaux[4]=(snhaux[3]*svnhaux[3]*svnhaux[3]-kelvin_aux_s)/snhaux[4];
      svnhaux[4]=svnhaux[4]+sgnhaux[4]*dt4;
      
    }
  }
  //printf("scale_t= %.16f scale_s= %.16f \n",scale_t,scale_s);
  for (int i = 0; i < nlocal; i++){
    if(t_s_flag)
      vt_EL_Scf[i]=vt_EL_Scf[i]*scale_t;
    vs_EL_Scf[i]=vs_EL_Scf[i]*scale_s;
  }
  if(t_s_flag==0)
    atom->vChi_eq_iEL_Scf=atom->vChi_eq_iEL_Scf*scale_s;

}
