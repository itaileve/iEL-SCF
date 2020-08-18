/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(nve,FixNVE)

#else

#ifndef LMP_FIX_NVE_H
#define LMP_FIX_NVE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixNVE : public Fix {
 public:
  FixNVE(class LAMMPS *, int, char **);
  virtual ~FixNVE() {}
  int setmask();
  virtual void init();
  virtual void initial_integrate(int);
  virtual void final_integrate();
  virtual void initial_integrate_respa(int, int, int);
  virtual void final_integrate_respa(int, int);
  virtual void reset_dt();
  

  virtual int pack_forward_comm(int, int *, double *, int, int *);
  virtual void unpack_forward_comm(int, int, double *);
  virtual void Nose_Hoover();
  virtual void Berendersen();
  virtual void kinaux(double &,double &); // add by Itai for the iEL_Scf
  void get_names(char *, double *&);
 
  double Omega_t;
  double Omega_s;
  double Omega;
  double gamma_t;
  double gamma_s;
  double Energy_s_init;
  double Energy_t_init;
  int iEL_Scf_flag;
  int thermo_flag;
  int t_s_flag;
  double tautemp_aux;
  double kelvin_aux_t;
  double kelvin_aux_s;
  double tgnhaux [4];
  double tvnhaux [4];
  double tnhaux [4];
  
  double sgnhaux [4];
  double svnhaux [4];
  double snhaux [4]; 
protected:
  double dtv,dtf;
  double *step_respa;
  int mass_require;
  
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

*/
