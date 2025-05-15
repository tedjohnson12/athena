//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// Problem Generator History:
// March-29-2014, Zhaohuan Zhu, support various disk density, velocity,
//                temperature, magnetic fields configurations,and polar boundary
// April-1-2015, Zhaohuan Zhu & Wenhua Ju, add binary/planet in inertial or corotating
//                frame, add stratified x2 boundary condition.             
//
// May 2025, Ted Johnson, Updating to make compatible with latest version of athena.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================

// C++ headers
#include <iostream>   // endl
#include <fstream>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <cmath>      // sqrt
#include <algorithm>  // min
#include <cstdlib>    // srand
#include <cfloat>     // FLT_MIN

// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../hydro/hydro.hpp"
#include "../eos/eos.hpp"
#include "../bvals/bvals.hpp"
#include "../field/field.hpp"
#include "../coordinates/coordinates.hpp"
#include "../field/field_diffusion/field_diffusion.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"

//----------------------------------------
// class for planetary system including mass, position, velocity

class BinarySystem
{
public:
  double mass;
  double xp, yp, zp;         // position in Cartesian coord.
  double vxp, vyp, vzp;      // velocity in Cartesian coord.
  int FeelOthers;
  BinarySystem();
  ~BinarySystem();
private:
  double xpn, ypn, zpn;       // intermediate position for leap-frog integrator
  double vxpn, vypn, vzpn;
public:
  void integrate(double dt);     // integrate planetary orbit
  void fixorbit(double dt);      // circular planetary orbit
  void Rotframe(double dt);      // for frame rotating at omegarot
};

//------------------------------------------
// constructor for planetary system for np planets

/**
 * Initialize a planetary system with `np0` planets.
 * @param np0 Number of planets in the system.
 */
BinarySystem::BinarySystem()
{
  mass = 0.0;
  xp   = 0.0;
  yp   = 0.0;
  zp   = 0.0;
  vxp  = 0.0;
  vyp  = 0.0;
  vzp  = 0.0;
  xpn  = 0.0;
  ypn  = 0.0;
  zpn  = 0.0;
  vxpn = 0.0;
  vypn = 0.0;
  vzpn = 0.0;
  FeelOthers=0;
}

// File scope global variables
// initial condition
static Real gmass_primary=0.0, gms=0.0, gm1=0.0, r0 = 1.0, omegarot=0.0;
static int dflag, vflag, tflag, per;
static Real rho0, rho_floor0, vy0, slope_rho_floor, mm, rrigid, origid, dfloor;
static Real dslope, pslope, p0_over_r0, amp;
static Real ifield,b0,beta;
static Real rcut, rs;
static Real firsttime;
// readin table
static AthenaArray<Real> rtable, ztable, dentable, portable, tdusttable, tgrid; 
static int nrtable, nztable;
// convert unit
static Real TUNIT, LUNIT, MUNIT, PUNIT, TEUNIT; 
// planet center CPD
static Real sl, sh, wtran, gapw, rstart, rtrunc;
// manually cut a gap
static int manualgap;
// boundary condition
static std::string ix1_bc, ox1_bc, ix2_bc, ox2_bc;
static int hbc_ix1, hbc_ox1, mbc_ix1, mbc_ox1;
static int hbc_ix2, hbc_ox2, mbc_ix2, mbc_ox2;
// energy related
static double gamma_gas;
static Real tlow, thigh, tcool;
// grid related
static Real x1min, x1max, nx2coarse, xcut;

static Real xc, yc, zc;
static Real tdamp;
// for x-ray ionization
static Real lumx; 
static int ionization;
// planetary system
std::ofstream myfile; 
static BinarySystem *psys;
static int fixorb;
static int cylpot;
static Real insert_start,insert_time;
static Real rsoft2=0.0;
static int ind;
// planetary system: output
static Real timeout=0.0,dtorbit;
// planetary system: circumplanetary disk depletion
static Real rcird, tcird, dcird,rocird;

// disk inclination
static Real pert(const Real x1, const Real x2, const Real x3);      // perturbation angle 
static Real sin2th(const Real x1, const Real x2, const Real x3);    // 
static int pert_mode;
static Real pert_center, pert_width, pert_amp, pert_cut, diskinc;
// Viscosity
static Real alpha;

// output quantities to ifov
static int ifov_flag=0;
AthenaArray<Real> x1area, x2area, x2area_p1, x3area, x3area_p1, vol;

// Functions for initial condition
static Real rho_floor(const Real x1, const Real x2, const Real x3);
static Real rho_floorsf(const Real x1, const Real x2, const Real x3);
static Real A3(const Real x1, const Real x2, const Real x3);
static Real A2(const Real x1, const Real x2, const Real x3);
static Real A1(const Real x1, const Real x2, const Real x3);
static Real DenProfile(const Real x1, const Real x2, const Real x3);
static Real DenProfilesf(const Real x1, const Real x2, const Real x3);
static Real PoverR(const Real x1, const Real x2, const Real x3);
static Real PoverRsf(const Real x1, const Real x2, const Real x3);
static void VelProfile(const Real x1, const Real x2, const Real x3, const Real den, 
		       Real &v1, Real &v2, Real &v3);
static void VelProfilesf(const Real x1, const Real x2, const Real x3, const Real den,
                       Real &v1, Real &v2, Real &v3);
static Real Interp(const Real r, const Real z, const int nxaxis, const int nyaxis, AthenaArray<Real> &xaxis, 
                   AthenaArray<Real> &yaxis, AthenaArray<Real> &datatable );
// Functions for coordinate conversion
void ConvCarSph(const Real x, const Real y, const Real z, Real &rad, Real &theta, Real &phi);
void ConvSphCar(const Real rad, const Real theta, const Real phi, Real &x, Real &y, Real &z);
void ConvVCarSph(const Real x, const Real y, const Real z, const Real vx, const Real vy, const Real vz, Real &vr, Real &vt, Real &vp);
void ConvVSphCar(const Real rad, const Real theta, const Real phi, const Real vr, const Real vt, const Real vp, Real &vx, Real &vy, Real &vz);
// Planet Potential
// Real grav_pot_car_btoa(const Real xca, const Real yca, const Real zca,
//         const Real xcb, const Real ycb, const Real zcb, const Real gb);
Real grav_pot_car_ind(const Real xca, const Real yca, const Real zca,
        const Real xpp, const Real ypp, const Real zpp, const Real gmp);
// Force on the planet
Real PlanetForce(MeshBlock *pmb, int iout);
// Functions for boundary conditions
// Summary function for InnerX1, OuterX1, InnerX2, OuterX2
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

// Individual BC functions to be called determined by 
// hbc_ix1, hbc_ox1, mbc_ix1, mbc_ox1
// hbc_ix2, hbc_ox2, mbc_ix2, mbc_ox2
static void SteadyInnerX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, 
  		  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
static void SteadyOuterX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, 
		  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
static void DiodeInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
static void DiodeOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
static void UserOutflowInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
static void UserOutflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
static void InflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
static void SteadyInnerX2(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
static void SteadyOuterX2(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
static void StratInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
static void StratOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
static void FieldOutflowInnerX2(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
static void FieldOutflowOuterX2(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                  FaceField &bb, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
// Alpha viscosity
void AlphaViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);

//User defined diffusion function
void my_df(FieldDiffusion *pfdif, MeshBlock *pmb, const AthenaArray<Real> &w,
                      const AthenaArray<Real> &bmag, const int is, const int ie, const int js, const int je, const int ks, const int ke);

// Functions for Planetary Source terms
void PlanetarySourceTerms(
  MeshBlock *pmb,
  const double time,
  const double dt,
  const AthenaArray<Real> &prim,
  const AthenaArray<Real> &prim_scalar,
  const AthenaArray<Real> &bcc,
  AthenaArray<Real> &cons,
  AthenaArray<Real> &cons_scalar
);
void Damp(MeshBlock *pmb, const Real dt, const AthenaArray<Real> &prim, 
	  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
void DepleteCir(MeshBlock *pmb,const Real dt, const AthenaArray<Real> &prim, 
		AthenaArray<Real> &cons);
void Cooling(MeshBlock *pmb, const Real dt, const AthenaArray<Real> &prim, 
	     const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);

//======================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Init the Mesh properties
//======================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin)
{
  firsttime=time;

  x1min=mesh_size.x1min;
  x1max=mesh_size.x1max;

  // Get parameters for gravitatonal potential of central point mass
  gmass_primary = pin->GetOrAddReal("problem","GM",0.0);
  gms=gmass_primary;
  r0 = pin->GetOrAddReal("problem","r0",1.0);
  omegarot = pin->GetOrAddReal("problem","omegarot",0.0);

  // Get parameters for initial density and velocity
  rho0 = pin->GetReal("problem","rho0");
  rho_floor0 = pin->GetReal("problem","rho_floor0"); 
  slope_rho_floor = pin->GetOrAddReal("problem","slope_rho_floor",0.0);
  dflag = pin->GetInteger("problem","dflag");
  vflag = pin->GetInteger("problem","vflag");
  rrigid = pin->GetOrAddReal("problem","rrigid",0.0);
  origid = pin->GetOrAddReal("problem","origid",0.0);
  vy0 = pin->GetOrAddReal("problem","vy0",0.0);
  dslope = pin->GetOrAddReal("problem","dslope",0.0);
  per = pin->GetOrAddInteger("problem","per",0);
  amp = pin->GetOrAddReal("problem","amp",0.0);

  // Get viscosity
  alpha = pin->GetOrAddReal("problem","nu_iso",0.0);

  // Get the maximum tilt angle
  pert_mode = pin->GetOrAddInteger("problem","pertmode",0);
  pert_amp = pin->GetOrAddReal("problem","amplitude",0.0);
  pert_width = pin->GetOrAddReal("problem","width",r0/5.);
  pert_center = pin->GetOrAddReal("problem", "center", r0);
  pert_cut = pin->GetOrAddReal("problem", "pertcut", 0.0);
  diskinc = pin->GetOrAddReal("problem","diskinc",0.0);

  // CPD study or gap opening study
  sl = pin->GetOrAddReal("problem","sl",1.);
  sh = pin->GetOrAddReal("problem","sh",1.);
  wtran = pin->GetOrAddReal("problem","wtran",1.);
  gapw = pin->GetOrAddReal("problem","gapw",0.1);

  manualgap = pin->GetOrAddInteger("problem","manualgap",0);

  if(dflag==4){
    rstart = pin->GetOrAddReal("problem","rstart",x1min);
    rtrunc = pin->GetOrAddReal("problem","rtrunc",x1max);
  }

  // Get parameters of initial pressure and cooling parameters
  if(NON_BAROTROPIC_EOS){
    tflag = pin->GetOrAddInteger("problem","tflag",0);
    p0_over_r0 = pin->GetOrAddReal("problem","p0_over_r0",0.0025);
    pslope = pin->GetOrAddReal("problem","pslope",0.0);
    tlow = pin->GetOrAddReal("problem","tlow",0.0);
    thigh = pin->GetOrAddReal("problem","thigh",0.0);
    tcool = pin->GetOrAddReal("problem","tcool",0.0);
    gamma_gas = pin->GetReal("hydro","gamma");
  }else{
    p0_over_r0=SQR(pin->GetReal("hydro","iso_sound_speed"));
  }
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(FLT_MIN))); 
  xcut = pin->GetOrAddReal("problem","xcut",x1min);

  // damp quantities
  tdamp = pin->GetOrAddReal("problem","tdamp",0.0);

  // Get parameters for X-ray ionization
  lumx = pin->GetOrAddReal("problem","lumx",2.0e30);
  ionization = pin->GetOrAddInteger("problem","ionization",0);

  // Get loop center for field loop tests;
  xc = pin->GetOrAddReal("problem","xc",1.0);
  yc = pin->GetOrAddReal("problem","yc",0.0);
  zc = pin->GetOrAddReal("problem","zc",0.0);
  

  int nuser_out_var=pin->GetOrAddInteger("mesh","nuser_out_var",0);

  // Get IFOV choice
  if(nuser_out_var >=1){
    ifov_flag = pin->GetInteger("problem","ifov_flag");
  }

  // Get boundary condition flags
  ix1_bc = pin->GetOrAddString("mesh","ix1_bc","none");
  ox1_bc = pin->GetOrAddString("mesh","ox1_bc","none");
  ix2_bc = pin->GetOrAddString("mesh","ix2_bc","none");
  ox2_bc = pin->GetOrAddString("mesh","ox2_bc","none");

  if(ix1_bc == "user") {
    hbc_ix1 = pin->GetReal("problem","hbc_ix1");
  }
  if(ox1_bc == "user"){
    hbc_ox1 = pin->GetReal("problem","hbc_ox1");
  }
  if(ix2_bc == "user") {
    hbc_ix2 = pin->GetReal("problem","hbc_ix2");
  }
  if(ox2_bc == "user"){
    hbc_ox2 = pin->GetReal("problem","hbc_ox2");
  }

  // Get circumplanetary disk density depletion
  rcird = pin->GetOrAddReal("problem","rcird",0.0);
  rocird = pin->GetOrAddReal("problem","rocird",1.0e10);
  tcird = pin->GetOrAddReal("problem","tcird",0.0);
  dcird = pin->GetOrAddReal("problem","dcird",0.0);

  //read in table if necessary
  if(dflag==5 || tflag==5){
    LUNIT=1.496e13;   // 1 AU
    TUNIT=6003209.3; // 0.7 solar mass 1 AU velocity,  1 solar mass velocity 5022635.6  1 year/2pi
    MUNIT=1.341914e30;  // 0.10271e15(number density at 1 AU)*LUNIT^3*2.35(mean weight)*1.66054e-24(mole mass)
//    PUNIT=MUNIT/pow(LUNIT,4); // GM*MUNIT/LUNIT^4
    PUNIT=MUNIT/LUNIT/TUNIT/TUNIT;
    TEUNIT=PUNIT/8.3144598e7/(MUNIT/pow(LUNIT,3)); //Pcode/rhocode=Tcode/mu
    Real tdust, tgas, den, pre;
    nrtable = 96;
    nztable = 1779;
    rtable.NewAthenaArray(nrtable);
    ztable.NewAthenaArray(nztable);
    dentable.NewAthenaArray(nztable,nrtable);
    portable.NewAthenaArray(nztable,nrtable);
    tdusttable.NewAthenaArray(nztable,nrtable);
    std::ifstream infile("./Original_data.out");
    if (infile.is_open()){
      for (int i = 0; i < nrtable; i++) {
        for (int j = 0; j < nztable; j++) {
	  infile >> rtable(i) >> ztable(j) >> tgas >> tdust >> den;
	  rtable(i)=rtable(i)/LUNIT;
          ztable(j)=ztable(j)/LUNIT;
	  den=den*2.35*1.66054e-24; // in gram/cm^3
	  dentable(j,i)=den/(MUNIT/pow(LUNIT,3)); // code unit
	  portable(j,i) = tgas/TEUNIT/2.35; 
	  tdusttable(j,i) = tdust;
//	  std::cout<<rtable(i)<<' '<<ztable(j)<<' '<<portable(j,i)<<' '<<dentable(j,i)<<' '<<te<<' '<<den<<std::endl;
        }
      }
    }else{
      std::cout<<"Cannot open table input file"<<std::endl;
    }
    infile.close();
  }
/*
  for (int i = 0; i < nrtable; i++) {
    std::cout<<"r "<<rtable(i)<<std::endl;
  }
  for (int j = 0; j < nztable; j++) {
    std::cout<<"z "<<ztable(j)<<std::endl;
  }
*/
  // open planetary system and set up variables
  ind = pin->GetOrAddInteger("planets","ind",1);
  rsoft2 = pin->GetOrAddReal("planets","rsoft2",0.0);
  Real np = pin->GetOrAddInteger("planets","np",0);
  psys = new BinarySystem();
  fixorb = pin->GetOrAddInteger("planets","fixorb",0);
  insert_start = pin->GetOrAddReal("planets","insert_start",0.0);
  insert_time = pin->GetOrAddReal("planets","insert_time",0.0);
  cylpot = pin->GetOrAddInteger("planets","cylpot",0);

  // for planetary orbit output
  if(Globals::my_rank==0) myfile.open("orbit.txt",std::ios_base::app);
  dtorbit = pin->GetOrAddReal("planets","dtorbit",0.0);

  // set initial planet properties
  char pname[10];
  sprintf(pname,"mass%d",1);
  psys->mass=pin->GetOrAddReal("planets",pname,0.0);
  sprintf(pname,"x%d",1);
  psys->xp=pin->GetOrAddReal("planets",pname,0.0);
  sprintf(pname,"y%d",1);
  psys->yp=pin->GetOrAddReal("planets",pname,0.0);
  sprintf(pname,"z%d",1);
  psys->zp=pin->GetOrAddReal("planets",pname,0.0);
  sprintf(pname,"vx%d",1);
  psys->vxp=pin->GetOrAddReal("planets",pname,0.0);
  sprintf(pname,"vy%d",1);
  psys->vyp=pin->GetOrAddReal("planets",pname,0.0);
  sprintf(pname,"vz%d",1);
  psys->vzp=pin->GetOrAddReal("planets",pname,0.0);
  sprintf(pname,"feel%d",1);
  psys->FeelOthers=pin->GetOrAddInteger("planets",pname,0);

  gm1 = psys->mass;
  if(dflag==4) gms=gm1;

  // setup boundary condition
  if(mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }
  if(mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
  }
  if(mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DiskInnerX2);
  }
  if(mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DiskOuterX2);
  }
  //EnrollDiffusivityFunction(my_df); 
  // Enroll User Source terms
  
  EnrollUserExplicitSourceFunction(PlanetarySourceTerms);
  // Enroll Viscosity
  if (alpha > 0.0) {
    EnrollViscosityCoefficient(AlphaViscosity);
  }
  AllocateUserHistoryOutput(3);
  // EnrollUserHistoryOutput(0, PlanetForce, "fx");
  // EnrollUserHistoryOutput(1, PlanetForce, "fy");
  // EnrollUserHistoryOutput(2, PlanetForce, "fz");

  return;
}

void my_df(FieldDiffusion *pfdif, MeshBlock *pmb, const AthenaArray<Real> &w,
                      const AthenaArray<Real> &bmag, const int is, const int ie, const int js, const int je, const int ks, const int ke)
{ 
  double eta_A,Am,v_A,w_K,rho,c_s,eta_O,Lambda;
  double x1,x2,x3;
for(int k=ks; k<=ke; k++) {
      for(int j=js; j<=je; j++) {
        for(int i=is; i<=ie; i++) {
    x1=pfdif->pmy_block->pcoord->x1v(i);
    x2=pfdif->pmy_block->pcoord->x2v(j);
    x3=pfdif->pmy_block->pcoord->x3v(k);
    rho=pfdif->pmy_block->phydro->u(IDN,k,j,i); 
    c_s=sqrt(PoverR(x1, x2, x3));
    Real r = std::max(fabs(x1*sin(x2)),0.0);
    Real rcut=std::max(fabs(x1*sin(x2)),xcut);
    Real poverrmid = p0_over_r0*pow(r/r0, pslope);
    //std::cout<<"P0/R0 "<<p0_over_r0<<" p/r_mid "<<poverrmid<<std::endl;
    Real z = fabs(x1*cos(x2));
    Real h = sqrt(poverrmid)/sqrt(gms/r/r/r);
    Real hcut = sqrt(poverrmid)/sqrt(gms/rcut/rcut/rcut);

    w_K=sqrt(gmass_primary/(r+TINY_NUMBER)/(r+TINY_NUMBER)/(r+TINY_NUMBER));
    v_A=bmag(k,j,i)/sqrt(rho);
		if(tflag==13){
		Real theta_trans=0.3;
		Real delta_theta = std::fabs(x2-0.5*M_PI);
		Am=0.5/(0.5*(1.0-tanh(2.0*(delta_theta-theta_trans)/sqrt(poverrmid))));
    }
		else if(tflag==12 || tflag==14){
    Real theta_trans=0.3;
    Am=1.0/(0.5*(1.0-tanh(2.0*(z/h*0.1-theta_trans)/0.1)));
    }
		else{
    if (z<4.0*h) {
      Am = 1.0;
      eta_A = v_A*v_A/w_K/Am;
    }
    else if (z>=4.0*h && z<5.0*h){
	Am = 30.0/log(1.25)*log(z/(4.0*h+TINY_NUMBER))+1.0;
      //eta_A = v_A*v_A/w_K/Am;
      //eta_A = v_A*v_A/w_K*exp(-1.0*(z-4.0*h)/0.5/h); 
    }
    else if (z>=5.0*h && z<8.0*h){
        Am = 31.;
    }
    else Am = 31.0*exp(z/h-8.);

    //else Am = 31.0;
    }
    eta_A=v_A*v_A/w_K/Am;
    //eta_A=1e-30;
    if(0 && fabs(x1*cos(x2))<h){
    std::cout<<"Am "<<Am<<" eta_A "<<eta_A<<" v_A "<<v_A<<" bmag "<<bmag(k,j,i)<<" rho "<<rho;
    std::cout<<" P_B "<<bmag(k,j,i)*bmag(k,j,i)/2.0<<" P_g "<<rho*c_s*c_s<<" beta "<<2.0*(rho*c_s*c_s)/(bmag(k,j,i)*bmag(k,j,i))<<" w_K "<<w_K;
    std::cout<<" z/h "<<fabs(x1*cos(x2))/h<<" r "<<x1<<std::endl;
    }

    Lambda=pow(10,2*z/(h+TINY_NUMBER))*1e-3*pow(rcut/2.,4);
    eta_O=v_A*v_A/w_K/Lambda;
    // pfdif->etaB(I_A, k,j,i) = eta_A;
    pfdif->etaB(FieldDiffusion::DiffProcess::ambipolar, k,j,i) = eta_A;
    // pfdif->etaB(I_O, k,j,i) = eta_O;
    pfdif->etaB(FieldDiffusion::DiffProcess::ohmic, k,j,i) = eta_O;

  }
}
}
  return;
}
//======================================================================================
//! \fn void Mesh::TerminateUserMeshProperties(void)
//  \brief Clean up the Mesh properties
//======================================================================================
void Mesh::UserWorkAfterLoop(ParameterInput *pin)
{
  if(ifov_flag==1 or ifov_flag==2){
    x1area.DeleteAthenaArray();
    x2area.DeleteAthenaArray();
    x2area_p1.DeleteAthenaArray();
    x3area.DeleteAthenaArray();
    x3area_p1.DeleteAthenaArray();
    vol.DeleteAthenaArray();
  }

  return;
}


//======================================================================================
//! \file disk.cpp
//  \brief Initializes Keplerian accretion disk in spherical polar coords
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{

  std::srand(gid);

  //  Initialize density
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
	phydro->u(IDN,k,j,i) = DenProfile(pcoord->x1v(i),pcoord->x2v(j),pcoord->x3v(k));
      }
    }
  }


  // add density perturbation, needs to be after ifield 5 which assumes the disk is axisymmetric in the x3 direction so that divB=0.
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
	if(per==0){
 	  phydro->u(IDN,k,j,i) = std::max(phydro->u(IDN,k,j,i)*
		               (1+amp*((double)rand()/(double)RAND_MAX-0.5)), 
		               rho_floor(pcoord->x1v(i),pcoord->x2v(j),pcoord->x3v(k)));
	}
      }
    }
  }

  //  Initialize velocity
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
	Real x1 = pcoord->x1v(i);
        Real x2 = pcoord->x2v(j);
        Real x3 = pcoord->x3v(k);
        Real v1, v2, v3;
        VelProfile(x1, x2, x3, phydro->u(IDN,k,j,i), v1, v2, v3);
        phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i)*v1;
	phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i)*v2;
 	phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i)*v3;
      }
    }
  }
  //  Initialize pressure
  if (NON_BAROTROPIC_EOS){
    for(int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          Real x1 = pcoord->x1v(i);
          Real x2 = pcoord->x2v(j);
          Real r = std::max(fabs(x1*sin(x2)),xcut);
          Real p_over_r = PoverR(x1, x2, pcoord->x3v(k)); 
          phydro->u(IEN,k,j,i) = p_over_r*phydro->u(IDN,k,j,i)/(gamma_gas - 1.0);
          phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))+
				       SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
          // Initialize dust temperature
	  if (ionization==1){
	    Real rnocut = fabs(x1*sin(x2));
            Real z = fabs(x1*cos(x2));
	    ruser_meshblock_data[4](k,j,i)=Interp(rnocut, z, nrtable, nztable, rtable, ztable, tdusttable)/TEUNIT;
	  }
        }
      }
    }
  }

  return;
}

//--------------------------------------------------------------------------------------
//! \fn static Real rho_floor
//  \brief density floor
//  dflag==4 CPD centered on the planet

static Real rho_floor(const Real x1, const Real x2, const Real x3)
{
  Real rhof;
  if(dflag==4){
    Real xsf,ysf,zsf,xpf,ypf,zpf,rsphsf,thetasf,phisf;
    ConvSphCar(x1, x2, x3, xpf, ypf, zpf);
    xsf=xpf-r0;
    ysf=ypf;
    zsf=zpf;
    ConvCarSph(xsf,ysf,zsf,rsphsf,thetasf,phisf);
    rhof=rho_floorsf(rsphsf,thetasf,phisf);
  }else{
    Real x2h = asin(sqrt(sin2th(x1,x2,x3)));
    rhof=rho_floorsf(x1,x2h,0.0); 
  }
  return rhof;
}


static Real rho_floorsf(const Real x1, const Real x2, const Real x3)
{
  Real r = fabs(x1*sin(x2));
  Real zmod = std::max(fabs(x1*cos(x2)),xcut);
  Real rhofloor=0.0;
  if (r<xcut) {
    rhofloor=rho_floor0*pow(xcut/r0, slope_rho_floor);//*((x1min-r)/x1min*19.+1.);
    if(x1<3.*xcut) rhofloor=rhofloor*(5.-(x1-xcut)/xcut*2.)*((xcut-r)/xcut*4.+1.);
  }else{
    rhofloor=rho_floor0*pow(r/r0, slope_rho_floor);
  }
  rhofloor=rhofloor/zmod/zmod*x1min/x1;
  return std::max(rhofloor,dfloor);
}

//---------------------------------------------------------------------------------------
//! \f static Real DenProfile
//  dflag  1: uniform,     2: step function,    31: disk structure with numerical integration assuming hydrostatic equilibrium
//         3: normal disk structure with the anayltically derived hydrostatic equilibrium, tflag has to be 0
//         4: CPD setup
//
static Real DenProfile(const Real x1, const Real x2, const Real x3)
{
  Real den;
  if(dflag==4){
    Real xsf,ysf,zsf,xpf,ypf,zpf,rsphsf,thetasf,phisf;
    ConvSphCar(x1, x2, x3, xpf, ypf, zpf);
    xsf=xpf-r0;
    ysf=ypf;
    zsf=zpf;
    ConvCarSph(xsf,ysf,zsf,rsphsf,thetasf,phisf);
    den=DenProfilesf(rsphsf,thetasf,phisf);
  }else{
    Real x2h = asin(sqrt(sin2th(x1,x2,x3)));
    den=DenProfilesf(x1,x2h,0.0);
  }
  return den;
}

static Real DenProfilesf(const Real x1, const Real x2, const Real x3)
{  
  Real den;
  std::stringstream msg;
  if (dflag == 1) {
    den = rho0;
  } else if(dflag == 2) {
    Real y = x1*fabs(sin(x2))*sin(x3);
    if (y<0.2*x1max)
      den = 0.5*rho0;
    else
      den = rho0;
  } else if (dflag == 31) {
    Real r = std::max(fabs(x1*sin(x2)),xcut);
    Real z = fabs(x1*cos(x2));
    Real denmid = rho0*pow(r/r0,dslope);
    Real zo = 0.0;
    Real zn = zo;
    den=denmid;
    Real x1o,x2o,x3o,x1n,x2n,x3n,coe,h,dz,poverro,poverrn; 
    while (zn <= z){
      coe = gms*0.5*(1./sqrt(r*r+zn*zn)-1./sqrt(r*r+zo*zo));
      x1o = sqrt(r*r+zo*zo);
      x2o = atan(r/zo);
      x3o = 0.0;
      poverro=PoverRsf(x1o,x2o,x3o);
      Real poverr_mid=p0_over_r0*pow(r/r0, pslope);
      h = sqrt(poverr_mid)/sqrt(gms/r/r/r);
      dz = h/32.;
      x1n = sqrt(r*r+zn*zn);
      x2n = atan(r/zn);
      x3n = 0.0;
      poverrn=PoverRsf(x1n,x2n,x3n);
      den = den*(coe+poverro)/(poverrn-coe);
      zo = zn;
      zn = zo+dz;
    }
  } else if (dflag==3){
    if(tflag!=0){
      msg <<"### FATAL ERROR in Problem Generator"  << std::endl
          <<"tflag has to be zero when dflag is 3" << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }
    Real r = std::max(fabs(x1*sin(x2)),xcut);
    Real z = fabs(x1*cos(x2));
    Real p_over_r = p0_over_r0;
    if (NON_BAROTROPIC_EOS) p_over_r = PoverRsf(x1, x2, x3);
    Real denmid = rho0*pow(r/r0,dslope);
    den = denmid*exp(gms/p_over_r*(1./sqrt(SQR(r)+SQR(z))-1./r));
  } else if (dflag==4){ //CPD density
    Real xsf,ysf,zsf,xpf,ypf,zpf,rcylsf,rcyld,den0;
    ConvSphCar(x1, x2, x3, xsf, ysf, zsf);  
    rcylsf=std::max(sqrt(xsf*xsf+ysf*ysf),xcut); 
    rcyld=r0-rcylsf;
    if(rcyld<-gapw) den0=(2.-exp((rcyld+gapw)/wtran))*(sh-sl)/2.+sl;
    if(rcyld>-gapw&&rcyld<0.0) den0=exp((-rcyld-gapw)/wtran)*(sh-sl)/2.+sl;
    if(rcyld<gapw&&rcyld>0.0)  den0=exp((rcyld-gapw)/wtran)*(sh-sl)/2.+sl;
    if(rcyld>gapw)  den0=(2.-exp((gapw-rcyld)/wtran))*(sh-sl)/2.+sl;
    Real p_over_r = p0_over_r0;
    if (NON_BAROTROPIC_EOS) p_over_r = PoverRsf(x1, x2, x3);
    Real denmid = den0*pow(rcylsf/r0,dslope);
    den = denmid*exp(gms/p_over_r*(1./sqrt(SQR(rcylsf)+SQR(zsf))-1./rcylsf));
  } else if (dflag==5){
    Real r = fabs(x1*sin(x2));
    Real z = fabs(x1*cos(x2));
    den = Interp(r, z, nrtable, nztable, rtable, ztable, dentable);
  }
  if (manualgap!=0){
    Real rcyld=fabs(x1*sin(x2))-r0;
    if(rcyld<-gapw) den=den*((2.-exp((rcyld+gapw)/wtran))*(sh-sl)/2.+sl);
    if(rcyld>-gapw&&rcyld<0.0) den=den*(exp((-rcyld-gapw)/wtran)*(sh-sl)/2.+sl);
    if(rcyld<gapw&&rcyld>0.0)  den=den*(exp((rcyld-gapw)/wtran)*(sh-sl)/2.+sl);
    if(rcyld>gapw)  den=den*((2.-exp((gapw-rcyld)/wtran))*(sh-sl)/2.+sl);
  }
  return(std::max(den,rho_floorsf(x1, x2, x3)));
}

static Real Interp(const Real r, const Real z, const int nxaxis, const int nyaxis, AthenaArray<Real> &xaxis, AthenaArray<Real> &yaxis, AthenaArray<Real> &datatable )
{
  Real drf, dzf, data;
  int i=0;
  if(r<xaxis(0)){
    i=0;
    drf=0.0;
  }else if(r>xaxis(nxaxis-1)){
    i=nxaxis-2;
    drf=1.0;
  }else{
    while(i< nxaxis-1 && (r-xaxis(i))*(r-xaxis(i+1))>=0.0){i++;}
    drf=(r-xaxis(i))/(xaxis(i+1)-xaxis(i));
  }
  int j=0;
  if(z<yaxis(0)){
    j=0;
    dzf=0.0;
  }else if(z>yaxis(nyaxis-1)){
    j=nyaxis-2;
    dzf=1.0;
  }else{
    while(j< nyaxis-1 && (z-yaxis(j))*(z-yaxis(j+1))>=0.0){j++;}
    dzf=(z-yaxis(j))/(yaxis(j+1)-yaxis(j));
  }
  data = datatable(j,i)+(datatable(j+1,i)-datatable(j,i))*dzf+
                       (datatable(j,i+1)-datatable(j,i))*drf;
  return data; 
}
//---------------------------------------------------------------------------------------
//! \f static Real PoverR
//  tflag   0:  radial power law, vertical isothermal
//          1:  radial power law, vertical within h, T, beyond 4 h, 50*T, between power law
//          dflag==4 CPD centered on the planet
static Real PoverR(const Real x1, const Real x2, const Real x3)
{
  Real por;
  if(dflag==4){
    Real xsf,ysf,zsf,xpf,ypf,zpf,rsphsf,thetasf,phisf;
    ConvSphCar(x1, x2, x3, xpf, ypf, zpf);
    xsf=xpf-r0;
    ysf=ypf;
    zsf=zpf;
    ConvCarSph(xsf,ysf,zsf,rsphsf,thetasf,phisf);
    por=PoverRsf(rsphsf,thetasf,phisf);
  }else{
    Real x2h = asin(sqrt(sin2th(x1,x2,x3)));
    por=PoverRsf(x1,x2h,0.0);
  }
  return por;
}


static Real PoverRsf(const Real x1, const Real x2, const Real x3)
{  
  Real poverr;
  if (tflag == 0) {
    Real r = std::max(fabs(x1*sin(x2)),xcut);
    poverr = p0_over_r0*pow(r/r0, pslope);
  } else if(tflag == 1){
    Real r = std::max(fabs(x1*sin(x2)),xcut);
    Real poverrmid = p0_over_r0*pow(r/r0, pslope);
    Real z = fabs(x1*cos(x2));
    Real h = sqrt(poverrmid)/sqrt(gms/r/r/r);
    if (z<h) {
      poverr=poverrmid;
    } else if (z>4*h){
      poverr=50.*poverrmid;
    } else {
      poverr=poverrmid*pow(3.684,(z-h)/h);
    }
  } else if(tflag == 11){
    Real r = std::max(fabs(x1*sin(x2)),xcut);
    Real poverrmid = p0_over_r0*pow(r/r0, pslope);
    Real z = fabs(x1*cos(x2));
    Real h = sqrt(poverrmid)/sqrt(gms/r/r/r);
    if (z<h) {
      poverr=poverrmid;
    } else if (z>8*h){
      poverr=0.04*gms/r;
    } else {
      poverr=(poverrmid+0.04*gms/r)/2+(0.04*gms/r-poverrmid)/2*sin((z/h-1.)/7*M_PI-M_PI/2.);
    }
  } else if(tflag == 12){
    Real r = std::max(fabs(x1*sin(x2)),xcut);
    Real poverrmid = p0_over_r0*pow(r/r0, pslope);
    Real z = fabs(x1*cos(x2));
    Real h = sqrt(poverrmid)/sqrt(gms/r/r/r);
		Real pr_hot1 = 4.5*poverrmid;
		Real pr_hot2 = 130.*poverrmid;
    if (z<1.5*h) {
      poverr=poverrmid;
    } else if (z>1.5*h && z<=7.*h){
      poverr=(poverrmid+pr_hot1)/2.+(pr_hot1-poverrmid)/2.*sin((z/h-1.5)/5.5*M_PI-M_PI/2.);
    } else {
      poverr=pr_hot1+(pr_hot2-pr_hot1)*tanh((z/h-7.)/10.);
    }
  } else if(tflag == 13){ // exactly the same as Bai & Stone 2017
    Real r = std::max(fabs(x1*sin(x2)),xcut);
    Real poverrmid = p0_over_r0*pow(r/r0, pslope);
		Real theta_trans=0.3;
    Real HoR0=sqrt(poverrmid);
    Real HoRc=0.3;
    Real HoRp=0.5;
    Real gc,delta_theta,thetat;
		delta_theta = std::fabs(x2-0.5*M_PI)-theta_trans;
    thetat = 0.5*M_PI-theta_trans;
    gc = 1.0 + (HoRc-HoR0+(HoRp-HoRc)*std::max(delta_theta,0.0)/thetat)*0.5*(tanh(delta_theta/HoR0)+1.0)/HoR0;
		poverr = gc*HoR0*gc*HoR0;
  } else if(tflag == 14){
    Real r = std::max(fabs(x1*sin(x2)),xcut);
    Real poverrmid = p0_over_r0*pow(r/r0, pslope);
    Real HoR0=sqrt(p0_over_r0)*pow(r/r0,(pslope+1.)/2.);
    Real HoRc=0.4*pow(r/r0,(pslope+1.)/2.);
    Real HoRp=0.67*pow(r/r0,(pslope+1.)/2.);
    Real theta_trans=atan(3.*HoR0);
    Real gc,delta_theta,thetat;
		delta_theta = std::fabs(x2-0.5*M_PI)-theta_trans;
    thetat = 0.5*M_PI-theta_trans;
    gc = 1.0 + (HoRc-HoR0+(HoRp-HoRc)*std::max(delta_theta,0.0)/thetat)*0.5*(tanh(delta_theta/HoR0)+1.0)/HoR0;
    gc = 1.0 + (HoRc-HoR0+(HoRp-HoRc)*std::max(delta_theta,0.0)/thetat)*0.5*(tanh(delta_theta/atan(HoR0))+1.0)/HoR0;
		poverr = gc*HoR0*gc*HoR0;
  } else if(tflag == 2){
    poverr = p0_over_r0*pow(x1/r0, pslope); 
  } else if(tflag == 5){
    Real r = fabs(x1*sin(x2));
    Real z = fabs(x1*cos(x2));
    poverr = Interp(r, z, nrtable, nztable, rtable, ztable, portable);
//    std::cout<<"por "<<poverr<<std::endl;
  }
  return(poverr);
}

//------------------------------------------------------------------------------------
////! \f horseshoe velocity profile for CPD
//

static void VelProfile(const Real x1, const Real x2, const Real x3,
                                const Real den, Real &v1, Real &v2, Real &v3)
{
  std::stringstream msg;
  Real xsf,ysf,zsf,xpf,ypf,zpf,rcylsf,rcyld,den0, rsphsf, thetasf, phisf;
  Real vrsphsf, vthetasf, vphisf, vxsf, vysf, vzsf;
  if(dflag==4){
    ConvSphCar(x1, x2, x3, xpf, ypf, zpf);
    xsf=xpf-r0;
    ysf=ypf;
    zsf=zpf;
    ConvCarSph(xsf, ysf, zsf, rsphsf, thetasf, phisf);
    VelProfilesf(rsphsf, thetasf, phisf, den, vrsphsf, vthetasf, vphisf); 
    ConvVSphCar(rsphsf, thetasf, phisf, vrsphsf, vthetasf, vphisf, vxsf, vysf, vzsf);
    ConvVCarSph(xpf, ypf, zpf, vxsf, vysf, vzsf, v1, v2, v3);
  }else{
    Real sinx2h=sqrt(sin2th(x1,x2,x3));
    Real x2h = asin(sinx2h);
    Real v1s, v2s, v3s;
    VelProfilesf(x1,x2h,x3,den,v1s,v2s,v3s);
    v1 = v1s;
    v2 = v3s*sin(pert(x1,x2,x3))*cos(x3)/(sinx2h+1.e-10);
    v3 = v3s*(cos(pert(x1,x2,x3))*sin(x2) - sin(x3)*sin(pert(x1,x2,x3))*cos(x2))/(sinx2h+1.e-10);
  }
  return;
}

void ConvCarSph(const Real x, const Real y, const Real z, Real &rad, Real &theta, Real &phi){
  rad=sqrt(x*x+y*y+z*z);
  theta=acos(z/rad);
  phi=atan2(y,x);
  return;
}

void ConvSphCar(const Real rad, const Real theta, const Real phi, Real &x, Real &y, Real &z){
  x=rad*sin(theta)*cos(phi);
  y=rad*sin(theta)*sin(phi);
  z=rad*cos(theta);
  return;
}

void ConvVCarSph(const Real x, const Real y, const Real z, const Real vx, const Real vy, const Real vz, Real &vr, Real &vt, Real &vp){
  Real rads=sqrt(x*x+y*y+z*z);
  Real radc=sqrt(x*x+y*y);
  vr=vx*x/rads+vy*y/rads+vz*z/rads;
  vt=((x*vx+y*vy)*z-radc*radc*vz)/rads/radc;
  vp=vy*x/radc-vx*y/radc;
  return;
}

void ConvVSphCar(const Real rad, const Real theta, const Real phi, const Real vr, const Real vt, const Real vp, Real &vx, Real &vy, Real &vz){
  vx=vr*sin(theta)*cos(phi)+vt*cos(theta)*cos(phi)-vp*sin(phi);
  vy=vr*sin(theta)*sin(phi)+vt*cos(theta)*sin(phi)+vp*cos(phi);
  vz=vr*cos(theta)-vt*sin(theta);
  return;
}


//------------------------------------------------------------------------------------
//! \f velocity profile
// vflag  1: uniform in cartesion coordinate with vy0 along +y direction
//        2: disk velocity with the analytically derived profile, dflag has to be 3 or 4
//        21: similar to 2, but within rigid it is a solid body rotation, dflag has to be 3 or 4
//        22: disk velocity derived by numericlally solving radial pressure gradient
//        3: solid body rotation
//	  4: 2D Keplerian velocity 
//
static void VelProfilesf(const Real x1, const Real x2, const Real x3, 
		       const Real den, Real &v1, Real &v2, Real &v3)
{  
  std::stringstream msg;
  if (vflag == 1) {
    v1 = vy0*sin(x2)*sin(x3);
    v2 = vy0*cos(x2)*sin(x3);
    v3 = vy0*cos(x3);
  } else if (vflag == 2 || vflag == 21 || vflag == 22) {       
    Real r = std::max(fabs(x1*sin(x2)),xcut);
    Real z = fabs(x1*cos(x2));
    Real vel;
    if (den <= (1.+amp)*rho_floorsf(x1, x2, x3)) {
      vel = sqrt(gms*SQR(fabs(x1*sin(x2)))/(SQR(fabs(x1*sin(x2)))+SQR(z))
		 /sqrt(SQR(fabs(x1*sin(x2)))+SQR(z)));
    } else {
      if (NON_BAROTROPIC_EOS){
        if(vflag==2 || vflag==21){
	  if(dflag!=3&&dflag!=4){
            msg <<"### FATAL ERROR in Problem Generator"  << std::endl
            <<"dflag has to be 3 or 4 when vflag is 2 or 21" << std::endl;
            throw std::runtime_error(msg.str().c_str());
          }
 	  Real p_over_r = PoverRsf(x1, x2, x3);
          vel = (dslope+pslope)*p_over_r/(gms/r) + (1.+pslope) - pslope*r/x1;
          vel = sqrt(gms/r)*sqrt(vel);
        }
        if(vflag==22){
          Real dx1=0.01*x1;
          Real dx2=PI*0.01;
	  Real dpdR= (PoverRsf(x1+dx1, x2, x3)*DenProfilesf(x1+dx1, x2, x3)
                    -PoverRsf(x1-dx1, x2, x3)*DenProfilesf(x1-dx1, x2, x3))/2./dx1*sin(x2)+
		   (PoverRsf(x1, x2+dx2, x3)*DenProfilesf(x1, x2+dx2, x3)
                    -PoverRsf(x1, x2-dx2, x3)*DenProfilesf(x1, x2-dx2, x3))/2./dx2*cos(x2)/x1;
	  vel = sqrt(std::max(gms*r*r/sqrt(r*r+z*z)/(r*r+z*z)
		     +r/DenProfilesf(x1, x2, x3)*dpdR,0.0));
          //std::cout<<"VelP"<<std::endl;
        }
      } else {
        vel = dslope*p0_over_r0/(gms/r)+1.0;
        vel = sqrt(gms/r)*sqrt(vel);
      }
    }
    if (vflag == 21) {
      if (x1 <= rrigid) {
	vel=origid*fabs(x1*sin(x2));
      }
    }
    v1 = 0.0;
    v2 = 0.0;
    v3 = vel;
  } else if (vflag ==3) {
    v1 = 0.0;
    v2 = 0.0;
    v3 = gms*x1*sin(x2);
  } 
//  std::cout<<" v3b "<<v3;
  if(omegarot!=0.0) v3-=omegarot*fabs(x1*sin(x2));
//  std::cout<<"omegarot "<<omegarot<<" x1 "<<x1<<" x2 "<<x2<<" x3 "<<x3<<" v3 "<<v3<<std::endl;
  return;
}

static Real sin2th(const Real x1, const Real x2, const Real x3){
    return SQR(cos(x3))*SQR(sin(x2))+SQR(cos(pert(x1,x2,x3)))*SQR(sin(x3))*SQR(sin(x2))+SQR(sin(pert(x1,x2,x3)))*SQR(cos(x2)) - 2.0*cos(pert(x1,x2,x3))*sin(pert(x1,x2,x3))*cos(x2)*sin(x2)*sin(x3);
}

static Real pert(const Real x1, const Real x2, const Real x3){
    Real res;
    if (pert_mode==0) {
      res = diskinc;
    } else if (pert_mode==1) {
      res = pert_amp*exp(-SQR(x1-pert_center)/SQR(pert_width));
    } else if (pert_mode==2) {
      if (x1 <= pert_cut) res=0;
      else res=diskinc;
    } else if (pert_mode==3) {
      res = pert_amp*exp(-SQR(x1-pert_center)/SQR(pert_width));
      if (x1 >= pert_center) res=pert_amp;
    }
    return res;
}

void AlphaViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke)
{
  Coordinates *pco = pmb->pcoord;
  if (alpha > 0.0) {
    std::cout<<alpha<<std::endl;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma simd
        for (int i=is; i<=ie; ++i){
	        Real r=pco->x1v(i)*sin(pco->x2v(j));
          phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = alpha*PoverR(pco->x1v(i),pco->x2v(j),pco->x3v(k))/sqrt(gmass_primary/r/r/r);
	      }
      }
    }
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn static Real A3(const Real x1,const Real x2,const Real x3)
//  \brief A3: 3-component of vector potential
/*
   ifield  1:  constant B field parallel to z xaxis over Z direciton, constatnt over radius beyond rcut, smooth transition to 0 within rs
           6:  constatn B field parallel to z xaxis over Z direction, plasma beta at the midplane is constant over radius
	   7:  similar to 6, but the field becomes a constant over the radius within x1min to avoid the singularity
           8:  dipole field
   per     1:  perturbed along phi direction as |sin(phi)|
 * */
static Real A3(const Real x1, const Real x2, const Real x3)
{  
  Real a3=0.0;
  if(ifield==1) {
    Real r = fabs(x1*sin(x2));
    if(r>=rcut) {
      a3 = r*b0/2.0;
    }else if(r<=rs) {
      a3 = 0.0;
    }else{
      Real a=rs/rcut;
      Real adenom=(-1.+a)*(-1.+a)*(-1.+a);
      Real b1=-3.*(1.+a*a)*b0/adenom;
      Real b2=2.*(1.+a+a*a)*b0/adenom/(1.+a);
      Real b3=(3.+a+a*a+a*a*a)*a*b0/adenom/(1.+a);
      Real c=-a*a*a*b0*rcut*rcut/2./adenom/(1.+a);
      a3 = b1/rcut*r*r/3.+b2/rcut/rcut*r*r*r/4.+b3*r/2.+c/r;
    }    
  }
  if(ifield==6) {
    Real dx2coarse=PI/nx2coarse;
    Real r1 = x1*dx2coarse/2.;
    Real r = fabs(x1*sin(x2))+1.e-6;
    Real a=(pslope+dslope)/2.;
    a3 = b0/pow(r0,a)*pow(r,a+1.)/(a+2.)+b0/pow(r0,a)*pow(r1,a+2.)/r*(1.-1./(a+2.));
  }
  if(ifield==7) {
    Real r = fabs(x1*sin(x2));
    Real a=(pslope+dslope)/2.;
    if (r<=xcut){
      a3 = r/2.0*b0*pow(xcut/r0,a);
    }else{
      a3 = b0/pow(r0,a)*pow(r,a+1.)/(a+2.)+b0*(pow(xcut,a+2)/pow(r0,a)*(1./2.-1./(a+2.)))/r;
    }
  }
  if(ifield==8) {
    Real r = fabs(x1*sin(x2));
    a3 = mm*r/x1/x1/x1; 
  }
  if(per==1){
    a3 = a3*fabs(sin(x3));
  }
  return(a3);
}

//--------------------------------------------------------------------------------------
//! \fn static Real A2(const Real x1,const Real x2,const Real x3)
/*  \brief A2: 2-component of vector potential
    ifield:2  field loop with 1.e-6 amplitude
    ifield:3  uniform field parallel to x xaxis
    ifield:4  uniform field parallel to x xaxis, but when y<1.0, field become half
*/
static Real A2(const Real x1, const Real x2, const Real x3)
 { 
  Real a2=0.0;
  Real az=0.0;
  if(ifield==2) {
    Real x=x1*fabs(sin(x2))*cos(x3);
    Real y=x1*fabs(sin(x2))*sin(x3);
    if(x2<0.0||x2>PI){
     x=-x;
     y=-y;
    }
    Real z=x1*cos(x2);
    if(sqrt(SQR(x-xc)+SQR(y-yc))<=0.5 && fabs(z-zc)<0.2){
      az=1.0e-6*(0.5-sqrt(SQR(x-xc)+SQR(y-yc)));
    }
    a2=-az*fabs(sin(x2));
  }else if(ifield==3){
    Real y=x1*fabs(sin(x2))*sin(x3);
    if(x2<0.0||x2>PI)y=-y;
    a2=-b0*y*fabs(sin(x2));
  }else if(ifield==4){
    Real y=x1*fabs(sin(x2))*sin(x3);
    if(x2<0.0||x2>PI)y=-y;
    a2=-b0*y*fabs(sin(x2));
    a2=a2*sin(y);
  }
  return(a2);
}

//--------------------------------------------------------------------------------------
//! \fn static Real A1(const Real x1,const Real x2,const Real x3)
//  \brief A1: 1-component of vector potential
/*
    ifield:2  field loop with 1.e-6 amplitude
    ifield:3  uniform field parallel to x xaxis
    ifield:4  uniform field parallel to x xaxis, but the amplitude is sin(y) 
  */
static Real A1(const Real x1, const Real x2, const Real x3)
{
  Real a1=0.0;
  Real az=0.0;
  if(ifield==2) {
    Real x=x1*fabs(sin(x2))*cos(x3);
    Real y=x1*fabs(sin(x2))*sin(x3);
    if(x2<0.0||x2>PI){
     x=-x;
     y=-y;
    }
    Real z=x1*cos(x2);
    if(sqrt(SQR(x-xc)+SQR(y-yc))<=0.5 && fabs(z-zc)<0.2){
      az=1.e-6*(0.5-sqrt(SQR(x-xc)+SQR(y-yc)));
    }
    a1=az*cos(x2);
  }else if(ifield==3) {
    Real y=x1*fabs(sin(x2))*sin(x3);
    if(x2<0.0||x2>PI)y=-y;
    a1=b0*y*cos(x2);
  }else if(ifield==4) {
    Real y=x1*fabs(sin(x2))*sin(x3);
    if(x2<0.0||x2>PI)y=-y;
    a1=b0*y*cos(x2);
    a1=a1*sin(y);
  }
  return(a1);
}

//------------------------------------------------------------
// f: User-defined boundary Condition
// 
// Summary of all BCs for inner X1 
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
               Real time, Real dt,
              int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  // inner x1 hydro BCs
  if(hbc_ix1 == 1)
    SteadyInnerX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke);
  else if(hbc_ix1 == 2)
    DiodeInnerX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke);
  else if(hbc_ix1 == 4)
    UserOutflowInnerX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke);
}

// Summary of all BCs for outer X1 
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 double time, double dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  // outer x1 hydro BCs
  if(hbc_ox1 == 1)
    SteadyOuterX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke);
  else if(hbc_ox1 == 2)
    DiodeOuterX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke);
  else if(hbc_ox1 == 3)
    InflowOuterX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke);
  else if(hbc_ox1 == 4)
    UserOutflowOuterX1(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke);
}

// Summary of all BCs for Inner X2
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 double time, double dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  // inner X2 hydro BCs
  if(hbc_ix2 == 1)
    SteadyInnerX2(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke);
  else if(hbc_ix2 == 2)
    StratInnerX2(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke);
}

// Summary of all BCs for Outer X2
void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 double time, double dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  // outer X2 hydro BCs
  if(hbc_ox2 == 1)
    SteadyOuterX2(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke);
  else if(hbc_ox2 == 2)
    StratOuterX2(pmb, pco, prim, b, time, dt, is,ie,js,je,ks,ke);
}



// Hydro BC at inner X1: reset to initial condition
void SteadyInnerX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &bb,
 		   Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
 if(pmb->pmy_mesh->time==firsttime){  // cannot use time since time here is temperaory and can be the halfstep and fullstep time, 
                                      // first time is meshtime which will not change during one whole integration including predict and correct
  for (int k=ks; k<=ke; ++k) { 
    for (int j=js; j<=je; ++j) { 
      for (int i=1; i<=(NGHOST); ++i) {
        Real x1 = pcoord->x1v(is-i);
        Real x2 = pcoord->x2v(j);
        Real x3 = pcoord->x3v(k);
        prim(IDN,k,j,is-i) = DenProfile(x1, x2, x3);

        Real v1, v2, v3;
	
        VelProfile(x1, x2, x3, prim(IDN,k,j,is-i), v1, v2, v3);
        
        prim(IM1,k,j,is-i) = v1;
        prim(IM2,k,j,is-i) = v2;
        prim(IM3,k,j,is-i) = v3;
        if (NON_BAROTROPIC_EOS) 
          prim(IEN,k,j,is-i) = PoverR(x1, x2, x3)*prim(IDN,k,j,is-i);
      }
    }
  }
 }
}

//  Hydro BC at outer X1: reset to initial condition
void SteadyOuterX1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &bb,
                   Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
 if(pmb->pmy_mesh->time==firsttime){ 
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=(NGHOST); ++i) {
          Real x1 = pcoord->x1v(ie+i);
          Real x2 = pcoord->x2v(j);
          Real x3 = pcoord->x3v(k);
          prim(IDN,k,j,ie+i) = DenProfile(x1, x2, x3);

          Real v1, v2, v3;
          VelProfile(x1, x2, x3, prim(IDN,k,j,ie+i), v1, v2, v3);

          prim(IM1,k,j,ie+i) = v1;
          prim(IM2,k,j,ie+i) = v2;
          prim(IM3,k,j,ie+i) = v3;
          if (NON_BAROTROPIC_EOS) prim(IEN,k,j,ie+i) = PoverR(x1, x2, x3)*prim(IDN,k,j,ie+i);
        }
      }
    }
  }
 }
}

// Hydro BC at inner X1: copy density and pressure, diode Vr, copy Vphi, Vtheta=0
void DiodeInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &bb, 
                  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=(NGHOST); ++i) {
        prim(IDN,k,j,is-i) = prim(IDN,k,j,is);

	Real v1, v2, v3;
        Real x1 = pco->x1v(is-i);
        Real x2 = pco->x2v(j);
        Real x3 = pco->x3v(k);
	VelProfile(x1, x2, x3, prim(IDN,k,j,is-i), v1, v2, v3);
        prim(IM1,k,j,is-i) = std::min(prim(IM1,k,j,is), 0.0);
        prim(IM2,k,j,is-i) = 0.0;
        prim(IM3,k,j,is-i) = v3;

        if(NON_BAROTROPIC_EOS) 
          prim(IEN,k,j,is-i) = prim(IEN,k,j,is);
      }
    }
  }
}


// Hydro BC at outer X1: copy density and pressure, diode Vr, copy Vphi, Vtheta=0 
void DiodeOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &bb, 
                  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=(NGHOST); ++i) {
        prim(IDN,k,j,ie+i) = prim(IDN,k,j,ie);

        Real v1, v2, v3;
        Real x1 = pco->x1v(ie+i);
        Real x2 = pco->x2v(j);
        Real x3 = pco->x3v(k);
        VelProfile(x1, x2, x3, prim(IDN,k,j,ie+i), v1, v2, v3);
        prim(IM1,k,j,ie+i) = std::max(prim(IM1,k,j,ie), 0.0);
        prim(IM2,k,j,ie+i) = 0.0;
        prim(IM3,k,j,ie+i) = v3;
        if(NON_BAROTROPIC_EOS)
          prim(IEN,k,j,ie+i) = prim(IEN,k,j,ie);
      }
    }
  }
}

// Hydro BC at inner X1: similar to built-in outflow except do not allow mass flow in to the active region
void UserOutflowInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &bb,
                        Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma simd
        for (int i=1; i<=(NGHOST); ++i) {
          prim(n,k,j,is-i) = prim(n,k,j,is);
          if(n==IVX&&prim(n,k,j,is-i)>0.0) prim(n,k,j,is-i)=0.0;
        }
      }
    }
  }
}

// Hydro BC at outer X1: similar to built-in outflow except do not allow mass flow in to the active region
void UserOutflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &bb,
                        Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma simd
        for (int i=1; i<=(NGHOST); ++i) {
          prim(n,k,j,ie+i) = prim(n,k,j,ie);
          if(n==IVX&&prim(n,k,j,ie+i)<0.0)prim(n,k,j,ie+i)=0.0;
        }
      }
    }
  }
}

//  Hydro BC at outer X1: L1 inflow
void InflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &bb, 
                   Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{

}

// Hydro BC at inner X2: reset to initial condition
void SteadyInnerX2(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &bb,
 		   Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
 if(pmb->pmy_mesh->time==firsttime){ 
  for (int k=ks; k<=ke; ++k) { 
    for (int j=1; j<=(NGHOST); ++j) { 
      for (int i=is; i<=ie; ++i) {
        Real x1 = pcoord->x1v(i);
        Real x2 = pcoord->x2v(js-j);
        Real x3 = pcoord->x3v(k);
        prim(IDN,k,js-j,i) = DenProfile(x1, x2, x3);

        Real v1, v2, v3;
        VelProfile(x1, x2, x3, prim(IDN,k,js-j,i), v1, v2, v3);
        
        prim(IM1,k,js-j,i) = v1;
        prim(IM2,k,js-j,i) = v2;
        prim(IM3,k,js-j,i) = v3;
        if (NON_BAROTROPIC_EOS) 
          prim(IEN,k,js-j,i) = PoverR(x1, x2, x3)*prim(IDN,k,js-j,i);
      }
    }
  }
 }
}

//  Hydro BC at outer X1: reset to initial condition
void SteadyOuterX2(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim, FaceField &bb,
                   Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
 if(pmb->pmy_mesh->time==firsttime){ 
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=(NGHOST); ++j) {
        for (int i=is; i<=ie; ++i) {
          Real x1 = pcoord->x1v(i);
          Real x2 = pcoord->x2v(je+j);
          Real x3 = pcoord->x3v(k);
          prim(IDN,k,je+j,i) = DenProfile(x1, x2, x3);

          Real v1, v2, v3;
          VelProfile(x1, x2, x3, prim(IDN,k,je+j,i), v1, v2, v3);

          prim(IM1,k,je+j,i) = v1;
          prim(IM2,k,je+j,i) = v2;
          prim(IM3,k,je+j,i) = v3;
          if (NON_BAROTROPIC_EOS) prim(IEN,k,je+j,i) = PoverR(x1, x2, x3)*prim(IDN,k,je+j,i);
        }
      }
    }
  }
 }
}


// Hydro BC at inner X2
void StratInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &bb, 
                  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
    for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=(NGHOST); ++j) {
        for (int i=is; i<=ie; ++i) {
	  Real this_rho_floor = rho_floor(pco->x1v(i),pco->x2v(js),pco->x3v(k));
	  Real tempa = prim(IEN,k,js,i) / prim(IDN,k,js,i);
          Real vpa = prim(IM3,k,js,i);

          prim(IDN,k,js-j,i) = prim(IDN,k,js,i)*pow(fabs(sin(pco->x2v(js-j))/sin(pco->x2v(js))),vpa*vpa/tempa);
          if (prim(IDN,k,js-j,i) < this_rho_floor) 
	    prim(IDN,k,js-j,i) = this_rho_floor;
          prim(IM1,k,js-j,i) = 0.0;
          prim(IM2,k,js-j,i) = 0.0;
          prim(IM3,k,js-j,i) = prim(IM3,k,js,i);
          if(NON_BAROTROPIC_EOS) 
	    prim(IEN,k,js-j,i) = tempa * prim(IDN,k,js-j,i);
        }
      }
    }
}

// Hydro BC at outer X2
void StratOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &bb, 
                  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
    for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=(NGHOST); ++j) {
        for (int i=is; i<=ie; ++i) {
          Real this_rho_floor = rho_floor(pco->x1v(i),pco->x2v(je),pco->x3v(k));
	  Real tempa = prim(IEN,k,je,i) / prim(IDN,k,je,i);
          Real vpa = prim(IM3,k,je,i);
          prim(IDN,k,je+j,i) = prim(IDN,k,je,i)*pow(fabs(sin(pco->x2v(je+j))/sin(pco->x2v(je))),vpa*vpa/tempa);
          if (prim(IDN,k,je+j,i) < this_rho_floor) 
	    prim(IDN,k,je+j,i) = this_rho_floor;
          prim(IM1,k,je+j,i) = 0.0;
          prim(IM2,k,je+j,i) = 0.0;
          prim(IM3,k,je+j,i) = prim(IM3,k,je,i);
          if(NON_BAROTROPIC_EOS)
	    prim(IEN,k,je+j,i) = tempa * prim(IDN,k,je+j,i);
        }
      }
    }
}
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{
  int nuser_out=pin->GetOrAddInteger("mesh","nuser_out_var",0);
  AllocateUserOutputVariables(nuser_out);
  if(ifov_flag==1 or ifov_flag==2){
    int nx1 = (ie-is)+1 + 2*(NGHOST);
    x1area.NewAthenaArray(nx1);
    x2area.NewAthenaArray(nx1);
    x2area_p1.NewAthenaArray(nx1);
    x3area.NewAthenaArray(nx1);
    x3area_p1.NewAthenaArray(nx1);
    vol.NewAthenaArray(nx1);
  }

  if (ionization==1){
    int nx1 = (ie-is)+1 + 2*(NGHOST);
    int nx2 = (je-js)+1 + 2*(NGHOST);
    int nx3 = (ke-ks)+1;
    if(block_size.nx3 > 1) nx3 = nx3+2*(NGHOST);
    AllocateRealUserMeshBlockDataField(11);
    AllocateIntUserMeshBlockDataField(1);
    ruser_meshblock_data[0].NewAthenaArray(nx3,nx2,nx1); // local surface density in x1 direction within the meshblock
    ruser_meshblock_data[1].NewAthenaArray(nx3,nx2,nx1); // local surface density in x1 direction within the process
    ruser_meshblock_data[2].NewAthenaArray(nx3*nx2); // total surface density in x1 direction to send to the next process
    ruser_meshblock_data[3].NewAthenaArray(nx3,nx2,nx1); // ionization parameter
    ruser_meshblock_data[4].NewAthenaArray(nx3,nx2,nx1); // dust temperature
    ruser_meshblock_data[5].NewAthenaArray(nx3,nx2,nx1); // local surface density in x2 direction within the meshblock    
    ruser_meshblock_data[6].NewAthenaArray(nx3,nx2,nx1); // local surface density in x2 direction within the process
    ruser_meshblock_data[7].NewAthenaArray(nx3,nx2,nx1); // ionization fraction
    ruser_meshblock_data[8].NewAthenaArray(nx3,nx2,nx1); // Ohmic diffusion coefficients
    ruser_meshblock_data[9].NewAthenaArray(nx3,nx2,nx1); // Hall coefficients
    ruser_meshblock_data[10].NewAthenaArray(nx3,nx2,nx1); // ambipolar diffusion coefficients

    iuser_meshblock_data[0].NewAthenaArray(2); // smallest and biggest loc.lx1 within the process
     
  }
  return;
} 



//----------------------------------------------------------------------
// f: cooling function with damping boundary
// tcool: orbital cooling time
//
void Cooling(MeshBlock *pmb, const Real dt, const AthenaArray<Real> &prim,  
	     const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
 if(tcool>0.0) {
   Coordinates *pco = pmb->pcoord;
   for(int k=pmb->ks; k<=pmb->ke; ++k){
     for (int j=pmb->js; j<=pmb->je; ++j) {
       Real sinx2=sin(pco->x2v(j));
       for (int i=pmb->is; i<=pmb->ie; ++i) {
        if (cons(IDN,k,j,i)<rho_floor(pco->x1v(i),pco->x2v(j),pco->x3v(k))) {
                cons(IDN,k,j,i)=rho_floor(pco->x1v(i),pco->x2v(j),pco->x3v(k));
              }
	      // to avoid the divergence at 0 for both Keplerian motion and p_over_r
        Real r = std::max(fabs(pco->x1v(i)*sinx2),xcut);
        Real eint = cons(IEN,k,j,i)-0.5*(SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))
                            +SQR(cons(IM3,k,j,i)))/cons(IDN,k,j,i);
        Real pres_over_r=eint*(gamma_gas-1.0)/cons(IDN,k,j,i);
        Real p_over_r = PoverR(pco->x1v(i),pco->x2v(j),pco->x3v(k)); 
	      // reset temperature when the temperature is below tlow times the initial temperature
        if(tlow>0 && pres_over_r<tlow*p_over_r){
          eint=tlow*p_over_r*cons(IDN,k,j,i)/(gamma_gas-1.0);
	        cons(IEN,k,j,i)=eint+0.5*(SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))
			             +SQR(cons(IM3,k,j,i)))/cons(IDN,k,j,i);
         }
	      // reset temperature when the temperature is above thigh times the initial temperature
        if(thigh>0 && pres_over_r>thigh*p_over_r){
          eint=thigh*p_over_r*cons(IDN,k,j,i)/(gamma_gas-1.0);
	        cons(IEN,k,j,i)=eint+0.5*(SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))
				     +SQR(cons(IM3,k,j,i)))/cons(IDN,k,j,i);
         }
        Real dtr = std::max(tcool*2.*PI/sqrt(gmass_primary/r/r/r),dt);
        Real dfrac=dt/dtr;
        Real dE=eint-p_over_r/(gamma_gas-1.0)*cons(IDN,k,j,i);
        cons(IEN,k,j,i) -= dE*dfrac;
      }
    }
  }
}
 return;
}

//--------------------------------------------------------------------------------
// f: damp to the initital condition close to the inner boundary
//
//

void Damp(MeshBlock *pmb, const Real dt, const AthenaArray<Real> &prim, 
	  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
 Real rdi1=1.25*xcut;
 Real ramp = 0.0, tau=0.0, lambda=0.0, e_nomag=0.0;
 if(tdamp>0.0) {
  Coordinates *pco = pmb->pcoord;
  for(int k=pmb->ks; k<=pmb->ke; ++k){
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real sinx2=sin(pco->x2v(j));
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real x1 = pco->x1v(i);
        Real x2 = pco->x2v(j);
        Real x3 = pco->x3v(k);
              // to avoid the divergence at 0 for both Keplerian motion and p_over_r
        Real r = std::max(fabs(x1*sinx2),xcut);
              // damp timescale
        ramp = 0.0;
        if(x1 < rdi1){
                ramp = (x1-rdi1)/(rdi1-xcut);
                ramp = ramp*ramp;
                tau = 2.0*PI*sqrt(r*r*r/gmass_primary)*tdamp;
              }
              // desired quantities
        Real den, v1, v2, v3, m1, m2, m3, eint;
              den = DenProfile(x1, x2, x3);

              VelProfile(x1, x2, x3, den, v1, v2, v3);
        m1 = den*v1;
        m2 = den*v2;
        m3 = den*v3;
        if (NON_BAROTROPIC_EOS){
          Real p_over_r = PoverR(x1, x2, x3); 
          eint = p_over_r*den/(gamma_gas - 1.0);
        }
              // damp quantities 
        if(ramp>0.0){
          lambda = ramp/tau*dt;
          cons(IDN,k,j,i)=(cons(IDN,k,j,i)+lambda*den)/(1.+lambda);
          cons(IM1,k,j,i)=(cons(IM1,k,j,i)+lambda*m1)/(1.+lambda);
          cons(IM2,k,j,i)=(cons(IM2,k,j,i)+lambda*m2)/(1.+lambda);     
          cons(IM3,k,j,i)=(cons(IM3,k,j,i)+lambda*m3)/(1.+lambda);
          if(NON_BAROTROPIC_EOS) {
            e_nomag = cons(IEN,k,j,i); 
            e_nomag=(e_nomag+lambda*(eint+0.5*(m1*m1+m2*m2+m3*m3)/den))/(1.+lambda);
            cons(IEN,k,j,i) = e_nomag;
          }  
        }
      }
    }
  }
 }
}

//************************************************
//////* Additional Physical Source Terms 
//////************************************************
////
//////****** Use grav potential to calculate forces 

/**
 * Graviational potential of b on a
 * @param xca x-coord of object a
 * @param yca y-coord of object a
 * @param zca z-coord of object a
 * @param xcb x-coord of object b
 * @param ycb y-coord of object b
 * @param zcb z-coord of object b
 * @param gb Mass of object b
 * @returns The graviational potential experience by object a
 * due to object b.
 */
Real grav_pot_car_btoa(const Real xca, const Real yca, const Real zca,
        const Real xcb, const Real ycb, const Real zcb, const Real gb)
{
  Real dist_sq;
  if(cylpot==1) dist_sq = (xca-xcb)*(xca-xcb) + (yca-ycb)*(yca-ycb);
  else dist_sq = (xca-xcb)*(xca-xcb) + (yca-ycb)*(yca-ycb) + (zca-zcb)*(zca-zcb);
  Real pot = -gb*(dist_sq+1.5*rsoft2)/(dist_sq+rsoft2)/sqrt(dist_sq+rsoft2);
  return(pot);
}

Real grav_pot_car_ind(const Real xca, const Real yca, const Real zca,
        const Real xpp, const Real ypp, const Real zpp, const Real gmp)
{
  Real pdist=sqrt(xpp*xpp+ypp*ypp+zpp*zpp);
  Real pot = gmp/pdist/pdist/pdist*(xca*xpp+yca*ypp+zca*zpp);
  return(pot);
}

Real grav_pot_car_cen(const Real xca, const Real yca, const Real zca) {
// Centrifugal force
  Real pot;
  if(omegarot!=0.0) pot = -0.5*omegarot*omegarot*(xca*xca+yca*yca);
  return(pot);
}

void DepleteCir(MeshBlock *pmb, const Real dt, const AthenaArray<Real> &prim, AthenaArray<Real> &cons)
{
  Coordinates *pco = pmb->pcoord;

  for(int k=pmb->ks; k<=pmb->ke; ++k){
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real R=pco->x1v(i);
        Real th=pco->x2v(j);
        Real phi=pco->x3v(k);
        Real xg = R*sin(th)*cos(phi);
        Real yg = R*sin(th)*sin(phi);
        Real zg = R*cos(th);
        Real rdist = sqrt(SQR(xg-psys->xp)+SQR(yg-psys->yp)+SQR(zg-psys->zp));
        Real v1=0.0;
        Real v2=0.0;
        Real v3=0.0;
        if(rdist>rocird){
          cons(IDN,k,j,i) = dcird;
          cons(IM1,k,j,i) = dcird*v1;
          cons(IM2,k,j,i) = dcird*v2;
          cons(IM3,k,j,i) = dcird*v3;
        }
        if(rdist<3.*rcird){
          Real dtr = std::max(tcird, dt);
          Real dfrac = dt/dtr;
          cons(IDN,k,j,i) -= (cons(IDN,k,j,i) - dcird)*dfrac;
          cons(IM1,k,j,i) -= (cons(IM1,k,j,i) - dcird*v1)*dfrac;
          cons(IM2,k,j,i) -= (cons(IM2,k,j,i) - dcird*v2)*dfrac;
          cons(IM3,k,j,i) -= (cons(IM3,k,j,i) - dcird*v3)*dfrac;
          if(NON_BAROTROPIC_EOS) {
            Real ende=0.5*dcird*(SQR(v1)+SQR(v2)+SQR(v3))+PoverR(R,th,phi)*dcird/(gamma_gas - 1.0);
            cons(IEN,k,j,i) -= (cons(IEN,k,j,i) - ende)*dfrac;
          }
        }
      }
    }
  }
}

//******** Grav force from GM1, and indirect term
void PlanetarySourceTerms(
  MeshBlock *pmb,
  const double time,
  const double dt,
  const AthenaArray<Real> &prim,
  const AthenaArray<Real> &prim_scalar,
  const AthenaArray<Real> &bcc,
  AthenaArray<Real> &cons,
  AthenaArray<Real> &cons_scalar
)
{
  Real src[NHYDRO];
  Coordinates *pco = pmb->pcoord;
  // integrate planet orbit
  if(myfile.is_open()&&Globals::my_rank==0&&time>=timeout) {
    myfile<<time+dt<<' ';
    Real th=atan(psys->yp/psys->xp);
    if(psys->xp<0.0) th+=PI;
    myfile<<psys->xp<<' '<<psys->yp<<' '<<psys->zp
    <<' '<<psys->vxp<<' '<<psys->vyp<<' '<<psys->vzp<<' ';
    myfile<<'\n'<<std::flush;

    timeout+=dtorbit;
  }
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    Real x3=pco->x3v(k);
    Real cosx3=cos(x3);
    Real sinx3=sin(x3);
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real x2=pco->x2v(j);
      Real cosx2=cos(x2);
      Real sinx2=sin(x2);
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real drs = pco->dx1v(i) / 10000.;
        Real xcar = pco->x1v(i)*sinx2*cosx3;
        Real ycar = pco->x1v(i)*sinx2*sinx3;
        Real zcar = pco->x1v(i)*cosx2;
        Real f_x1 = 0.0;
        Real f_x2 = 0.0;
        Real f_x3 = 0.0;
        Real xpp=psys->xp;
        Real ypp=psys->yp;
        Real zpp=psys->zp;
        Real mp;
        /* insert the planet at insert_start and gradually increase its mass during insert_time */
        if(time<insert_start){
          mp = 0.0;
        }else{
          mp = 1.0*std::min(1.0,((time-insert_start+1.e-10)/(insert_time+1.e-10)))*psys->mass;
        }
        /* forces calculated using gradient of potential */
        Real f_xca = -1.0* (grav_pot_car_btoa(xcar+drs, ycar, zcar,xpp,ypp,zpp,mp)
            -grav_pot_car_btoa(xcar-drs, ycar, zcar,xpp,ypp,zpp,mp))/(2.0*drs);
        Real f_yca = -1.0* (grav_pot_car_btoa(xcar, ycar+drs, zcar,xpp,ypp,zpp,mp)
            -grav_pot_car_btoa(xcar, ycar-drs, zcar,xpp,ypp,zpp,mp))/(2.0*drs);
        Real f_zca = -1.0* (grav_pot_car_btoa(xcar, ycar, zcar+drs,xpp,ypp,zpp,mp)
          -grav_pot_car_btoa(xcar, ycar, zcar-drs,xpp,ypp,zpp,mp))/(2.0*drs);
        if(ind!=0){
          f_xca += -1.0* (grav_pot_car_ind(xcar+drs, ycar, zcar,xpp,ypp,zpp,mp)
                                  -grav_pot_car_ind(xcar-drs, ycar, zcar,xpp,ypp,zpp,mp))/(2.0*drs);
          f_yca += -1.0* (grav_pot_car_ind(xcar, ycar+drs, zcar,xpp,ypp,zpp,mp)
                                  -grav_pot_car_ind(xcar, ycar-drs, zcar,xpp,ypp,zpp,mp))/(2.0*drs);
          f_zca += -1.0* (grav_pot_car_ind(xcar, ycar, zcar+drs,xpp,ypp,zpp,mp)
                                  -grav_pot_car_ind(xcar, ycar, zcar-drs,xpp,ypp,zpp,mp))/(2.0*drs);
        }
        f_x1 += f_xca*sinx2*cosx3+f_yca*sinx2*sinx3+f_zca*cosx2;
        f_x2 += f_xca*cosx2*cosx3+f_yca*cosx2*sinx3-f_zca*sinx2;
        f_x3 += f_xca*(-sinx3) + f_yca*cosx3;
        if(omegarot!=0.0) {
          Real omegar=omegarot*cosx2;
          Real omegat=-omegarot*sinx2;
          /* centrifugal force */
          Real f_xca = -1.0* (grav_pot_car_cen(xcar+drs, ycar, zcar)
            -grav_pot_car_cen(xcar-drs, ycar, zcar))/(2.0*drs);
          Real f_yca = -1.0* (grav_pot_car_cen(xcar, ycar+drs, zcar)
            -grav_pot_car_cen(xcar, ycar-drs, zcar))/(2.0*drs);
          Real f_zca = -1.0* (grav_pot_car_cen(xcar, ycar, zcar+drs)
            -grav_pot_car_cen(xcar, ycar, zcar-drs))/(2.0*drs);
          f_x1 += f_xca*sinx2*cosx3+f_yca*sinx2*sinx3+f_zca*cosx2;
          f_x2 += f_xca*cosx2*cosx3+f_yca*cosx2*sinx3-f_zca*sinx2;
          f_x3 += f_xca*(-sinx3) + f_yca*cosx3;
          /* Coriolis force */
          f_x1 -= 2.0*omegat*prim(IM3,k,j,i);
          f_x2 += 2.0*omegar*prim(IM3,k,j,i);
          f_x3 -= 2.0*omegar*prim(IM2,k,j,i)-2.0*omegat*prim(IM1,k,j,i);
        }

        src[IM1] = dt*prim(IDN,k,j,i)*f_x1;
        src[IM2] = dt*prim(IDN,k,j,i)*f_x2;
        src[IM3] = dt*prim(IDN,k,j,i)*f_x3;

        cons(IM1,k,j,i) += src[IM1];
        cons(IM2,k,j,i) += src[IM2];
        cons(IM3,k,j,i) += src[IM3];

        if(NON_BAROTROPIC_EOS) {
          src[IEN] = src[IM1]*prim(IM1,k,j,i)+ src[IM2]*prim(IM2,k,j,i) 
                + src[IM3]*prim(IM3,k,j,i);
          cons(IEN,k,j,i) += src[IEN];
        }
      }
    }
  }
  if(rcird>0.0) DepleteCir(pmb,dt,prim,cons); 
  if(tdamp>0.0) Damp(pmb,dt,prim,bcc,cons);
  if(NON_BAROTROPIC_EOS&&tcool>0.0) Cooling(pmb,dt,prim,bcc,cons);
}

//------------------------------------------
// f: circular planet orbit
//
void BinarySystem::fixorbit(double dt)
{
  double dis=sqrt(xp*xp+yp*yp);
  double ome=sqrt((gmass_primary+mass)/dis/dis/dis);
  double ang=acos(xp/dis);
  if(yp<0.0) ang=2*PI-ang;
  ang += ome*dt;
  xp=dis*cos(ang);
  yp=dis*sin(ang);
  return;
}

//------------------------------------------
//  f: planet position in the frame rotating at omegarot
//
void BinarySystem::Rotframe(double dt)
{
  double dis=sqrt(xp*xp+yp*yp);
  double ang=acos(xp/dis);
  if(yp<0.0) ang=2*PI-ang;
  ang -= omegarot*dt;
  xp=dis*cos(ang);
  yp=dis*sin(ang);
  return;
}

//----------------------------------------------------
// f: planetary orbit integrator
// 
// void BinarySystem::integrate(double dt)
// {
//   int i,j;
//   double forcex,forcey,forcez;
//   double forcexi=0., forceyi=0., forcezi=0.;
//   double *dist;
//   dist=new double[np];
//   for(i=0; i<np; ++i){
//     xpn[i]=xp[i]+vxp[i]*dt/2.;
//     ypn[i]=yp[i]+vyp[i]*dt/2.;
//     zpn[i]=zp[i]+vzp[i]*dt/2.;
//   }
//   for(i=0; i<np; ++i) dist[i]=sqrt(xpn[i]*xpn[i]+ypn[i]*ypn[i]+zpn[i]*zpn[i]);
//   // indirect term (acceleration of the central star) from the gravity of the planets themselves.
//   // will be added to direct force for each planet
//   for(j=0; j<np; ++j){
//     forcexi -= mass[j]/dist[j]/dist[j]/dist[j]*xpn[j];
//     forceyi -= mass[j]/dist[j]/dist[j]/dist[j]*ypn[j];
//     forcezi -= mass[j]/dist[j]/dist[j]/dist[j]*zpn[j];
//   }
//   for(i=0; i<np; ++i){
//     // direct term from the central star
//     forcex= -gmass_primary/dist[i]/dist[i]/dist[i]*xpn[i];
//     forcey= -gmass_primary/dist[i]/dist[i]/dist[i]*ypn[i];
//     forcez= -gmass_primary/dist[i]/dist[i]/dist[i]*zpn[i];
//     forcex += forcexi;
//     forcey += forceyi;
//     forcez += forcezi;
//     // gravity from other planets
//     for(j=0; j<np; ++j){
//       if(j!=i){
//         double dis=(xpn[i]-xpn[j])*(xpn[i]-xpn[j])+(ypn[i]-ypn[j])*
// 		   (ypn[i]-ypn[j])+(zpn[i]-zpn[j])*(zpn[i]-zpn[j]);
//         dis=sqrt(dis);
//         forcex += mass[j]/dis/dis/dis*(xpn[j]-xpn[i]);
//         forcey += mass[j]/dis/dis/dis*(ypn[j]-ypn[i]);
//         forcez += mass[j]/dis/dis/dis*(zpn[j]-zpn[i]);
//       }
//     }
//     vxpn[i] = vxp[i] + forcex*dt;
//     vypn[i] = vyp[i] + forcey*dt;
//     vzpn[i] = vzp[i] + forcez*dt;
//   }
//   for(i=0; i<np; ++i){
//     xpn[i]=xpn[i]+vxpn[i]*dt/2.;
//     ypn[i]=ypn[i]+vypn[i]*dt/2.;
//     zpn[i]=zpn[i]+vzpn[i]*dt/2.;
//   }
//   for(i=0; i<np; ++i){
//     xp[i]=xpn[i];
//     yp[i]=ypn[i];
//     zp[i]=zpn[i];
//     vxp[i]=vxpn[i];
//     vyp[i]=vypn[i];
//     vzp[i]=vzpn[i];
//   }
//   delete[] dist;
//   return;
// }

