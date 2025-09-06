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
// class to store vector info
class Vec3D
{
public:
  double x;
  double y;
  double z;
  Vec3D(double x, double y, double z);
  static Vec3D FromSph(double r, double theta, double phi);
  Vec3D operator-(Vec3D);
  Vec3D operator*(double);
  double magnitude();
};

Vec3D::Vec3D(double x, double y, double z) {
  this->x = x;
  this->y = y;
  this->z = z;
}

Vec3D Vec3D::operator-(Vec3D other) {
  return Vec3D(
    x - other.x,
    y - other.y,
    z - other.z
  );
} 

Vec3D Vec3D::operator*(double k) {
  return Vec3D(
    x*k,
    y*k,
    z*k
  );
}

// Vector magnitude
double Vec3D::magnitude() {
  return sqrt(
    x*x + y*y + z*z
  );
}

Vec3D Vec3D::FromSph(double r, double theta, double phi) {
  return Vec3D(
    r*sin(theta)*cos(phi),
    r*sin(theta)*sin(phi),
    r*cos(theta)
  );
}
// Get the vector normal to the midplanet of the disk.
// This assumes the disk with the highest z-value is on the y axis,
// and the midplane intersects the z-axis.
// In other words, the vector points to the second quadrant of the y-z plane.
Vec3D get_nhat_tilt(double angle) {
  return Vec3D(
    0,
    -1.0*sin(angle),
    cos(angle)
  );
}

// Dot product
double dot(Vec3D v1, Vec3D v2) {
  return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

// Vector parallel to normal
Vec3D r_parallel(Vec3D v, double angle) {
  Vec3D nhat = get_nhat_tilt(angle);
  return nhat * dot(v,nhat);
}

// Vector orthogonal to normal
Vec3D r_perpendicular(Vec3D v, double angle) {
  Vec3D r_par = r_parallel(v,angle);
  return v - r_par;
}

// Get the height above the midplane of the inclined disk
double get_z_above_midplane(Vec3D v, double angle) {
  return r_parallel(v,angle).magnitude();
}

// Get the distance from the origin projected into the midplane
double get_midplane_projection_distance(Vec3D v, double angle) {
  return r_perpendicular(v, angle).magnitude();
}

// Vector defining phi=0 in the disk
Vec3D get_ascending_node(double angle) {
  return Vec3D(
    0.0,
    cos(angle),
    sin(angle)
  );
}

Vec3D cross(Vec3D v1, Vec3D v2) {
  return Vec3D(
    v1.y*v2.z - v1.z*v2.y,
    v1.z*v2.x - v2.z*v1.z,
    v1.x*v2.y - v2.x*v1.y
  );
}



//----------------------------------------
// class for planetary system including mass, position, velocity

class BinarySystem
{
public:
  double mass;
  double xp, yp, zp;         // position in Cartesian coord.
  // double vxp, vyp, vzp;      // velocity in Cartesian coord.
  // int FeelOthers;
  BinarySystem();
  ~BinarySystem();
// private:
  // double xpn, ypn, zpn;       // intermediate position for leap-frog integrator
  // double vxpn, vypn, vzpn;
public:
  // void integrate(double dt);     // integrate planetary orbit
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
}

// File scope global variables
// initial condition
static Real gm_primary=0.0;
// radius used to normalize density profile
static Real r0 = 1.0;
static Real omegarot=0.0;
// 1: uniform,     2: step function,    31: disk structure with numerical integration assuming hydrostatic equilibrium
// 3: normal disk structure with the anayltically derived hydrostatic equilibrium, tflag has to be 0
// 4: CPD setup
static int dflag;
// 1: uniform in cartesion coordinate with vy0 along +y direction
// 2: disk velocity with the analytically derived profile, dflag has to be 3 or 4
// 21: similar to 2, but within rigid it is a solid body rotation, dflag has to be 3 or 4
// 22: disk velocity derived by numericlally solving radial pressure gradient
// 3: solid body rotation
// 4: 2D Keplerian velocity
static int vflag;
// tflag   0:  radial power law, vertical isothermal
// 1:  radial power law, vertical within h, T, beyond 4 h, 50*T, between power law
//          dflag==4 CPD centered on the planet
static int tflag;
static Real rho0, rho_floor0, slope_rho_floor, mm, dfloor;
// Density profile cuts
static Real rin, rout, fin, fout;
// Only for vflag=1
static Real dslope, pslope, p0_over_r0;
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
// boundary condition
static std::string ix1_bc, ox1_bc, ix2_bc, ox2_bc;
static int hbc_ix1, hbc_ox1, mbc_ix1, mbc_ox1;
static int hbc_ix2, hbc_ox2, mbc_ix2, mbc_ox2;
// energy related
static double gamma_gas;
static Real tlow, thigh, tcool;
// grid related
static Real x1min, x1max, nx2coarse;

static Real tdamp;
// planetary system
std::ofstream myfile; 
static BinarySystem *psys;
// planetary system: output
static Real timeout=0.0,dtorbit;
// planetary system: circumplanetary disk depletion

// disk inclination
static Real pert(const Real x1, const Real x2, const Real x3);      // perturbation angle 
static Real sin2th(const Real x1, const Real x2, const Real x3);    // 
static int pert_mode;
static Real pert_center, pert_width, pert_amp, pert_cut, diskinc;
// Viscosity
static Real alpha;

AthenaArray<Real> x1area, x2area, x2area_p1, x3area, x3area_p1, vol;

// Functions for initial condition
static Real rho_floor(const Real x1, const Real x2, const Real x3);
static Real rho_floorsf(const Real x1, const Real x2, const Real x3);
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
Real grav_pot_car_btoa(const Real xca, const Real yca, const Real zca,
        const Real xcb, const Real ycb, const Real zcb, const Real gb);

/**
 * Gravitational potential -- indirect term
 * 
 * @param xca x position of the fluid
 * @param yca y position of the fluid
 * @param zca z position of the fluid
 * @param xpp x position of the star
 * @param ypp y position of the star
 * @param zpp z position of the star
 * @param gmp G times stellar mass
 * @returns Gravitational potential
 * 
 * P = GM/|R|^3 * r\cdot R
 */
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

void Cooling(MeshBlock *pmb, const Real dt, const AthenaArray<Real> &prim, 
	     const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);

// Convert from binary-coplanar spherical coordinates to coordinates of the inclined
// disk
Real get_inclined_cyl_z(
  Real r_bin, Real theta_bin, Real phi_bin, Real incl_disk
) {
    return (
      (r_bin*cos(theta_bin) - r_bin*cos(phi_bin)*sin(theta_bin)*tan(incl_disk))
      / cos(incl_disk)
    );
}

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
  gm_primary = pin->GetReal("problem","GM");
  r0 = pin->GetReal("problem","r0");
  omegarot = pin->GetReal("problem","omegarot");

  // Get parameters for initial density and velocity
  rho0 = pin->GetReal("problem","rho0");
  rho_floor0 = pin->GetReal("problem","rho_floor0"); 
  slope_rho_floor = pin->GetReal("problem","slope_rho_floor");
  dflag = pin->GetInteger("problem","dflag");
  vflag = pin->GetInteger("problem","vflag");
  dslope = pin->GetReal("problem","dslope");

  // Get viscosity
  alpha = pin->GetReal("problem","nu_iso");

  // Get the maximum tilt angle
  diskinc = pin->GetReal("problem","diskinc");

  // Get parameters of initial pressure and cooling parameters
  if(NON_BAROTROPIC_EOS){
    tflag = pin->GetInteger("problem","tflag");
    p0_over_r0 = pin->GetReal("problem","p0_over_r0");
    pslope = pin->GetReal("problem","pslope");
    tlow = pin->GetReal("problem","tlow");
    thigh = pin->GetReal("problem","thigh");
    tcool = pin->GetReal("problem","tcool");
    gamma_gas = pin->GetReal("hydro","gamma");
  }else{
    p0_over_r0=SQR(pin->GetReal("hydro","iso_sound_speed"));
  }
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(FLT_MIN))); 
  rin = pin->GetReal("problem", "rin");
  rout = pin->GetReal("problem", "rout");
  fin = pin->GetReal("problem", "fin");
  fout = pin->GetReal("problem", "fout"); 

  // damp quantities
  tdamp = pin->GetReal("problem","tdamp");

  // Get boundary condition flags
  ix1_bc = pin->GetString("mesh","ix1_bc");
  ox1_bc = pin->GetString("mesh","ox1_bc");
  ix2_bc = pin->GetString("mesh","ix2_bc");
  ox2_bc = pin->GetString("mesh","ox2_bc");

  if(ix1_bc == "user") {
    hbc_ix1 = pin->GetInteger("problem","hbc_ix1");
  }
  if(ox1_bc == "user"){
    hbc_ox1 = pin->GetInteger("problem","hbc_ox1");
  }
  if(ix2_bc == "user") {
    hbc_ix2 = pin->GetReal("problem","hbc_ix2");
  }
  if(ox2_bc == "user"){
    hbc_ox2 = pin->GetReal("problem","hbc_ox2");
  }

  // open planetary system and set up variables
  psys = new BinarySystem();

  // for planetary orbit output
  if(Globals::my_rank==0) myfile.open("orbit.txt",std::ios_base::app);
  if (myfile.is_open()&&Globals::my_rank==0) {
    myfile << "time " << "x y z ";
    myfile << '\n' << std::flush;
  }
  dtorbit = pin->GetOrAddReal("planets","dtorbit",0.001);

  // set initial planet properties
  psys->mass = pin->GetReal("secondary", "mass");
  psys->xp = pin->GetReal("secondary", "x0");
  psys->yp = pin->GetReal("secondary", "y0");
  psys->zp = pin->GetReal("secondary", "z0");

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
  // Enroll User Source terms
  
  // EnrollUserExplicitSourceFunction(PlanetarySourceTerms);
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
//======================================================================================
//! \fn void Mesh::TerminateUserMeshProperties(void)
//  \brief Clean up the Mesh properties
//======================================================================================
void Mesh::UserWorkAfterLoop(ParameterInput *pin)
{
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
  std::cout << "[Block" << gid << "] Initializing Density\n";
  for(int k=ks; k<=ke; ++k) {
    std::cout << "[Block" << gid << ", DEN] Starting ix1=" << k << " of " << ke << "\n";
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
	      phydro->u(IDN,k,j,i) = DenProfile(pcoord->x1v(i),pcoord->x2v(j),pcoord->x3v(k));
      }
    }
  }

  //  Initialize velocity
  std::cout << "[Block" << gid << "] Initializing Velocity\n";
  for(int k=ks; k<=ke; ++k) {
    std::cout << "[Block" << gid << " VEL] Starting ix1=" << k << " of " << ke << '\n';
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
    std::cout << "[Block" << gid << "] Initializing Pressure\n";
    for(int k=ks; k<=ke; ++k) {
      std::cout << "[Block" << gid << " PRE] Starting ix1=" << k << " of " << ke << '\n';
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          Real x1 = pcoord->x1v(i);
          Real x2 = pcoord->x2v(j);
          Real p_over_r = PoverR(x1, x2, pcoord->x3v(k)); 
          phydro->u(IEN,k,j,i) = p_over_r*phydro->u(IDN,k,j,i)/(gamma_gas - 1.0);
          phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))+
				       SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
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
    // Real x2h = asin(sqrt(sin2th(x1,x2,x3)));
    // rhof=rho_floorsf(x1,x2h,0.0);
    rhof=rho_floorsf(x1,x2,x3);
  }
  return rhof;
}

Real midplane_density_cutoff_factor(
  const Real r, const Real r_in, const Real r_out,
  const Real slope_in, const Real slope_out
) {
  Real factor_in = powf(
    exp(slope_in * (-1.0*r/r_in + 1.0)) + 1.0,
    -1.0
  );
  Real factor_out = powf(
    exp(slope_out * (r/r_out - 1.0)) + 1.0,
    -1.0
  );
  return factor_in * factor_out;
}

static Real rho_floorsf(const Real x1, const Real x2, const Real x3)
{
  Vec3D v = Vec3D::FromSph(x1,x2,x3);
  Real r = get_midplane_projection_distance(v,diskinc);
  Real z = fabs(get_z_above_midplane(v,diskinc));
  Real zmod = std::max(z,rin);
  Real rhofloor= rho_floor0 * pow(r/r0,slope_rho_floor) * midplane_density_cutoff_factor(
    r,rin,rout,fin,fout
  );
  // if (r<rin) {
  //   // rhofloor=rho_floor0*pow(rin/r0, slope_rho_floor);//*((x1min-r)/x1min*19.+1.);
  //   rhofloor=rho_floor0 * pow(rin/r0, slope_rho_floor);
  //   // rhofloor = rho_floor0*pow(rin/r0, slope_rho_floor)*exp(-1*(rin-r)/sqrt(r*r + zmod*zmod));
    
  //   if(r<3.*rin) rhofloor=rhofloor*(5.-(r-rin)/rin*2.)*((rin-r)/rin*4.+1.);
  // }else{
  //   rhofloor=rho_floor0*pow(r/r0, slope_rho_floor);
  // }
  rhofloor=rhofloor/zmod/zmod*x1min/sqrt(z*z + r*r);
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
    // Real x2h = asin(sqrt(sin2th(x1,x2,x3)));
    // den=DenProfilesf(x1,x2h,0.0);
    den=DenProfilesf(x1,x2,x3);
  }
  return den;
}

static Real DenProfileMod(
  const Real g_m_primary,
  const Real radius_midplane,
  const Real alt_above_midplane,
  const Real midplane_density_fiducial,
  const Real radius_midplane_fiducial,
  const Real _dslope,
  const Real radius_inner_cutoff,
  const Real radius_outer_cutoff,
  const Real inner_cutoff_factor,
  const Real outer_cutoff_factor,
  const Real pressure_over_density
) {
  Real midplane_density = midplane_density_fiducial * pow(radius_midplane/radius_midplane_fiducial,_dslope)
    * midplane_density_cutoff_factor(radius_midplane,radius_inner_cutoff,radius_outer_cutoff,inner_cutoff_factor,outer_cutoff_factor);
  Real zfactor = exp(
    g_m_primary/pressure_over_density*(
      1./sqrt(SQR(radius_midplane)+SQR(alt_above_midplane))-1./radius_midplane
    )
  );
  return midplane_density * zfactor;
}

static Real DenProfilesf(const Real x1, const Real x2, const Real x3)
{  
  Real den;
  std::stringstream msg;
  if (dflag == 1) { // Const.
    den = rho0;
  } else if (dflag == 31) {
    Vec3D v = Vec3D::FromSph(x1,x2,x3);
    Real r = get_midplane_projection_distance(v,diskinc);
    Real z = fabs(get_z_above_midplane(v,diskinc));
    Real denmid = rho0*pow(r/r0,dslope)*midplane_density_cutoff_factor(r,rin,rout,fin,fout);
    Real zo = 0.0;
    Real zn = zo;
    den=denmid;
    Real x1o,x2o,x3o,x1n,x2n,x3n,coe,h,dz,poverro,poverrn; 
    while (zn <= z){
      coe = gm_primary*0.5*(1./sqrt(r*r+zn*zn)-1./sqrt(r*r+zo*zo));
      x1o = sqrt(r*r+zo*zo);
      x2o = atan(r/zo);
      x3o = 0.0;
      poverro=PoverRsf(x1o,x2o,x3o);
      Real poverr_mid=p0_over_r0*pow(r/r0, pslope);
      h = sqrt(poverr_mid)/sqrt(gm_primary/r/r/r);
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
    Vec3D v = Vec3D::FromSph(x1,x2,x3);
    // std::cout<<"v = ("<<v.x<<", "<<v.y<<", "<<v.z<<").\n";
    Real r = get_midplane_projection_distance(v,diskinc);
    Real z = fabs(get_z_above_midplane(v,diskinc));
    // std::cout<<"The point (" << x1 <<", "<<x2<<", "<<x3<<") is at ("<<r<<", "<<z<<").\n";
    Real p_over_r = p0_over_r0;
    if (NON_BAROTROPIC_EOS) p_over_r = PoverRsf(x1, x2, x3);
    // Real denmid;
    // Real zfactor;
    den = DenProfileMod(
      gm_primary,r,z,rho0,r0,dslope,rin,rout,fin,fout,p_over_r
    );
    // if (r > rin) {
    //   denmid = rho0*pow(r/r0,dslope);
    //   zfactor = exp(gm_primary/p_over_r*(1./sqrt(SQR(r)+SQR(z))-1./r));
    // }
    // else {
    //   denmid = rho0*pow(rin/r0,dslope);
    //   zfactor = exp(gm_primary/p_over_r*(1./sqrt(SQR(rin)+SQR(z))-1./rin));
    // } 
    // den = denmid*zfactor;
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
    // Real x2h = asin(sqrt(sin2th(x1,x2,x3)));
    // por=PoverRsf(x1,x2h,0.0);
    por=PoverRsf(x1,x2,x3);
  }
  return por;
}

static Real PoverR_vertical_isothermal(
  const Real midplane_radius,
  const Real midplane_inner_edge_radius,
  const Real p_over_rho_fiducial,
  const Real midplane_radius_fiducial,
  const Real _pslope
) {
  if (midplane_radius < midplane_inner_edge_radius) {
    return p_over_rho_fiducial * pow(midplane_inner_edge_radius/midplane_radius_fiducial,_pslope);
  }
  else {
    return p_over_rho_fiducial * pow(midplane_radius/midplane_radius_fiducial,_pslope);
  }
}


static Real PoverRsf(const Real x1, const Real x2, const Real x3)
{  
  Real poverr;
  Vec3D v = Vec3D::FromSph(x1,x2,x3);
  Real r = get_midplane_projection_distance(v,diskinc);
  Real z = fabs(get_z_above_midplane(v,diskinc));
  if (tflag == 0) {
    poverr = PoverR_vertical_isothermal(
      r,rin,p0_over_r0,r0,pslope
    );
  } else if(tflag == 1){
    Real poverrmid = p0_over_r0*pow(r/r0, pslope);
    Real h = sqrt(poverrmid)/sqrt(gm_primary/r/r/r);
    if (z<h) {
      poverr=poverrmid;
    } else if (z>4*h){
      poverr=50.*poverrmid;
    } else {
      poverr=poverrmid*pow(3.684,(z-h)/h);
    }
  } else if(tflag == 11){
    Real poverrmid = p0_over_r0*pow(r/r0, pslope);
    Real h = sqrt(poverrmid)/sqrt(gm_primary/r/r/r);
    if (z<h) {
      poverr=poverrmid;
    } else if (z>8*h){
      poverr=0.04*gm_primary/r;
    } else {
      poverr=(poverrmid+0.04*gm_primary/r)/2+(0.04*gm_primary/r-poverrmid)/2*sin((z/h-1.)/7*M_PI-M_PI/2.);
    }
  } else if(tflag == 12){
    Real poverrmid = p0_over_r0*pow(r/r0, pslope);
    Real h = sqrt(poverrmid)/sqrt(gm_primary/r/r/r);
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
    poverr = p0_over_r0*pow(r/r0, pslope); 
  } else if(tflag == 5){
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
    // Real sinx2h=sqrt(sin2th(x1,x2,x3));
    // Real x2h = asin(sinx2h);
    // Real v1s, v2s, v3s;
    // VelProfilesf(x1,x2h,x3,den,v1s,v2s,v3s);
    // v1 = v1s;
    // v2 = v3s*sin(pert(x1,x2,x3))*cos(x3)/(sinx2h+1.e-10);
    // v3 = v3s*(cos(pert(x1,x2,x3))*sin(x2) - sin(x3)*sin(pert(x1,x2,x3))*cos(x2))/(sinx2h+1.e-10);
    VelProfilesf(x1,x2,x3,den,v1,v2,v3);
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


static Real Pressure(const Real x1, const Real x2, const Real x3) {
  return PoverRsf(x1, x2, x3)*DenProfilesf(x1, x2, x3);
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
  Vec3D v = Vec3D::FromSph(x1,x2,x3);
  Real r = get_midplane_projection_distance(v,diskinc);
  Real z = fabs(get_z_above_midplane(v,diskinc));
  Real angle_above_midplane = atan2(z,r);
  std::stringstream msg;
  if (vflag == 2 || vflag == 21 || vflag == 22) {       
    Real vel;
    if (den <= rho_floorsf(x1, x2, x3)) { // probably not
      vel = sqrt(
        gm_primary*SQR(fabs(r))
        /(SQR(fabs(r))+SQR(z))
		    /sqrt(SQR(fabs(r))+SQR(z))
      );
    }
    else { // yes
      if (NON_BAROTROPIC_EOS){
        if(vflag==2 || vflag==21){
	        if(dflag!=3&&dflag!=4){
            msg <<"### FATAL ERROR in Problem Generator"  << std::endl
            <<"dflag has to be 3 or 4 when vflag is 2 or 21" << std::endl;
            throw std::runtime_error(msg.str().c_str());
          }
 	        Real p_over_r = PoverRsf(x1, x2, x3);
          vel = (dslope+pslope)*p_over_r/(gm_primary/r) + (1.+pslope) - pslope*r/x1;
          vel = sqrt(gm_primary/r)*sqrt(vel);
        }
        if(vflag==22){ // yes
          Real dx1=0.01*x1;
          Real dx2=PI*0.01;
          Real dx3 = PI*0.01;
          Real grad_pressure_x1 = (Pressure(x1+dx1, x2, x3) - Pressure(x1-dx1, x2, x3)) / (2.0*dx1);
          Real grad_pressure_x2 = (Pressure(x1, x2+dx2, x3) - Pressure(x1, x2-dx2, x3)) / (2.0*dx2*x1);
          Real grad_pressure_x3 = (Pressure(x1, x2, x3+dx2) - Pressure(x1, x2, x3-dx3)) / (2.0*dx3*x1*sin(x2));
          Vec3D grad_pressure = Vec3D::FromSph(
            grad_pressure_x1, grad_pressure_x2, grad_pressure_x3
          );
          Vec3D nhat = get_nhat_tilt(diskinc);
          Vec3D grad_pressure_z = nhat * dot(nhat,grad_pressure);
          Vec3D grad_pressure_r = grad_pressure - grad_pressure_z;
          Vec3D position_on_midplane = r_perpendicular(Vec3D::FromSph(x1,x2,x3), diskinc);
          Vec3D midplane_direction = position_on_midplane * (1.0/position_on_midplane.magnitude());
          Real dpdR = dot(grad_pressure_r, midplane_direction);
          
	        // Real dpdR= (PoverRsf(x1+dx1, x2, x3)*DenProfilesf(x1+dx1, x2, x3)
          //           -PoverRsf(x1-dx1, x2, x3)*DenProfilesf(x1-dx1, x2, x3))/2./dx1*cos(angle_above_midplane)+
		      //           (PoverRsf(x1, x2+dx2, x3)*DenProfilesf(x1, x2+dx2, x3)
          //           -PoverRsf(x1, x2-dx2, x3)*DenProfilesf(x1, x2-dx2, x3))/2./dx2*sin(angle_above_midplane)/x1;
          // Real vel2 = gms*r*r/sqrt(r*r+z*z)/(r*r+z*z)+r/DenProfilesf(x1, x2, x3)*dpdR; // original
          Real vel2 = gm_primary*sqrt(r*r+z*z)/(r*r+z*z)+r/DenProfilesf(x1, x2, x3)*dpdR; // changing grav term
          if (vel2 < 0) {
            std::cout << "\033[1;31m[ERROR]\033[0m Imaginary Velocity at ("
            << x1 << ", " << x2 << ", " << x3 << ")\n"
            << "r = " << r << "\nz = " << z << "\ndpdR = "
            << dpdR << "\nrho = " << DenProfilesf(x1, x2, x3) << "\n";
          }
	        vel = sqrt(
            std::max(
              vel2,
              0.0
            )
          );
        }
      } else {
        vel = dslope*p0_over_r0/(gm_primary/r)+1.0;
        vel = sqrt(gm_primary/r)*sqrt(vel);
      }
    }
    Vec3D nhat = get_nhat_tilt(diskinc);
    Vec3D r_perp = r_perpendicular(v,diskinc);
    Vec3D vel_direction = cross(nhat, r_perp);
    vel_direction = vel_direction * (1/vel_direction.magnitude());
    double direction_x1 = vel_direction.x * sin(x2)*cos(x3) \
                          + vel_direction.y * sin(x2)*sin(x3) \
                          + vel_direction.z * cos(x2);
    double direction_x2 = vel_direction.x * cos(x2)*cos(x3) \
                          + vel_direction.y * cos(x2)*sin(x3) \
                          - vel_direction.z * sin(x2);
    double direction_x3 = vel_direction.x * -1.0 * sin(x3) + vel_direction.y * cos(x3);
    v1 = direction_x1 * vel;
    v2 = direction_x2 * vel;
    v3 = direction_x3 * vel;
  } else if (vflag ==3) {
    v1 = 0.0;
    v2 = 0.0;
    v3 = gm_primary*x1*sin(x2);
  } 
  if(omegarot!=0.0) v3-=omegarot*fabs(x1*sin(x2));
  return;
}


// When there is no perturbation, this reduces to sin^2(theta)
static Real sin2th(const Real x1, const Real x2, const Real x3){
    return (
      SQR(cos(x3))*SQR(sin(x2))
      +SQR(cos(pert(x1,x2,x3)))*SQR(sin(x3))*SQR(sin(x2))
      +SQR(sin(pert(x1,x2,x3)))*SQR(cos(x2))
      - 2.0*cos(pert(x1,x2,x3))*sin(pert(x1,x2,x3))*cos(x2)*sin(x2)*sin(x3)
    );
}

void AlphaViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke)
{
  Coordinates *pco = pmb->pcoord;
  if (alpha > 0.0) {
    // std::cout<<alpha<<std::endl;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma simd
        for (int i=is; i<=ie; ++i){
	        Real r=pco->x1v(i)*sin(pco->x2v(j));
          phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = alpha*PoverR(pco->x1v(i),pco->x2v(j),pco->x3v(k))/sqrt(gm_primary/r/r/r);
	      }
      }
    }
  }
  return;
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
        Real r = std::max(fabs(pco->x1v(i)*sinx2),rin);
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
        Real dtr = std::max(tcool*2.*PI/sqrt(gm_primary/r/r/r),dt);
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
 Real rdi1=1.25*rin;
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
        Real r = std::max(fabs(x1*sinx2),rin);
              // damp timescale
        ramp = 0.0;
        if(x1 < rdi1){
                ramp = (x1-rdi1)/(rdi1-rin);
                ramp = ramp*ramp;
                tau = 2.0*PI*sqrt(r*r*r/gm_primary)*tdamp;
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
  Real dist_sq = (xca-xcb)*(xca-xcb) + (yca-ycb)*(yca-ycb) + (zca-zcb)*(zca-zcb);
  return -gb / dist_sq;
}

/**
 * Gravitational potential -- indirect term
 * 
 * @param xca x position of the fluid
 * @param yca y position of the fluid
 * @param zca z position of the fluid
 * @param xpp x position of the star
 * @param ypp y position of the star
 * @param zpp z position of the star
 * @param gmp G times stellar mass
 * @returns Gravitational potential
 * 
 * P = GM/|R|^3 * r\cdot R
 */
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
    // Real th=atan(psys->yp/psys->xp);
    // if(psys->xp<0.0) th+=PI;
    myfile<<psys->xp<<' '<<psys->yp<<' '<<psys->zp<<' ';
    // <<' '<<psys->vxp<<' '<<psys->vyp<<' '<<psys->vzp<<' ';
    myfile<<'\n'<<std::flush;

    timeout+=dtorbit;
  }
  // x3 is theta
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    Real x3=pco->x3v(k);
    Real cosx3=cos(x3);
    Real sinx3=sin(x3);
    // x2 is phi
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real x2=pco->x2v(j);
      Real cosx2=cos(x2);
      Real sinx2=sin(x2);
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real drs = pco->dx1v(i) / 10000.;
        // Position of fluid in cartesian coords
        Real xcar = pco->x1v(i)*sinx2*cosx3;
        Real ycar = pco->x1v(i)*sinx2*sinx3;
        Real zcar = pco->x1v(i)*cosx2;
        // cartesian force components
        Real f_x1 = 0.0;
        Real f_x2 = 0.0;
        Real f_x3 = 0.0;
        // cartesian coordinates of the secondary
        Real xpp=psys->xp;
        Real ypp=psys->yp;
        Real zpp=psys->zp;
        Real mp=psys->mass;        
        /* forces calculated using gradient of potential */
        Real f_xca = -1.0* (grav_pot_car_btoa(xcar+drs, ycar, zcar,xpp,ypp,zpp,mp)
            -grav_pot_car_btoa(xcar-drs, ycar, zcar,xpp,ypp,zpp,mp))/(2.0*drs);
        Real f_yca = -1.0* (grav_pot_car_btoa(xcar, ycar+drs, zcar,xpp,ypp,zpp,mp)
            -grav_pot_car_btoa(xcar, ycar-drs, zcar,xpp,ypp,zpp,mp))/(2.0*drs);
        Real f_zca = -1.0* (grav_pot_car_btoa(xcar, ycar, zcar+drs,xpp,ypp,zpp,mp)
          -grav_pot_car_btoa(xcar, ycar, zcar-drs,xpp,ypp,zpp,mp))/(2.0*drs);
        // Indirect terms
        f_xca += -1.0* (grav_pot_car_ind(xcar+drs, ycar, zcar,xpp,ypp,zpp,mp)
                                -grav_pot_car_ind(xcar-drs, ycar, zcar,xpp,ypp,zpp,mp))/(2.0*drs);
        f_yca += -1.0* (grav_pot_car_ind(xcar, ycar+drs, zcar,xpp,ypp,zpp,mp)
                                -grav_pot_car_ind(xcar, ycar-drs, zcar,xpp,ypp,zpp,mp))/(2.0*drs);
        f_zca += -1.0* (grav_pot_car_ind(xcar, ycar, zcar+drs,xpp,ypp,zpp,mp)
                                -grav_pot_car_ind(xcar, ycar, zcar-drs,xpp,ypp,zpp,mp))/(2.0*drs);
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
  if(tdamp>0.0) Damp(pmb,dt,prim,bcc,cons);
  if(NON_BAROTROPIC_EOS&&tcool>0.0) Cooling(pmb,dt,prim,bcc,cons);
}

//------------------------------------------
// f: circular planet orbit
//
void BinarySystem::fixorbit(double dt)
{
  double dis=sqrt(xp*xp+yp*yp);
  double ome=sqrt((gm_primary+mass)/dis/dis/dis);
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

void MeshBlock::UserWorkInLoop() {
  psys->fixorbit(pmy_mesh->dt);
}