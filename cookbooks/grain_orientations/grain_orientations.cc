/*
  Copyright (C) 2015 - 2019 by the authors of the ASPECT code.

 This file is part of ASPECT.

 ASPECT is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2, or (at your option)
 any later version.

 ASPECT is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with ASPECT; see the file LICENSE.  If not see
 <http://www.gnu.org/licenses/>.
 */


#include <aspect/particle/property/interface.h>
#include <aspect/simulator_access.h>
#include <random>

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      /**
       * A class that calculates the evolution of grain orientations in an 
       * olivine aggregate.
       *
       * @ingroup ParticleProperties
       */
      template <int dim>
      class GrainOrientations : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
      {
        public:
          /**
           * Initialization function. This function is called once at the
           * creation of every particle for every property to initialize its
           * value.
           *
           * @param [in] position The current particle position.
           * @param [in,out] particle_properties The properties of the particle
           * that is initialized within the call of this function. The purpose
           * of this function should be to extend this vector by a number of
           * properties.
           */
          virtual
          void
          initialize_one_particle_property (const Point<dim> &position,
                                            std::vector<double> &particle_properties) const;

          /**
           * Update function. This function is called every time an update is
           * request by need_update() for every particle for every property.
           *
           * @param [in] data_position An unsigned integer that denotes which
           * component of the particle property vector is associated with the
           * current property. For properties that own several components it
           * denotes the first component of this property, all other components
           * fill consecutive entries in the @p particle_properties vector.
           *
           * @param [in] position The current particle position.
           *
           * @param [in] solution The values of the solution variables at the
           * current particle position.
           *
           * @param [in] gradients The gradients of the solution variables at
           * the current particle position.
           *
           * @param [in,out] data The properties of the particle
           * that is updated within the call of this function.
           */
          virtual
          void
          update_one_particle_property (const unsigned int data_position,
                                        const Point<dim> &,
                                        const Vector<double> &,
                                        const std::vector<Tensor<1,dim> > &gradients,
                                        const ArrayView<double> &data) const;

          /**
           * This implementation tells the particle manager that
           * we need to update particle properties every time step.
           */
          UpdateTimeFlags
          need_update () const;

          /**
           * Return which data has to be provided to update the property.
           * The grain orientations need the gradients of the velocity.
           */
          virtual
          UpdateFlags
          get_needed_update_flags () const;

          /**
           * Set up the information about the names and number of components
           * this property requires.
           *
           * @return A vector that contains pairs of the property names and the
           * number of components this property plugin defines.
           */
          virtual
          std::vector<std::pair<std::string, unsigned int> >
          get_property_information() const;

          /**
           * Declare the parameters this class takes through input files.
           */
          static
          void
          declare_parameters (ParameterHandler &prm);

          /**
           * Read the parameters this class declares from the parameter file.
           */
          virtual
          void
          parse_parameters (ParameterHandler &prm);

	private:
        /**
         * Calculates derivatives of direction cosines and volume
	 * fractions, and the per-grain strain energies, in an aggregate
         */
        std::vector<double> calculate_derivs (Tensor<2,3> &lij,
				 Tensor<2,3> &eij,
				 double &eps0,
				 unsigned int &n_grains,
				 std::vector<double> &acs_odf_rt) const;
        unsigned int ijkl_ind(unsigned int &i, unsigned int &j) const;
        Tensor<4,3> get_Cijkl_ol() const;
        Tensor<4,3> voigt_avg(std::vector<double> &acs_odf_rt,
                              unsigned int &n_grains) const;
        Tensor<2,6> reduce_81_36(Tensor<4,3> &Cav) const;
        //std::vector<double> a_axis(Tensor<4,3> &Cav) const;


        double stress_exp;  // stress exponent for power law rheology
        double mob;  // grain boundary mobility M*
        double lam;  // nucleation parameter lambda*
        double chi;  // volume fraction chi
        unsigned int n_grains;  // number of grains in aggregate, per particle
        std::vector<double> tau;  // reference dimensionless resolved shear stresses
      };
    }
  }
}



namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      template <int dim>
      void
      GrainOrientations<dim>::initialize_one_particle_property(const Point<dim> &,
                                                    std::vector<double> &data) const
      {
	// Order of particle data vector elements for this plugin:
	//	n_grains (1 element) [number of grains in aggregate/particle]
	//	Fij (9) [finite strain tensor, starts as identity]
	//	acs, odf, rt (9+1+1 = 11 elements PER GRAIN) * n_grains
	//		[direction cosines, volume fractions, stored strain energies]
	//		values start as (9 random cosines), (1/n_grains), (0)

	const double PI = 3.141592653589793;
	data.push_back(n_grains);

	// Fij (finite strain tensor, 3x3) - starts as identity matrix, changes with time
	for (unsigned int i=0; i<3; ++i)
	  {
	  for (unsigned int j=0; j<3; ++j)
	    {
	    if (i==j) data.push_back(1);
	    else data.push_back(0);
	    }
	  }

	// Parameters for each grain:
	//	direction cosines (9 elements, 3x3) (initialized random)
	//	volume fraction (starts out as 1/n_grains)
	//	strain energy (starts out as 0)

	// get random euler angles for all grains (initialize aggregate with random LPO)
	std::random_device rd;   // get a seed for the random number engine
	std::mt19937 gen(rd());  // mersenne twister engine seeded with rd
	std::uniform_real_distribution<> dis(0.0,1.0);  // generator

	double init_vol_frac = 1./n_grains;  // initial volume fraction for each orientation

	for (unsigned int i=0; i<n_grains; ++i)
	  {
	  double a = dis(gen)*2*PI;	// generate a set of random euler angles
	  double g = dis(gen)*2*PI;
	  double b = std::acos(2*dis(gen) - 1);

	  double c1 = std::cos(a);
	  double s1 = std::sin(a);
	  double c2 = std::cos(b);
	  double s2 = std::sin(b);
	  double c3 = std::cos(g);
	  double s3 = std::sin(g);

	  // convert euler angles to direction cosines and push back
	  data.push_back(c1*c3 - c2*s1*s3);  // acs(0,0)
	  data.push_back(c3*s1 + c1*c2*s3);  // acs(0,1)
	  data.push_back(s2*s3);  // acs(0,2)
	  data.push_back(-c1*s3 - c2*c3*s1);  // acs(1,0)
	  data.push_back(c1*c2*c3 - s1*s3);  // acs(1,1)
	  data.push_back(c3*s2);  // acs(1,2)
	  data.push_back(s1*s2);  // acs(2,0)
	  data.push_back(-c1*s2);  // acs(2,1)
	  data.push_back(c2);  // acs(2,2)

	  // push back initial volume fraction for orientation
	  data.push_back(init_vol_frac);

	  // push back initial strain energy (starts at 0)
	  data.push_back(0);
	  }

      for (int i=0; i<6; ++i)
      for (int j=0; j<6; ++j)
        {
        data.push_back(0);  // initial Sav, voigt-averaged elastic tensor
        }

      }

      template <int dim>
      void
      GrainOrientations<dim>::update_one_particle_property(const unsigned int data_position,
                                                const Point<dim> &,
                                                const Vector<double> &,
                                                const std::vector<Tensor<1,dim> > &gradients,
                                                const ArrayView<double> &data) const
      {

	unsigned int n_grains = data[data_position];  // retrieve n_grains for parsing the rest

	// retrieve Fij (finite train tensor, 3x3)
	Tensor<2,3> Fij;
	int pos = data_position + 1;
	for (unsigned int i=0; i<3; ++i)
	  {
	  for (unsigned int j=0; j<3; ++j)
	    {
	    Fij[i][j] = data[pos];
	    pos = pos + 1;
	    }
	  }
	// You may notice that tensors are explicitly sized to rank 2, dim 3 instead of using the dim of 
	// the model. This is because even when the flow is 2D, the grains are still treated as 3D objects
	// and we need to multiply tensors accordingly. For 2D flow, we just treat velocities and gradients
	// in the "1" direction as being 0, and proceed in pseudo-3D for the grains. In 3D, we use the full
	// velocity gradient tensor.

	// get lij (local velocity gradient tensor)
	// The only difference between 2D and 3D flow is this velocity gradient tensor:
	//	for 2D, grab components for directions 0 and 2 (and ignore 1)
	//	for 3D, we just grab the whole tensor
	Tensor<2,3> lij;  // initializes all entries to 0
	if (dim==2)  // we need l(0,0), l(0,2), and l(2,0); l(1,1) is assumed to be 0 and l(2,2) is -l(0,0)-l(1,1)
	{
	lij[0][0] = gradients[this->introspection().component_indices.velocities[0]][0];
       lij[0][1] = 0;
	lij[0][2] = gradients[this->introspection().component_indices.velocities[0]][1];
       lij[1][0] = 0;
       lij[1][1] = 0;
       lij[1][2] = 0;
	lij[2][0] = gradients[this->introspection().component_indices.velocities[1]][0];
       lij[2][1] = 0;
	lij[2][2] = gradients[this->introspection().component_indices.velocities[1]][1];
	} else {  // there's probably a more efficient way to fill this tensor
	lij[0][0] = gradients[this->introspection().component_indices.velocities[0]][0];
	lij[0][1] = gradients[this->introspection().component_indices.velocities[0]][1];
	lij[0][2] = gradients[this->introspection().component_indices.velocities[0]][2];
	lij[1][0] = gradients[this->introspection().component_indices.velocities[1]][0];
	lij[1][1] = gradients[this->introspection().component_indices.velocities[1]][1];
	lij[1][2] = gradients[this->introspection().component_indices.velocities[1]][2];
	lij[2][0] = gradients[this->introspection().component_indices.velocities[2]][0];
	lij[2][1] = gradients[this->introspection().component_indices.velocities[2]][1];
	lij[2][2] = gradients[this->introspection().component_indices.velocities[2]][2];
	}

	// get eij (strain rate tensor) and eps0 (reference strain rate) from lij (velocity gradient tensor)
	Tensor<2,3> eij;
	for (unsigned int i=0; i<3; ++i)
	  {
	  for (unsigned int j=0; j<3; ++j)
	    {
	    if (i==j)
	      {
	      eij[i][j] = lij[i][j];
	      } else {
	      eij[i][j] = (lij[i][j] + lij[j][i])/2.;
	      }
	    }
	  }
	const SymmetricTensor<2,3> eij_sym = symmetrize(eij);
#if DEAL_II_VERSION_GTE(9,0,0)  // eigenvalues() for symmetric tensors was a new feature in dealii 9.0.0
        const std::array<double,3> eij_eigenvalues = eigenvalues(eij_sym);  // eigenvalues are sorted in descending order
	double eps0 = eij_eigenvalues[0];  // reference strain rate is largest eigenvalue of strain rate tensor

#else
        AssertThrow (false, ExcMessage ("Grain orientations require deal.II version 9.0 or higher."));
#endif

	// get timestep
	const double dt = this->get_timestep();

	// step Fij (finite strain tensor) forward in time via lij (vel grad tensor) using RK4
	Tensor<2,3> fsei;
	const Tensor<2,3> kfse1 = lij*Fij*dt;
	fsei = Fij + 0.5*kfse1;
	const Tensor<2,3> kfse2 = lij*fsei*dt;
	fsei = Fij + 0.5*kfse2;
	const Tensor<2,3> kfse3 = lij*fsei*dt;
	fsei = Fij + kfse3;
	const Tensor<2,3> kfse4 = lij*fsei*dt;
	Fij = Fij + (kfse1/2. + kfse2 + kfse3 + kfse4/2.)/3.;
	

	// loop over grains; read back in acs (direction cosines), 
	//				  odf (volume fractions), and 
	//				  rt (stored strain energy)
	//				  into one long array in groups of 11 values per grain
	std::vector<double> acs_odf_rt(11*n_grains,0);    // to hold initial/final values
	std::vector<double> acs_odf_rt_i(11*n_grains,0);  // to hold intermediate values for RK4 steps

	// pos is already set at the right point in the particle data vector because we've gone past
	// the initial parts (n_grains, Fij) at the beginning of this function
	for (unsigned int i=0; i<11*n_grains; ++i)
	  {
	  acs_odf_rt[i] = data[pos];
	  pos = pos + 1;
	  }

	// time-step direction cosines and volume fractions using RK4, and recalculate strain energy
	// check direction cosines and volume fractions for sanity and re-normalize volume fractions at each step
	double acs_test = 0;  // to hold acs value for checking
	double odf_test = 0;  // to hold odf value for checking
	double odf_sum = 0;   // running sum of volume fractions for normalization
	// step 1
	std::vector<double> dot_acs_odf_rt = calculate_derivs(lij,eij,eps0,n_grains,acs_odf_rt);  // calculate derivatives/rt
	std::vector<double> kacs_odf_rt_1(11*n_grains,0);
	for (unsigned int i=0; i<n_grains; ++i)
	  {
	  for (unsigned int j=0; j<9; ++j) // 9 elements for direction cosines
	    {
	    kacs_odf_rt_1[11*i+j] = dot_acs_odf_rt[11*i+j]*dt*eps0;  // RK4 factor
	    acs_test = acs_odf_rt[11*i+j] + 0.5*kacs_odf_rt_1[11*i+j];  // step forward and hold as temp value
	    if (acs_test > 1) acs_test = 1;  // check for reasonable angle values
	    if (acs_test < -1) acs_test = -1;
	    acs_odf_rt_i[11*i+j] = acs_test;  // and save new direction cosine
	    }
	  kacs_odf_rt_1[11*i+9] = dot_acs_odf_rt[11*i+9]*dt*eps0;  // RK4 factor for volume fraction
	  odf_test = acs_odf_rt[11*i+9] + 0.5*kacs_odf_rt_1[11*i+9]; // step forward and hold as temp value
	  if (odf_test < 0) odf_test = 0;  // check for reasonable volume fraction (non-negative)
	  acs_odf_rt_i[11*i+9] = odf_test;  // and save new volume fraction
	  odf_sum = odf_sum + odf_test; // add to sum for later normalization
	  acs_odf_rt_i[11*i+10] = dot_acs_odf_rt[11*i+10]; // strain energy is calculated, not time-stepped
	  }
	for (unsigned int i=0; i<n_grains; ++i)
	  {
	  acs_odf_rt_i[11*i+9] = acs_odf_rt_i[11*i+9]/odf_sum;  // re-normalize volume fractions
	  }

	// step 2
	acs_test = 0;
	odf_test = 0;
	odf_sum = 0;
	dot_acs_odf_rt = calculate_derivs(lij,eij,eps0,n_grains,acs_odf_rt_i);  // derivatives/rt from intermediate
	std::vector<double> kacs_odf_rt_2(11*n_grains,0);
	for (unsigned int i=0; i<n_grains; ++i)
	  {
	  for (unsigned int j=0; j<9; ++j)
	    {
	    kacs_odf_rt_2[11*i+j] = dot_acs_odf_rt[11*i+j]*dt*eps0;
	    acs_test = acs_odf_rt[11*i+j] + 0.5*kacs_odf_rt_2[11*i+j];
	    if (acs_test > 1) acs_test = 1;
	    if (acs_test < -1) acs_test = -1;
	    acs_odf_rt_i[11*i+j] = acs_test;
	    }
	  kacs_odf_rt_2[11*i+9] = dot_acs_odf_rt[11*i+9]*dt*eps0;
	  odf_test = acs_odf_rt[11*i+9] + 0.5*kacs_odf_rt_2[11*i+9];
	  if (odf_test < 0) odf_test = 0;
	  acs_odf_rt_i[11*i+9] = odf_test;
	  odf_sum = odf_sum + odf_test;
	  acs_odf_rt_i[11*i+10] = dot_acs_odf_rt[11*i+10];
	  }
	for (unsigned int i=0; i<n_grains; ++i)
	  {
	  acs_odf_rt_i[11*i+9] = acs_odf_rt_i[11*i+9]/odf_sum;
	  }

	// step 3
	acs_test = 0;
	odf_test = 0;
	odf_sum = 0;
	dot_acs_odf_rt = calculate_derivs(lij,eij,eps0,n_grains,acs_odf_rt_i);
	std::vector<double> kacs_odf_rt_3(11*n_grains,0);
	for (unsigned int i=0; i<n_grains; ++i)
	  {
	  for (unsigned int j=0; j<9; ++j)
	    {
	    kacs_odf_rt_3[11*i+j] = dot_acs_odf_rt[11*i+j]*dt*eps0;
	    acs_test = acs_odf_rt[11*i+j] + kacs_odf_rt_3[11*i+j];
	    if (acs_test > 1) acs_test = 1;
	    if (acs_test < -1) acs_test = -1;
	    acs_odf_rt_i[11*i+j] = acs_test;
	    }
	  kacs_odf_rt_3[11*i+9] = dot_acs_odf_rt[11*i+9]*dt*eps0;
	  odf_test = acs_odf_rt[11*i+9] + kacs_odf_rt_3[11*i+9];
	  if (odf_test < 0) odf_test = 0;
	  acs_odf_rt_i[11*i+9] = odf_test;
	  odf_sum = odf_sum + odf_test;
	  acs_odf_rt_i[11*i+10] = dot_acs_odf_rt[11*i+10];
	  }
	for (unsigned int i=0; i<n_grains; ++i)
	  {
	  acs_odf_rt_i[11*i+9] = acs_odf_rt_i[11*i+9]/odf_sum;
	  }

	// step 4!
	acs_test = 0;
	odf_test = 0;
	odf_sum = 0;
	dot_acs_odf_rt = calculate_derivs(lij,eij,eps0,n_grains,acs_odf_rt_i);
	std::vector<double> kacs_odf_rt_4(11*n_grains,0);
	for (unsigned int i=0; i<n_grains; ++i)
	  {
	  for (unsigned int j=0; j<9; ++j)
	    {
	    kacs_odf_rt_4[11*i+j] = dot_acs_odf_rt[11*i+j]*dt*eps0;
	    acs_test = acs_odf_rt[11*i+j] + (kacs_odf_rt_1[11*i+j]/2. +
					     kacs_odf_rt_2[11*i+j] +
					     kacs_odf_rt_3[11*i+j] +
					     kacs_odf_rt_4[11*i+j]/2.)/3.;  // (almost) final acs/odf
	    if (acs_test > 1) acs_test = 1;
	    if (acs_test < -1) acs_test = -1;
	    acs_odf_rt[11*i+j] = acs_test;
	    }
	  kacs_odf_rt_4[11*i+9] = dot_acs_odf_rt[11*i+9]*dt*eps0;
	  odf_test = acs_odf_rt[11*i+9] + (kacs_odf_rt_1[11*i+9]/2. +
				     kacs_odf_rt_2[11*i+9] +
				     kacs_odf_rt_3[11*i+9] +
				     kacs_odf_rt_4[11*i+9]/2.)/3.;  // (almost) final acs/odf
	  if (odf_test < 0) odf_test = 0;
	  acs_odf_rt[11*i+9] = odf_test;
	  odf_sum = odf_sum + odf_test;
	  acs_odf_rt[11*i+10] = dot_acs_odf_rt[11*i+10]; 
	  }
	for (unsigned int i=0; i<n_grains; ++i)
	  {
	  acs_odf_rt[11*i+9] = acs_odf_rt[11*i+9]/odf_sum;
	  }

	pos = data_position + 1;  // reset pos to just after n_grains (which doesn't change)
	for (unsigned int i=0; i<3; ++i)
	  {
	  for (unsigned int j=0; j<3; ++j)
	    {
	    data[pos] = Fij[i][j];  // put new finite strain tensor into particle data vector
	    pos = pos + 1;
	    }
	  }

	// loop over grains to update acs, odf, and rt in the particle data vector
	for (unsigned int i=0; i<n_grains*11; ++i)
	  {
	  data[pos] = acs_odf_rt[i];
	  pos = pos + 1;
	  }
      Tensor<4,3> Cav = voigt_avg(acs_odf_rt, n_grains);
      Tensor<2,6> Sav = reduce_81_36(Cav);
      for (int i=0; i<6; ++i)
        {
      for (int j=0; j<6; ++j)
        {
        data[pos] = Sav[i][j];
        pos = pos + 1;
        }
        }
      }

      template <int dim>
      std::vector<double>
      GrainOrientations<dim>::
      calculate_derivs (Tensor<2,3> &lij,
			Tensor<2,3> &eij,
			double &eps0,
			unsigned int &n_grains,
			std::vector<double> &acs_odf_rt) const
      {
	Tensor<2,3> lx = lij/eps0;
	Tensor<2,3> ex = eij/eps0;  // non-dimensionalize
	Tensor<3,3> eijk;  // levi-civita
	eijk[0][1][2] = 1;
	eijk[1][2][0] = 1;
	eijk[2][0][1] = 1;
	eijk[0][2][1] = -1;
	eijk[1][0][2] = -1;
	eijk[2][1][0] = -1;

	std::vector<double> dot_all(11*n_grains,0);  // to hold dotted things and rt

	double Emean = 0;  // start accounting for mean energy

	// loop over grains and calculate derivatives
	for (unsigned int i=0; i<n_grains; ++i)
	  {
	  std::array<double,4> bigI = {0,0,0,0};  // initialize some things with zeros
	  std::array<double,4> gam = {0,0,0,0};
	  Tensor<2,3> g;

	  // retrieve the parameters for this grain
	  Tensor<2,3> acs_in;
	  acs_in[0][0] = acs_odf_rt[11*i];
	  acs_in[0][1] = acs_odf_rt[11*i+1];
	  acs_in[0][2] = acs_odf_rt[11*i+2];
	  acs_in[1][0] = acs_odf_rt[11*i+3];
	  acs_in[1][1] = acs_odf_rt[11*i+4];
	  acs_in[1][2] = acs_odf_rt[11*i+5];
	  acs_in[2][0] = acs_odf_rt[11*i+6];
	  acs_in[2][1] = acs_odf_rt[11*i+7];
	  acs_in[2][2] = acs_odf_rt[11*i+8];

	  double odf_in = acs_odf_rt[11*i+9];
	  //double rt_in = acs_odf_rt[i+10]; // this doesn't actually get used in the derivative calc

	  // calculate I
	  for (unsigned int j=0; j<3; ++j)
	    {
	    for (unsigned int k=0; k<3; ++k)
	      {
		// bigI sums eij (strain rate tensor) multiplied by unit vectors for slip dir [] and slip plane ()
		// for each of the 4 slip systems in olivine
		bigI[0] = bigI[0] + ex[j][k]*acs_in[0][j]*acs_in[1][k]; // [100](010)
		bigI[1] = bigI[1] + ex[j][k]*acs_in[0][j]*acs_in[2][k]; // [100](001)
		bigI[2] = bigI[2] + ex[j][k]*acs_in[2][j]*acs_in[1][k]; // [001](010)
		bigI[3] = bigI[3] + ex[j][k]*acs_in[2][j]*acs_in[0][k]; // [001](100)
	      }
	    }

	  // calculate I/tau for each slip system (tau is the dimensionless resolved shear stress for
	  // each slip system)
	  std::array<double,4> qab = {0,0,0,0};
	  for (unsigned int j=0; j<4; ++j) qab[j] = std::abs(bigI[j]/tau[j]);

	  // sort I/tau to figure out which is the weakest slip system
	  int imax = std::max_element(qab.begin(),qab.end()) - qab.begin();
	  qab[imax] = -1;
	  int iint = std::max_element(qab.begin(),qab.end()) - qab.begin();
	  qab[iint] = -1;
	  int imin = std::max_element(qab.begin(),qab.end()) - qab.begin();
	  qab[imin] = -1;
	  int inac = std::max_element(qab.begin(),qab.end()) - qab.begin();

	  // calculate weighting factors (gam; beta in paper) relative to the value for which I/tau is largest
	  gam[imax] = 1.;

	  double rat = tau[imax]/bigI[imax];
	  double qint = rat*bigI[iint]/tau[iint];
	  double qmin = rat*bigI[imin]/tau[imin];
	  double sn1 = stress_exp - 1;

	  gam[iint] = qint*std::pow(std::abs(qint),sn1);
	  gam[imin] = qmin*std::pow(std::abs(qmin),sn1);
	  gam[inac] = 0.;  // tau[inac] -> inf

	  // calculate g, the slip tensor, from gam and unit vectors
	  for (unsigned int j=0; j<3; ++j)
	    {
	    for (unsigned int k=0; k<3; ++k)
	      {
		g[j][k] =  2.0*(gam[0]*acs_in[0][j]*acs_in[1][k] + 
				gam[1]*acs_in[0][j]*acs_in[2][k] +
				gam[2]*acs_in[2][j]*acs_in[1][k] +
				gam[3]*acs_in[2][j]*acs_in[0][k]);
	      }
	    }

	  // calculate the strain rate on the softest slip system (gamma in paper)
	  double R1 = 0.;
	  double R2 = 0.;
	  for (unsigned int j=0; j<3; ++j)
	    {
	    unsigned int k = j + 2;
	    if (k > 2) k = k-3;

	    R1 = R1 - (g[j][k]-g[k][j])*(g[j][k]-g[k][j]);
	    R2 = R2 - (g[j][k]-g[k][j])*(lx[j][k]-lx[k][j]);
	    for (unsigned int kk=0; kk<3; ++kk)
	      {
	      R1 = R1 + 2.0*g[j][kk]*g[j][kk];
	      R2 = R2 + 2.0*lx[j][kk]*g[j][kk];
	      }
	    }

	  double gam0 = R2/R1;

	  // calculate dislocation density (disl density ~ stress to the power p=1.5)
	  double rt1 = std::pow(tau[imax],1.5-stress_exp)*std::pow(std::abs(gam[imax]*gam0),1.5/stress_exp);
	  double rt2 = std::pow(tau[iint],1.5-stress_exp)*std::pow(std::abs(gam[iint]*gam0),1.5/stress_exp);
	  double rt3 = std::pow(tau[imin],1.5-stress_exp)*std::pow(std::abs(gam[imin]*gam0),1.5/stress_exp);
	  double rt4 = std::pow(tau[inac],1.5-stress_exp)*std::pow(std::abs(gam[inac]*gam0),1.5/stress_exp);

	  // calculate stored strain energy using nucleation parameter (lam)
	  double rt_out = rt1*std::exp(-lam*std::pow(rt1,2)) +
			  rt2*std::exp(-lam*std::pow(rt2,2)) +
			  rt3*std::exp(-lam*std::pow(rt3,2)) +
			  rt4*std::exp(-lam*std::pow(rt4,2));

	  // calculate rotation rate (omega in paper) (i+1, i+2, pacman around (0,1,2))
	  std::array<double,3> rot = {0,0,0};
	  rot[2] = (lx[1][0]-lx[0][1])/2.0 - (g[1][0]-g[0][1])/2.0*gam0;
	  rot[1] = (lx[0][2]-lx[2][0])/2.0 - (g[0][2]-g[2][0])/2.0*gam0;
	  rot[0] = (lx[2][1]-lx[1][2])/2.0 - (g[2][1]-g[1][2])/2.0*gam0;

	  // calculate derivatives of direction cosines
	  Tensor<2,3> dot_acs;
	  for (unsigned int i1=0; i1<3; ++i1)
	    {
	    for (unsigned int i2=0; i2<3; ++i2)
	      {
	      for (unsigned int i3=0; i3<3; ++i3)
	        {
	        for (unsigned int i4=0; i4<3; ++i4)
	          {
                 dot_acs[i1][i2] = dot_acs[i1][i2] + eijk[i2][i3][i4]*acs_in[i1][i4]*rot[i3];
	          }
	        }
	      }
	    }
         if (odf_in < chi/n_grains)  // if not, dot_acs will be zeros as initialized
           {
           rt_out = 0; // for small grains (volume fraction below threshold), strain energy goes to 0
	    for (unsigned int i1=0; i1<3; ++i1)
	      {
	      for (unsigned int i2=0; i2<3; ++i2)
	        {
               dot_acs[i1][i2] = 0;
               }
             }
	    }

	  Emean = Emean + odf_in*rt_out;  // add to vol-averaged strain energy

	  // put derivatives of direction cosines and stored strain energy into output vector
	  dot_all[11*i] = dot_acs[0][0];
	  dot_all[11*i+1] = dot_acs[0][1];
	  dot_all[11*i+2] = dot_acs[0][2];
	  dot_all[11*i+3] = dot_acs[1][0];
	  dot_all[11*i+4] = dot_acs[1][1];
	  dot_all[11*i+5] = dot_acs[1][2];
	  dot_all[11*i+6] = dot_acs[2][0];
	  dot_all[11*i+7] = dot_acs[2][1];
	  dot_all[11*i+8] = dot_acs[2][2];

	  dot_all[11*i+10] = rt_out;

	  } // end loop over grains

	// now that we have the full volume-averaged  strain energy, go back and loop to get
	// time derivatives of volume fractions via grain boundary migration
	for (unsigned int i=0; i<n_grains; ++i)
	  {
	  dot_all[11*i+9] = mob*acs_odf_rt[11*i+9]*(Emean-dot_all[11*i+10]);
	  }  // end loop over grains again

	return dot_all;
      }


      template <int dim>
      Tensor<4,3>
      GrainOrientations<dim>::
      get_Cijkl_ol () const
      {
      Tensor<4,3> Cijkl_ol;
      Cijkl_ol[0][0][0][0] = 320.71;
      Cijkl_ol[0][0][1][1] = 69.74;
      Cijkl_ol[0][0][2][2] = 71.22;
      Cijkl_ol[0][1][0][1] = 78.36;
      Cijkl_ol[0][1][1][0] = 78.36;
      Cijkl_ol[0][2][0][2] = 77.67;
      Cijkl_ol[0][2][2][0] = 77.67;
      Cijkl_ol[1][0][0][1] = 78.36;
      Cijkl_ol[1][0][1][0] = 78.36;
      Cijkl_ol[1][1][0][0] = 69.84;
      Cijkl_ol[1][1][1][1] = 197.25;
      Cijkl_ol[1][1][2][2] = 74.80;
      Cijkl_ol[1][2][1][2] = 63.77;
      Cijkl_ol[1][2][2][1] = 63.77;
      Cijkl_ol[2][0][0][2] = 77.67;
      Cijkl_ol[2][0][2][0] = 77.67;
      Cijkl_ol[2][1][1][2] = 63.77;
      Cijkl_ol[2][1][2][1] = 63.77;
      Cijkl_ol[2][2][0][0] = 71.22;
      Cijkl_ol[2][2][1][1] = 74.80;
      Cijkl_ol[2][2][2][2] = 234.32;

      return Cijkl_ol;
      }

      template <int dim>
      unsigned int 
      GrainOrientations<dim>::
      ijkl_ind(unsigned int &i, unsigned int &j) const
      {
      Tensor<2,3> ijkl;
      ijkl[0][0] = 0;
      ijkl[0][1] = 5;
      ijkl[0][2] = 4;
      ijkl[1][0] = 5;
      ijkl[1][1] = 1;
      ijkl[1][2] = 3;
      ijkl[2][0] = 4;
      ijkl[2][1] = 3;
      ijkl[2][2] = 2;

      return ijkl[i][j];
      }

      template <int dim>
      Tensor<4,3> 
      GrainOrientations<dim>::
      voigt_avg(std::vector<double> &acs_odf_rt,
                unsigned int &n_grains) const
      {
      Tensor<4,3> C0 = get_Cijkl_ol();
      Tensor<4,3> Cav;

      for (unsigned int ng=0; ng<n_grains; ++ng)
        {
         // retrieve the parameters for this grain
	  Tensor<2,3> acs_in;
	  acs_in[0][0] = acs_odf_rt[11*ng];
	  acs_in[0][1] = acs_odf_rt[11*ng+1];
	  acs_in[0][2] = acs_odf_rt[11*ng+2];
	  acs_in[1][0] = acs_odf_rt[11*ng+3];
	  acs_in[1][1] = acs_odf_rt[11*ng+4];
	  acs_in[1][2] = acs_odf_rt[11*ng+5];
	  acs_in[2][0] = acs_odf_rt[11*ng+6];
	  acs_in[2][1] = acs_odf_rt[11*ng+7];
	  acs_in[2][2] = acs_odf_rt[11*ng+8];
	  double odf_in = acs_odf_rt[11*ng+9];

        Tensor<4,3> Cav2;

        for (int i=0; i<3; ++i)
          {
        for (int j=0; j<3; ++j)
          {
        for (int k=0; k<3; ++k)
          {
        for (int l=0; l<3; ++l)
          {
          for (int p=0; p<3; ++p)
            {
          for (int q=0; q<3; ++q)
            {
          for (int r=0; r<3; ++r)
            {
          for (int s=0; s<3; ++s)
            {
            Cav2[i][j][k][l] = Cav2[i][j][k][l] +
                               acs_in[p][i]*acs_in[q][j]*acs_in[r][k]*acs_in[s][l]*C0[p][q][r][s];
            }
            }
            }
            }
          Cav[i][j][k][l] = Cav[i][j][k][l] + Cav2[i][j][k][l]*odf_in;
          }
          }
          }
          }
        }
      return Cav;
      }

      template <int dim>
      Tensor<2,6>
      GrainOrientations<dim>::
      reduce_81_36(Tensor<4,3> &Cav) const
      {
      std::array<double,6> l1 = {0,1,2,1,2,0};
      std::array<double,6> l2 = {0,1,2,2,0,1};
      Tensor<2,6> Sav;
      for (int i=0; i<6; ++i)
        {
      for (int j=0; j<6; ++j)
        {
        Sav[i][j] = Cav[l1[i]][l2[i]][l1[j]][l2[j]];
        }
        }

      return Sav;
      }

      template <int dim>
      UpdateTimeFlags
      GrainOrientations<dim>::need_update() const
      {
        return update_output_step;
      }

      template <int dim>
      UpdateFlags
      GrainOrientations<dim>::get_needed_update_flags () const
      {
        return update_values | update_gradients;
      }

      template <int dim>
      std::vector<std::pair<std::string, unsigned int> >
      GrainOrientations<dim>::get_property_information() const
      {
      std::vector<std::pair<std::string,unsigned int> > property_information (1,std::make_pair("n_grains",1));

	for (unsigned int i=0; i<3; ++i)
	  {
	  for (unsigned int j=0; j<3; ++j)
	    {
	    property_information.emplace_back("F"+std::to_string(i)+std::to_string(j),1);
	    }
	  }

	for (unsigned int i=0; i<n_grains; ++i)
	  {
	  for (unsigned int j=0; j<9; ++j)
	    {
	    property_information.emplace_back("acs_"+std::to_string(i)+'_'+std::to_string(j),1);
	    }
	  property_information.emplace_back("odf_"+std::to_string(i),1);
	  property_information.emplace_back("rt_"+std::to_string(i),1);
	  }

      for (int i=0; i<6; ++i)
        {
      for (int j=0; j<6; ++j)
        {
        property_information.emplace_back("Sav_"+std::to_string(i)+'_'+std::to_string(j),1);
        }
        }
	  
	return property_information;
      }

      template <int dim>
      void
      GrainOrientations<dim>::declare_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
	 {
          prm.enter_subsection("Particles");
	   {
	     prm.enter_subsection("Grain orientations");
	     {
	       prm.declare_entry("Number of grains","500",Patterns::Integer(1),
					"The number of grains in the aggregate carried on "
					"each particle. Larger numbers are best for useful "
					"statistics on fabric evolution.");
              prm.declare_entry("Grain boundary mobility","125",Patterns::Double(0),
					"Dimensionless grain boundary mobility parameter, "
					"M*. A value of 125 +/- 75 reproduces experimental "
					"results for olivine revolution pretty well "
					"(Kaminski and Ribe 2001, epsl).");
              prm.declare_entry("Nucleation parameter","5",Patterns::Double(0),
					"Dimensionless nucleation parameter, lambda*. "
					"A value of 5 is thought to be suitable for the "
					"upper mantle, but the exact value does not "
					"have a strong effect on LPO.");
              prm.declare_entry("Volume fraction","0.3",Patterns::Double(0,1),
					"Dimensionless volume fraction chi, defined as "
					"the ratio of the initial size of grains over the "
					"the size for which grain boundary sliding is "
					"the dominant deformation mechanism. For grains "
					"with a dimensionless volume smaller than chi, "
					"strain energy is set to 0 and the grains do not "
					"rotate by plastic deformation.");
              prm.declare_entry("Resolved shear stresses","1, 2, 3, 1e60",
					Patterns::List(Patterns::Double(0)),
					"List of rerference dimensionless resolved shear stresses for "
					"four olivine shear planes. The order is (010)[100], "
					"(001)[100], (010)[001], (100)[001].");
              prm.declare_entry("Stress exponent","3.5",Patterns::Double(0),
					"Stress exponent for power-law rheology");
            }
            prm.leave_subsection();
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }


      template <int dim>
      void
      GrainOrientations<dim>::parse_parameters (ParameterHandler &prm)
      {
	 prm.enter_subsection("Postprocess");
	 {
	   prm.enter_subsection("Particles");
	   {
	     prm.enter_subsection("Grain orientations");
	     {
              n_grains = prm.get_integer("Number of grains");
              mob = prm.get_double("Grain boundary mobility");
              lam = prm.get_double("Nucleation parameter");
              chi = prm.get_double("Volume fraction");
              tau = Utilities::string_to_double(Utilities::split_string_list(prm.get("Resolved shear stresses")));
              stress_exp = prm.get_double("Stress exponent");
            }
            prm.leave_subsection();
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      ASPECT_REGISTER_PARTICLE_PROPERTY(GrainOrientations,
                                        "grain orientations",
                                        "A plugin which implements the D-Rex model of "
                                        "Kaminski et al to track texture development "
					"in olivine polycrystals via plastic deformation "
					"and dynamic recrystallization.")
    }
  }
}

