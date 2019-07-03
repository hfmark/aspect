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
       * A class that gets the velocity gradient tensor on a particle
       * as it moves along a streamline
       *
       * @ingroup ParticleProperties
       */
      template <int dim>
      class VelGrad : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
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
                                        const Point<dim> &position,
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
      VelGrad<dim>::initialize_one_particle_property(const Point<dim> &position,
                                                    std::vector<double> &data) const
      {
      for (unsigned int i=0; i<dim; ++i)
        {
        data.push_back(position[i]);
        }

	// lij (local velocity gradient tensor) is 3D even in 2D flow because we're
	// eventually using this to play with 3D aggregates

	for (unsigned int i=0; i<3; ++i)
	  {
	  for (unsigned int j=0; j<3; ++j)
	    {
	    data.push_back(0);  // initialize all entries to 0
	    }
	  }
      }

      template <int dim>
      void
      VelGrad<dim>::update_one_particle_property(const unsigned int data_position,
                                                const Point<dim> &position,
                                                const Vector<double> &,
                                                const std::vector<Tensor<1,dim> > &gradients,
                                                const ArrayView<double> &data) const
      {
	// we don't care what's currently in there; we just want the new lij
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

	int pos = data_position;

	for (unsigned int i=0; i<dim; ++i)
         {
         data[pos] = position[i];
         pos = pos + 1;
         }

	for (unsigned int i=0; i<3; ++i)
	  {
	  for (unsigned int j=0; j<3; ++j)
	    {
           data[pos] = lij[i][j];
	    pos = pos + 1;
	    }
	  }
      }

      template <int dim>
      UpdateTimeFlags
      VelGrad<dim>::need_update() const
      {
        return update_output_step;
      }

      template <int dim>
      UpdateFlags
      VelGrad<dim>::get_needed_update_flags () const
      {
        return update_values | update_gradients;
      }

      template <int dim>
      std::vector<std::pair<std::string, unsigned int> >
      VelGrad<dim>::get_property_information() const
      {

       std::vector<std::pair<std::string,unsigned int> > property_information (1,std::make_pair("position",dim));

	for (unsigned int i=0; i<3; ++i)
	  {
	  for (unsigned int j=0; j<3; ++j)
	    {
	    property_information.emplace_back("l"+std::to_string(i)+std::to_string(j),1);
	    }
	  }

	return property_information;
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
      ASPECT_REGISTER_PARTICLE_PROPERTY(VelGrad,
                                        "velocity gradients",
                                        "A plugin which tracks the local velocity "
                                        "gradient tensor on particles")
    }
  }
}



