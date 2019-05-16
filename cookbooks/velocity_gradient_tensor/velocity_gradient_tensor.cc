/*
  Copyright (C) 2011 - 2018 by the authors of the ASPECT code.

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


#include <aspect/postprocess/visualization.h>
#include <aspect/simulator_access.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_postprocessor.h>

/* NOTE that dealii 9.0.0 or higher is needed for DataPostprocessorTensor (I think) */

namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      /**
       * A class derived from DataPostprocessor that takes an output vector
       * and computes a variable that represents the 3 or 6 independent
       * components (in 2d and 3d, respectively) of the ****** at
       * every point.
       *
       * The member functions are all implementations of those declared in the
       * base class. See there for their meaning.
       */
      template <int dim>
      class VelocityGradientTensor
        : public DataPostprocessorTensor<dim>,
          public SimulatorAccess<dim>,
          public Interface<dim>
      {
        public:
	  VelocityGradientTensor ();

          virtual
          void
          evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                                std::vector<Vector<double> > &computed_quantities) const;

      };
    }
  }
}



namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      template <int dim>
      VelocityGradientTensor<dim>::
      VelocityGradientTensor ()
        :
        DataPostprocessorTensor<dim> ("velocity_gradient_tensor",
				      update_gradients | update_q_points | update_values)
      {}

      template <int dim>
      void
      VelocityGradientTensor<dim>::
      evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                            std::vector<Vector<double> > &computed_quantities) const
      {
        const unsigned int n_quadrature_points = input_data.solution_values.size();
        Assert (computed_quantities.size() == n_quadrature_points, ExcInternalError());
        Assert ((computed_quantities[0].size() == Tensor<2,dim>::n_independent_components),ExcInternalError());
	Assert (input_data.solution_gradients[0].size() == this->introspection().n_components, ExcInternalError());

        for (unsigned int q=0; q<n_quadrature_points; ++q)
          {
            Tensor<2,dim> grad_v;
       	    for (unsigned int d=0; d<dim; ++d)
              {
              grad_v[d] = input_data.solution_gradients[q][this->introspection().component_indices.velocities[d]];  // matching rank-1 tensors, basically
              }

            for (unsigned int i=0; i<Tensor<2,dim>::n_independent_components; ++i)
              {
              computed_quantities[q](i) = grad_v[grad_v.unrolled_to_component_indices(i)];
              }
          }
      }

    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      ASPECT_REGISTER_VISUALIZATION_POSTPROCESSOR(VelocityGradientTensor,
                                                  "velocity gradient tensor",
                                                  "A visualization output object that generates output "
                                                  "for the 4 (in 2d) or 9 (in 3d) components of the "
                                                  "velocity gradient tensor.")
    }
  }
}
