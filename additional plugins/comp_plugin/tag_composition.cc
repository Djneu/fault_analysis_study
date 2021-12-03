/*
  Copyright (C) 2011 - 2017 by the authors of the ASPECT code.

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
  along with ASPECT; see the file doc/COPYING.  If not see
  <http://www.gnu.org/licenses/>.
*/


#include "tag_composition.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

namespace aspect
{
  namespace MeshRefinement
  {
    template <int dim>
    void
    TagComposition<dim>::tag_additional_cells() const
    {
      if (this->get_dof_handler().n_dofs() != 0)
        {

          AssertThrow (this->n_compositional_fields() >= 1,
                       ExcMessage ("This refinement criterion can not be used when no "
                                   "compositional fields are active!"));

          QGauss<dim> quadrature (this->get_fe().base_element(this->introspection().base_elements.compositional_fields).degree+1);

          FEValues<dim> fe_values (this->get_mapping(),
                                   this->get_fe(),
                                   quadrature,
                                   update_quadrature_points | update_values);

          // the values of the compositional fields are stored as blockvectors for each field
          // we have to extract them in this structure
          std::vector<std::vector<double> > prelim_composition_values
          (this->n_compositional_fields(),
           std::vector<double> (quadrature.size()));

          typename DoFHandler<dim>::active_cell_iterator
          cell = this->get_dof_handler().begin_active(),
          endc = this->get_dof_handler().end();
          for (; cell!=endc; ++cell)
            if (cell->is_locally_owned())
              {
                bool coarsen = false;
                bool refine = false;
                bool clear_refine = false;
                bool clear_coarsen = false;

                bool sed1_present = false;
                bool sed2_present = false;
                bool uc_present = false;
                bool lc_present = false;
                bool ml_present = false;

                fe_values.reinit(cell);

                for (unsigned int c = 0; c<this->n_compositional_fields(); c++)
                  {
                    fe_values[this->introspection().extractors.compositional_fields[c]].get_function_values (this->get_solution(),
                        prelim_composition_values[c]);
                  }


                for (unsigned int p=0; p<quadrature.size(); ++p)
                  {
                    if (prelim_composition_values[sed1_refinement[0]][p] > 0.2)
                      {
                        sed1_present = true;
                        //Crust will have smallest res, so not interested in other fields
                        break;
                      }
                    if (prelim_composition_values[sed2_refinement[0]][p] > 0.2)
                        sed2_present = true;
                    if (prelim_composition_values[uc_refinement[0]][p] > 0.2)
                        uc_present = true;
                    if (prelim_composition_values[lc_refinement[0]][p] > 0.2)
                        lc_present = true;
                    if (prelim_composition_values[ml_refinement[0]][p] > 0.2)
                        ml_present = true;
                  }


                //Only continue if at least one is true

                    int maximum_refinement_level = max_level;
                    int minimum_refinement_level = min_level;

                    if (sed1_present)
                      {
                        minimum_refinement_level = sed1_refinement[1];
                        maximum_refinement_level = sed1_refinement[2];
                      }
                    else if (sed2_present)
                      {
                        minimum_refinement_level = sed2_refinement[1];
                        maximum_refinement_level = sed2_refinement[2];
                      }
                    else if (uc_present)
                      {
                        minimum_refinement_level = uc_refinement[1];
                        maximum_refinement_level = uc_refinement[2];
                      }
                    else if (lc_present)
                      {
                        minimum_refinement_level = lc_refinement[1];
                        maximum_refinement_level = lc_refinement[2];
                      }
                    else if (ml_present)
                      {
                        minimum_refinement_level = ml_refinement[1];
                        maximum_refinement_level = ml_refinement[2];
                      }

                    const int cell_level = cell->level();
                    if (cell_level >= maximum_refinement_level)
                      {
                        clear_refine = true;
                      }
                    if (cell_level >  maximum_refinement_level)
                      {
                        coarsen = true;
                      }
                    if (cell_level <= minimum_refinement_level)
                      {
                        clear_coarsen = true;
                      }
                    if (cell_level < minimum_refinement_level)
                      {
                        refine = true;
                      }

                    if (clear_refine)
                      cell->clear_refine_flag ();
                    if (clear_coarsen)
                      cell->clear_coarsen_flag ();
                    if (refine)
                      cell->set_refine_flag ();
                    if (coarsen)
                      cell->set_coarsen_flag ();

              }
        }
    }

    template <int dim>
    void
    TagComposition<dim>::
    declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Mesh refinement");
      {
        prm.enter_subsection("Composition");
        {
          prm.declare_entry("Sediment 1 refinement","",
                            Patterns::List (Patterns::Integer(0)),
                            "The compositional field number of the crust, its minimum refinement level and "
                            "its maximum refinement level.");
          prm.declare_entry("Sediment 2 refinement","",
                            Patterns::List (Patterns::Integer(0)),
                            "The compositional field number of the slab mantle, its minimum refinement level and "
                            "its maximum refinement level.");
          prm.declare_entry("Upper crust refinement","",
                            Patterns::List (Patterns::Integer(0)),
                            "The compositional field number of the slab mantle, its minimum refinement level and "
                            "its maximum refinement level.");
          prm.declare_entry("Lower crust refinement","",
                            Patterns::List (Patterns::Integer(0)),
                            "The compositional field number of the slab mantle, its minimum refinement level and "
                            "its maximum refinement level.");
          prm.declare_entry("Mantle lithosphere refinement","",
                            Patterns::List (Patterns::Integer(0)),
                            "The compositional field number of the slab mantle, its minimum refinement level and "
                            "its maximum refinement level.");

        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    TagComposition<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Mesh refinement");
      {
        min_level = prm.get_integer("Minimum refinement level");
        max_level = prm.get_integer("Initial adaptive refinement") + prm.get_integer("Initial global refinement");
        prm.enter_subsection("Composition");
        {

          const std::vector<int> sed1
            = Utilities::string_to_int(
                Utilities::split_string_list(prm.get("Sediment 1 refinement")));

          sed1_refinement = std::vector<unsigned int> (sed1.begin(),sed1.end());

          AssertThrow (sed1_refinement.size() == 3,
                       ExcMessage ("The number of refinement data given here must be "
                                   "equal to 3 (field number + min level + max level). "));

          AssertThrow (sed1_refinement[0] < this->n_compositional_fields(),
                       ExcMessage ("The number of compositional field to refine (starting "
                                   "from 0) should be smaller than the number of fields. "));

          AssertThrow (sed1_refinement[1] >= min_level,
                       ExcMessage ("The minimum refinement for the crust cannot be "
                                   "smaller than the minimum level of the whole model. "));

          AssertThrow (sed1_refinement[2] <= max_level,
                       ExcMessage ("The maximum refinement for the crust cannot be "
                                   "greater than the maximum level of the whole model. "));

          const std::vector<int> sed2
            = Utilities::string_to_int(
                Utilities::split_string_list(prm.get("Sediment 2 refinement")));

          sed2_refinement = std::vector<unsigned int> (sed2.begin(),
                                                              sed2.end());

          AssertThrow (sed2_refinement.size() == 3,
                       ExcMessage ("The number of refinement data given here must be "
                                   "equal to 3 (field number + min level + max level). "));

          AssertThrow (sed2_refinement[0] < this->n_compositional_fields(),
                       ExcMessage ("The number of compositional field to refine (starting "
                                   "from 0) should be smaller than the number of fields. "));

          AssertThrow (sed2_refinement[1] >= min_level,
                       ExcMessage ("The minimum refinement for the slab mantle cannot be "
                                   "smaller than the minimum level of the whole model. "));

          AssertThrow (sed2_refinement[2] <= max_level,
                       ExcMessage ("The maximum refinement for the slab mantle cannot be "
                                   "greater than the maximum level of the whole model. "));
        

          const std::vector<int> uc
            = Utilities::string_to_int(
                Utilities::split_string_list(prm.get("Upper crust refinement")));

          uc_refinement = std::vector<unsigned int> (uc.begin(),
                                                              uc.end());

          AssertThrow (uc_refinement.size() == 3,
                       ExcMessage ("The number of refinement data given here must be "
                                   "equal to 3 (field number + min level + max level). "));

          AssertThrow (uc_refinement[0] < this->n_compositional_fields(),
                       ExcMessage ("The number of compositional field to refine (starting "
                                   "from 0) should be smaller than the number of fields. "));

          AssertThrow (uc_refinement[1] >= min_level,
                       ExcMessage ("The minimum refinement for the slab mantle cannot be "
                                   "smaller than the minimum level of the whole model. "));

          AssertThrow (uc_refinement[2] <= max_level,
                       ExcMessage ("The maximum refinement for the slab mantle cannot be "
                                   "greater than the maximum level of the whole model. "));
        


          const std::vector<int> lc
            = Utilities::string_to_int(
                Utilities::split_string_list(prm.get("Lower crust refinement")));

          lc_refinement = std::vector<unsigned int> (lc.begin(),
                                                              lc.end());

          AssertThrow (lc_refinement.size() == 3,
                       ExcMessage ("The number of refinement data given here must be "
                                   "equal to 3 (field number + min level + max level). "));

          AssertThrow (lc_refinement[0] < this->n_compositional_fields(),
                       ExcMessage ("The number of compositional field to refine (starting "
                                   "from 0) should be smaller than the number of fields. "));

          AssertThrow (lc_refinement[1] >= min_level,
                       ExcMessage ("The minimum refinement for the slab mantle cannot be "
                                   "smaller than the minimum level of the whole model. "));

          AssertThrow (lc_refinement[2] <= max_level,
                       ExcMessage ("The maximum refinement for the slab mantle cannot be "
                                   "greater than the maximum level of the whole model. "));
        


          const std::vector<int> ml
            = Utilities::string_to_int(
                Utilities::split_string_list(prm.get("Mantle lithosphere refinement")));

          ml_refinement = std::vector<unsigned int> (ml.begin(),
                                                              ml.end());

          AssertThrow (ml_refinement.size() == 3,
                       ExcMessage ("The number of refinement data given here must be "
                                   "equal to 3 (field number + min level + max level). "));

          AssertThrow (ml_refinement[0] < this->n_compositional_fields(),
                       ExcMessage ("The number of compositional field to refine (starting "
                                   "from 0) should be smaller than the number of fields. "));

          AssertThrow (ml_refinement[1] >= min_level,
                       ExcMessage ("The minimum refinement for the slab mantle cannot be "
                                   "smaller than the minimum level of the whole model. "));

          AssertThrow (ml_refinement[2] <= max_level,
                       ExcMessage ("The maximum refinement for the slab mantle cannot be "
                                   "greater than the maximum level of the whole model. "));
        }

        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MeshRefinement
  {
    ASPECT_REGISTER_MESH_REFINEMENT_CRITERION(TagComposition,
                                              "tag composition",
                                              "A mesh refinement criterion that "
                                              "(de)flags cells for refinement and coarsening "
                                              "based on what composition is present. Different max "
                                              "and min refinement levels are set for the mantle, the crustal "
                                              "field, the slab mantle and the overriding plate. ")
  }
}