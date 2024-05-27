# Create a point source for Poisson problem
# Author: Jørgen S. Dokken
# SPDX-License-Identifier: MIT

from mpi4py import MPI
from petsc4py import PETSc
import os

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
import pyvista

import point_source as ps

N = 80
domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
domain.name = "mesh"
domain.topology.create_connectivity(1, 2)

V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))

facets = dolfinx.mesh.exterior_facet_indices(domain.topology)

if domain.comm.rank == 0:
    points = np.array([[0.68, 0.36, 0]], dtype=domain.geometry.x.dtype)
    points1 = np.array([[0.3, 0.66, 0]], dtype=domain.geometry.x.dtype)
else:
    points = np.zeros((0, 3), dtype=domain.geometry.x.dtype)


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
a_compiled = dolfinx.fem.form(a)


dofs = dolfinx.fem.locate_dofs_topological(V, 1, facets)
u_bc = dolfinx.fem.Constant(domain, 0.)
bc = dolfinx.fem.dirichletbc(u_bc, dofs, V)

b = dolfinx.fem.Function(V)
b.x.array[:] = 0
cells, basis_values = ps.compute_cell_contributions(V, points)
for cell, basis_value in zip(cells, basis_values):
    dofs = V.dofmap.cell_dofs(cell)
    b.x.array[dofs] += basis_value
cells, basis_values = ps.compute_cell_contributions(V, points1)
for cell, basis_value in zip(cells, basis_values):
    dofs = V.dofmap.cell_dofs(cell)
    b.x.array[dofs] += basis_value
dolfinx.fem.petsc.apply_lifting(b.vector, [a_compiled], [[bc]])
b.x.scatter_reverse(dolfinx.la.InsertMode.add)
dolfinx.fem.petsc.set_bc(b.vector, [bc])
b.x.scatter_forward()

A = dolfinx.fem.petsc.assemble_matrix(a_compiled, bcs=[bc])
A.assemble()

ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.PREONLY)
ksp.getPC().setType(PETSc.PC.Type.LU)
ksp.getPC().setFactorSolverType("mumps")


uh = dolfinx.fem.Function(V)
ksp.solve(b.vector, uh.vector)
uh.x.scatter_forward()

try:
    import pyvista

    cells, types, x = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(grid)
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)
        plotter.screenshot("uh_poisson.png")
    else:
        plotter.show()
except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution")
    print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")