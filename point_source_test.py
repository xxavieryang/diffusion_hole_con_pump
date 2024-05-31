import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner
from mpi4py import MPI
import dolfinx.fem.petsc
from dolfinx.fem import functionspace, form, Function
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc 
import dolfinx.plot
from petsc4py.PETSc import ScalarType

msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (10.0, 10.0)), n=(100, 100),
                            cell_type=mesh.CellType.triangle,)
V = fem.FunctionSpace(msh, ("Lagrange", 1))

#facets = mesh.locate_entities_boundary(msh, dim=1, marker=lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 10.0)),
                                                                                  #np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 10.0))))
#dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
#bc = fem.dirichletbc(value=ScalarType(6), dofs=dofs, V=V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)

f = fem.Function(V)
dofs = fem.locate_dofs_geometrical(V,  lambda x: np.isclose(x.T, [2.5, 5, 0]).all(axis=1))
f.x.array[dofs] = 1
dofs = fem.locate_dofs_geometrical(V,  lambda x: np.isclose(x.T, [7.5, 5, 0]).all(axis=1))
f.x.array[dofs] = -1

a = inner(grad(u), grad(v)) * dx
L = f * v * dx
l = create_vector(form(L))
l.assemble()


problem = fem.petsc.LinearProblem(a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

try:
    import pyvista
    #pyvista.start_xvfb()
    cells, types, x = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=False)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)
    plotter.show(cpos='xy')
except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution")
    print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")