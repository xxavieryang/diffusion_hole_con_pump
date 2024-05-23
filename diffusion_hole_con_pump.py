from mpi4py import MPI
import numpy as np
import matplotlib as mpl
import gmsh
from dolfinx import fem, mesh, io, plot
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Function
from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system)
import pyvista


#Create the computation domain and geometric constant
gmsh.initialize()

L = W = 10
r = 0.2
c_x = 7.8
c_y = 7.4
gdim = 2
mesh_comm = MPI.COMM_WORLD
model_rank = 0


#Physical Constants
D = 1
a = 1
b = 1
t = 0  # Start time
T = 10.0  # Final time
num_steps = 500
dt = T / num_steps  # Time step size


#Define the Petri dish and the cells
if mesh_comm.rank == model_rank:
	dish = gmsh.model.occ.addRectangle(0, 0, 0, L, W, tag=1)
	cell1 = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
	#cell2 = gmsh.model.occ.addDisk(c_x+2, c_y+3, 0, r, r)
	#cell3 = gmsh.model.occ.addDisk(c_x+6, c_y+6, 0, r, r)


#Cut out for spatial exclusion model
if  mesh_comm.rank == model_rank:
	spatial_exclusion_domain = gmsh.model.occ.cut([(gdim,dish)],[(gdim,cell1)])
	#spatial_exclusion_domain = gmsh.model.occ.cut([(gdim,dish)],[(gdim,cell2)])
	#spatial_exclusion_domain = gmsh.model.occ.cut([(gdim,dish)],[(gdim,cell3)])
	gmsh.model.occ.synchronize()

#Spatial exclusion mesh with more refined meshing around the cells
liquid_marker = 1
if mesh_comm.rank == model_rank:
	volumes = gmsh.model.getEntities(dim=gdim)
	print(len(volumes))
	assert(len(volumes) == 1)
	gmsh.model.addPhysicalGroup(volumes[0][0],[volumes[0][1]],liquid_marker)
	gmsh.model.setPhysicalName(volumes[0][0],liquid_marker,"Liquid")

wall_marker, cells_marker = 2, 3

wall, cells = [], []

if mesh_comm.rank == model_rank:
	boundaries = gmsh.model.getBoundary(volumes, oriented = False)
	for boundary in boundaries:
		center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0],boundary[1])
		if np.allclose(center_of_mass,[L/2,0,0]) or np.allclose(center_of_mass,[0,W/2,0]) or \
		np.allclose(center_of_mass,[L,W/2,0]) or np.allclose(center_of_mass,[L/2,W,0]):
			wall.append(boundary[1])
		else:
			cells.append(boundary[1])
	gmsh.model.addPhysicalGroup(1,wall,wall_marker)
	gmsh.model.setPhysicalName(1,wall_marker,"Wall")
	gmsh.model.addPhysicalGroup(1,cells,cells_marker)
	gmsh.model.setPhysicalName(1,cells_marker,"Cell")

res_min = r / 3
if mesh_comm.rank == model_rank:
	distance_field = gmsh.model.mesh.field.add("Distance")
	gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", cells)
	threshold_field = gmsh.model.mesh.field.add("Threshold")
	gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
	gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
	gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * L)
	gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
	gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * L)
	min_field = gmsh.model.mesh.field.add("Min")
	gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
	gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

if mesh_comm.rank == model_rank:
	gmsh.model.mesh.generate(gdim)
	gmsh.model.mesh.setOrder(1)
	gmsh.model.mesh.optimize("Netgen")

domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim = gdim)
V = functionspace(domain, ("Lagrange", 1))


#Show mesh
#print(pyvista.global_theme.jupyter_backend)

#from dolfinx import plot

#topology, cell_types, geometry = plot.vtk_mesh(V)
#grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
#plotter = pyvista.Plotter()
#plotter.add_mesh(grid, color = [1.0,1.0,1.0], show_edges = True)
#plotter.view_xy()
#if not pyvista.OFF_SCREEN:
#    plotter.show()
#else:
#    figure = plotter.screenshot("fundamentals_mesh.png")


#Initial condition
def initial_condition(x, disp=1,cct=2):
    return 0*x[0]#np.exp(-disp*((x[0]-7)**2+(x[1]-7)**2)+cct)


#Boundary markers
fdim = domain.topology.dim - 1
#boundary_facets = mesh.locate_entities_boundary(
#    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
#bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

boundary_locator = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.isclose(x[0], L)),
              (3, lambda x: np.isclose(x[1], 0)),
              (4, lambda x: np.isclose(x[1], W)),
              (5, lambda x: np.isclose((x[0]-c_x)**2+(x[1]-c_y)**2,r**2))]

facet_indices, facet_markers = [], []
for (marker, locator) in boundary_locator:
    facets = mesh.locate_entities(domain, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

ds = Measure("ds", domain=domain, subdomain_data=facet_tag)


#Spatial exclusion model computation
u_s = TrialFunction(V)
v_s = TestFunction(V)
u_sn = Function(V)
u_sn.interpolate(initial_condition)

a_s = u_s * v_s * dx + dt * D * dot(grad(u_s), grad(v_s)) * dx + dt * a * u_s * v_s * ds(5)
L_s = u_sn * v_s * dx + dt * b * v_s * ds(5)

from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
from petsc4py import PETSc

bilinearform_s = form(a_s)
linearform_s = form(L_s)

A_s = assemble_matrix(bilinearform_s)
A_s.assemble()
b_s = create_vector(linearform_s)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A_s)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

u_s = Function(V)

grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

plotter = pyvista.Plotter()
plotter.open_gif("spatial_exclusion.gif", fps=10)

grid.point_data["u_s"] = u_s.x.array
warped = grid.warp_by_scalar("u_s", factor=1)

viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
             position_x=0.1, position_y=0.8, width=0.8, height=0.1)

renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
                            cmap=viridis, scalar_bar_args=sargs,
                            clim=[0, max(u_s.x.array)])

xdmf = io.XDMFFile(domain.comm, "spatial_exclusion.xdmf", "w")
xdmf.write_mesh(domain)

for i in range(num_steps):
	t += dt
	with b_s.localForm() as loc_b:
		loc_b.set(0)
	assemble_vector(b_s, linearform_s)
    # Solve linear problem
	solver.solve(b_s, u_s.vector)
	u_s.x.scatter_forward()

	u_sn.x.array[:] = u_s.x.array
	xdmf.write_function(u_s, t)

	new_warped = grid.warp_by_scalar("u_s", factor=1)
	warped.points[:, :] = new_warped.points
	warped.point_data["u_s"][:] = u_s.x.array
	plotter.write_frame()
plotter.close()
xdmf.close()