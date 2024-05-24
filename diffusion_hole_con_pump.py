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
r1 = 0.2
c1_x = 7.8
c1_y = 7.4
r2 = 0.2
c2_x = 4.5
c2_y = 4.7
gdim = 2
mesh_comm = MPI.COMM_WORLD
model_rank = 0


#Physical Constants
D = 0.5
a = 1
b = 1
t = 0  # Start time
T = 10.0  # Final time
num_steps = 500
dt = T / num_steps  # Time step size


#Define the Petri dish and the cells
if mesh_comm.rank == model_rank:
	dish = gmsh.model.occ.addRectangle(0, 0, 0, L, W, tag=1)	
	cell1 = gmsh.model.occ.addDisk(c1_x, c1_y, 0, r1, r1)
	cell2 = gmsh.model.occ.addDisk(c2_x-2, c2_y-3, 0, r2, r2)
	#cell3 = gmsh.model.occ.addDisk(c3_x, c3_y, 0, r3, r3)


#Cut out for spatial exclusion model
if  mesh_comm.rank == model_rank:
	gmsh.model.occ.cut([(gdim,dish)],[(gdim,cell1)])
	gmsh.model.occ.cut([(gdim,dish)],[(gdim,cell2)])
	#gmsh.model.occ.cut([(gdim,dish)],[(gdim,cell3)])
	gmsh.model.occ.synchronize()


#Spatial exclusion mesh with more refined meshing around the cells
liquid_marker_s = 1
if mesh_comm.rank == model_rank:
	volume_s = gmsh.model.getEntities(dim=gdim)
	print(len(volume_s))
	assert(len(volume_s) == 1)
	gmsh.model.addPhysicalGroup(volume_s[0][0],[volume_s[0][1]],liquid_marker_s)
	gmsh.model.setPhysicalName(volume_s[0][0],liquid_marker_s,"Liquid_s")


wall_marker, cells_marker = 2, 3

wall, cells = [], []

if mesh_comm.rank == model_rank:
	boundaries = gmsh.model.getBoundary(volume_s, oriented = False)
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

domain_spatial_exclusion, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim = gdim)
V_s = functionspace(domain_spatial_exclusion, ("Lagrange", 1))


#Show mesh
print(pyvista.global_theme.jupyter_backend)

from dolfinx import plot

topology, cell_types, geometry = plot.vtk_mesh(V_s)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, color = [1.0,1.0,1.0], show_edges = True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("fundamentals_mesh.png")

if  mesh_comm.rank == model_rank:
	gmsh.model.occ.addRectangle(0, 0, 0, L, W, tag=2)
	gmsh.model.occ.synchronize()

if mesh_comm.rank == model_rank:
	gmsh.model.mesh.generate(gdim)
	gmsh.model.mesh.setOrder(1)
	gmsh.model.mesh.optimize("Netgen")


#Add the hole back
if  mesh_comm.rank == model_rank:
	cell1b = gmsh.model.occ.addDisk(c1_x, c1_y, 0, r1, r1)
	cell2b = gmsh.model.occ.addDisk(c2_x, c2_y, 0, r2, r2)
	#cell3b = gmsh.model.occ.addDisk(c3_x, c3_y, 0, r3, r3)
	gmsh.model.occ.fuse([(gdim,dish)],[(gdim,cell1b)])
	gmsh.model.occ.fuse([(gdim,dish)],[(gdim,cell2b)])
	#gmsh.model.occ.fuse([(gdim,dish)],[(gdim,cell3b)])
	gmsh.model.occ.synchronize()

liquid_marker_p = 4
if mesh_comm.rank == model_rank:
	volume_p = gmsh.model.getEntities(dim=gdim)
	print(len(volume_p))
	assert(len(volume_p) == 2)
	gmsh.model.addPhysicalGroup(volume_p[0][0],[volume_p[0][1]],liquid_marker_p)
	gmsh.model.setPhysicalName(volume_p[0][0],liquid_marker_p,"Liquid_p")

if mesh_comm.rank == model_rank:
	gmsh.model.mesh.generate(gdim)
	gmsh.model.mesh.setOrder(1)
	gmsh.model.mesh.optimize("Netgen")

domain_point_source, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim = gdim)
V_p = functionspace(domain_point_source, ("Lagrange", 1))

topology, cell_types, geometry = plot.vtk_mesh(V_p)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
plotter = pyvista.Plotter()

plotter.add_mesh(grid, color = [1.0,1.0,1.0], show_edges = True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("fundamentals_mesh.png")


#Initial condition
def initial_condition(x, disp=1,cct=2):
    return np.exp(-disp*((x[0]-7)**2+(x[1]-7)**2)+cct)


"""
#Boundary markers
fdim = domain_spatial_exclusion.topology.dim - 1
#boundary_facets = mesh.locate_entities_boundary(
#    domain_spatial_exclusion, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
#bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

boundary_locator = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.isclose(x[0], L)),
              (3, lambda x: np.isclose(x[1], 0)),
              (4, lambda x: np.isclose(x[1], W)),
              (5, lambda x: np.isclose((x[0]-c1_x)**2+(x[1]-c1_y)**2,r1**2)),
              (6, lambda x: np.isclose((x[0]-c1_x+2)**2+(x[1]-c2_y+3)**2,r2**2))]

facet_indices, facet_markers = [], []
for (marker, locator) in boundary_locator:
    facets = mesh.locate_entities(domain_spatial_exclusion, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = mesh.meshtags(domain_spatial_exclusion, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

ds = Measure("ds", domain=domain_spatial_exclusion, subdomain_data=facet_tag)


#Spatial exclusion model computation
u_s = TrialFunction(V_s)
v_s = TestFunction(V_s)
u_sn = Function(V_s)
u_sn.interpolate(initial_condition)

a_s = u_s * v_s * dx + dt * D * dot(grad(u_s), grad(v_s)) * dx + dt * a * u_s * v_s * ds(5) + dt * a * u_s * v_s * ds(6)
L_s = u_sn * v_s * dx + dt * b * v_s * ds(5) + dt * b * v_s * ds(6)

from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
from petsc4py import PETSc

bilinearform_s = form(a_s)
linearform_s = form(L_s)

A_s = assemble_matrix(bilinearform_s)
A_s.assemble()
b_s = create_vector(linearform_s)

solver = PETSc.KSP().create(domain_spatial_exclusion.comm)
solver.setOperators(A_s)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

u_s = Function(V_s)

grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V_s))

plotter = pyvista.Plotter()
plotter.open_gif("spatial_exclusion.gif", fps=10)

grid.point_data["u_s"] = u_s.x.array
warped = grid.warp_by_scalar("u_s", factor=1)

viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
             position_x=0.1, position_y=0.8, width=0.8, height=0.1)

renderer = plotter.add_mesh(grid, show_edges=True, lighting=False,
                            cmap=viridis, scalar_bar_args=sargs,
                            clim=[0, max(u_s.x.array)])

xdmf = io.XDMFFile(domain_spatial_exclusion.comm, "spatial_exclusion.xdmf", "w")
xdmf.write_mesh(domain_spatial_exclusion)

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
"""

#Point source computation
fdim = domain_spatial_exclusion.topology.dim - 1
#boundary_facets = mesh.locate_entities_boundary(
#    domain_spatial_exclusion, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
#bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

u_p = TrialFunction(V_p)
v_p = TestFunction(V_p)
u_pn = Function(V_p)
u_pn.interpolate(initial_condition)

def entire_dish_domain(x):
	mask = np.full(x.shape[0], True, dtype=bool)
	return mask

def cell1_subdomain(x):
	mask = (x[0]-c1_x)**2 + (x[0]-c1_y)**2 <= r1**2 
	return mask

def cell2_subdomain(x):
	mask = (x[0]-c2_x)**2 + (x[0]-c2_y)**2 <= r2**2 
	return mask

subdomain_locator = [(1, entire_dish_domain),
		(2, cell1_subdomain),
              (3, cell2_subdomain)]


facet_indices, facet_markers = [], []
for (marker, locator) in subdomain_locator:
    facets = mesh.locate_entities(domain_spatial_exclusion, gdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = mesh.meshtags(domain_spatial_exclusion, gdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

dx = Measure("dx", domain=domain_point_source, subdomain_data=facet_tag)
