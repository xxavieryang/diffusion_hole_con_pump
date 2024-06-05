from mpi4py import MPI
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import gmsh
from dolfinx import fem, mesh, io, plot, geometry
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Function, assemble_scalar
from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction,
	             div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system)
import pyvista

import point_source as ps
#import point_source as ps


#Create the computation domain and geometric constant
gmsh.initialize()

L = W = 10
r1 = 0.2
c1_x = 5
c1_y = 5
r2 = 0.2
c2_x = 6.2
c2_y = 6.7
gdim = 2
fdim = 1
mesh_comm = MPI.COMM_WORLD
model_rank = 0


#Physical Constants
D = 10
a = 1
b = 1
t = 0  # Start time
T = 10000.0  # Final time
num_steps = 5000
dt = T / num_steps  # Time step size=

#Define the Petri dish and the cells
if mesh_comm.rank == model_rank:
	dish = gmsh.model.occ.addRectangle(0, 0, 0, L, W, tag=1)	
	cell1 = gmsh.model.occ.addDisk(c1_x, c1_y, 0, r1, r1)
	#cell2 = gmsh.model.occ.addDisk(c2_x, c2_y, 0, r2, r2)
	#cell3 = gmsh.model.occ.addDisk(c3_x, c3_y, 0, r3, r3)


#Cut out for spatial exclusion model
if  mesh_comm.rank == model_rank:
	gmsh.model.occ.cut([(gdim,dish)],[(gdim,cell1)])
	#gmsh.model.occ.cut([(gdim,dish)],[(gdim,cell2)])
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

res_min = r1 / 3
if mesh_comm.rank == model_rank:
	distance_field = gmsh.model.mesh.field.add("Distance")
	gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", cells)
	threshold_field = gmsh.model.mesh.field.add("Threshold")
	gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
	gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
	gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * L)
	gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r1)
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

#topology, cell_types, geometry = plot.vtk_mesh(V_s)
#grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
#plotter = pyvista.Plotter()
#plotter.add_mesh(grid, color = [1.0,1.0,1.0], show_edges = True)
#plotter.view_xy()
#if not pyvista.OFF_SCREEN:
    #plotter.show()
    #figure = plotter.screenshot("spatial_exclusion_mesh.png")
#else:
    #figure = plotter.screenshot("patial_exclusion_mesh.png")

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
	#cell2b = gmsh.model.occ.addDisk(c2_x, c2_y, 0, r2, r2)
	#cell3b = gmsh.model.occ.addDisk(c3_x, c3_y, 0, r3, r3)
	gmsh.model.occ.fuse([(gdim,dish)],[(gdim,cell1b)])
	#gmsh.model.occ.fuse([(gdim,dish)],[(gdim,cell2b)])
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

#topology, cell_types, geometry = plot.vtk_mesh(V_p)
#grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
#plotter = pyvista.Plotter()

#plotter.add_mesh(grid, color = [1.0,1.0,1.0], show_edges = True)
#plotter.view_xy()
#if not pyvista.OFF_SCREEN:
    #plotter.show()
    #plotter.screenshot("point_source_mesh.png")
#else:
    #figure = plotter.screenshot("point_source_mesh.png")


#Initial condition
def initial_condition(x, disp=1, cct=0.5):
    return x[0] * 0 + 1

#Spatial exclusion model

#Boundary markers
fdim = domain_spatial_exclusion.topology.dim - 1
#boundary_facets = mesh.locate_entities_boundary(
#    domain_spatial_exclusion, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
#bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

dx_s = Measure("dx", domain=domain_spatial_exclusion)

boundary_locator = [(0, lambda x: np.isclose(x[0], 0) | np.isclose(x[0], L) | np.isclose(x[1], 0) | np.isclose(x[1], W)),
              (1, lambda x: np.isclose((x[0]-c1_x)**2+(x[1]-c1_y)**2,r1**2)),
              (2, lambda x: np.isclose((x[0]-c2_x)**2+(x[1]-c2_y)**2,r2**2))]

facet_indices, facet_markers = [], []
for (marker, locator) in boundary_locator:
    facets = mesh.locate_entities(domain_spatial_exclusion, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = mesh.meshtags(domain_spatial_exclusion, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])


ds_s = Measure("ds", domain=domain_spatial_exclusion, subdomain_data=facet_tag)


u_s = TrialFunction(V_s)
v_s = TestFunction(V_s)
u_sn = Function(V_s)
u_sn.interpolate(initial_condition)

a_s = u_s * v_s * dx_s + dt * D * dot(grad(u_s), grad(v_s)) * dx_s + dt * a * u_s * v_s * ds_s(1) #+ dt * a * u_s * v_s * ds_s(2)
L_s = u_sn * v_s * dx_s + dt * b * v_s * ds_s(1) #+ dt * b * v_s * ds_s(2)

from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
from petsc4py import PETSc

bilinearform_s = form(a_s)
linearform_s = form(L_s)

A_s = assemble_matrix(bilinearform_s)
A_s.assemble()
b_s = create_vector(linearform_s)


solver_s = PETSc.KSP().create(domain_spatial_exclusion.comm)
solver_s.setOperators(A_s)
solver_s.setType(PETSc.KSP.Type.PREONLY)
solver_s.getPC().setType(PETSc.PC.Type.LU)


"""
grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V_s))
plotter = pyvista.Plotter()
plotter.open_gif("spactial_exclusion_try.gif", fps=10)

grid.point_data["u_s"] = u_s.x.array
#warped = grid.warp_by_scalar("u_s", factor=1)
viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
             position_x=0.1, position_y=0.8, width=0.8, height=0.1)

renderer = plotter.add_mesh(grid, show_edges=True, lighting=False,
                            cmap=viridis, scalar_bar_args=sargs,
                            clim=[0, 10])

#xdmf = io.XDMFFile(domain_spatial_exclusion.comm, "spatial_exclusion.xdmf", "w")
#xdmf.write_mesh(domain_spatial_exclusion)

for i in range(num_steps):
    t += dt
    with b_s.localForm() as loc_b:
    	loc_b.set(0)
    assemble_vector(b_s, linearform_s)
    # Solve linear problem
    solver.solve(b_s, u_s.vector)
    u_s.x.scatter_forward()
    u_sn.x.array[:] = u_s.x.array
    #xdmf.write_function(u_s, t)
    new_warped = grid.warp_by_scalar("u_s", factor=1)
    warped.points[:, :] = new_warped.points
    warped.point_data["u_s"][:] = u_s.x.array
    plotter.write_frame()
#os.remove("spactial_exclusion_try.gif")
#plotter.close()
#xdmf.close()
"""

#Point source model
#fdimp = domain_point_source.topology.dim - 1
#boundary_facets = mesh.locate_entities_boundary(
#    domain_spatial_exclusion, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
#bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

#Marking boundaries and subdomains


def diracd1(x):
    return np.exp(-((x[0]-c1_x)**2+(x[1]-c1_y)**2)/0.002)/(0.002*np.pi)

def xi(x):
    return (x[0]-c1_x)**2+(x[1]-c1_y)**2<=r1**2

def xx(x):
    return (x[0]-c1_x)**2/2

def yy(x):
    return (x[1]-c1_y)**2/2

#boundary_locator = [(0, lambda x: np.isclose(x[0],0) | np.isclose(x[0],L) | np.isclose(x[1],0) | np.isclose(x[1],W)),
#              (1, lambda x: np.isclose((x[0]-c1_x)**2+(x[1]-c1_y)**2,r1**2)),
#              (2, lambda x: np.isclose((x[0]-c2_x)**2+(x[1]-c2_y)**2,r2**2))]

#facet_indices, facet_markers = [], []
#for (marker, locator) in boundary_locator:
#    facets = mesh.locate_entities(domain_point_source, fdim, locator)
#    facet_indices.append(facets)
#    facet_markers.append(np.full_like(facets, marker))
#facet_indices = np.hstack(facet_indices).astype(np.int32)
#facet_markers = np.hstack(facet_markers).astype(np.int32)
#sorted_facets = np.argsort(facet_indices)
#facet_tag = mesh.meshtags(domain_point_source, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

#ds_p = Measure("ds_p", domain=domain_point_source, subdomain_data=facet_tag)


#def entire_dish_domain(x):
    #return np.full(x.shape[1],True,dtype=bool) 

def cell1_subdomain(x):
    return (x[0]-c1_x)**2 + (x[1]-c1_y)**2 <= r1**2 

def cell2_subdomain(x):
    return (x[0]-c2_x)**2 + (x[1]-c2_y)**2 <= r2**2 

subdomain_locator = [(1, cell1_subdomain),
              (2, cell2_subdomain)]

facet_indices2, facet_markers2 = [], []
for (marker, locator) in subdomain_locator:
    facets = mesh.locate_entities(domain_point_source, gdim, locator)
    facet_indices2.append(facets)
    facet_markers2.append(np.full_like(facets, marker))
facet_indices2 = np.hstack(facet_indices2).astype(np.int32)
facet_markers2 = np.hstack(facet_markers2).astype(np.int32)
sorted_facets2 = np.argsort(facet_indices2)
facet_tag2 = mesh.meshtags(domain_spatial_exclusion, gdim, facet_indices2[sorted_facets2], facet_markers2[sorted_facets2])

dx_p = Measure("dx", domain=domain_point_source, subdomain_data=facet_tag2)


from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
from petsc4py import PETSc

u_p = TrialFunction(V_p)
v_p = TestFunction(V_p)
u_pn = Function(V_p)
u_pn.interpolate(initial_condition)
delta1 = Function(V_p)
delta1.interpolate(diracd1)
fxi = Function(V_p)
fxi.interpolate(xi)
fxx = Function(V_p)
fxx.interpolate(xx)
fyy = Function(V_p)
fyy.interpolate(yy)


a_p = u_p * v_p * dx_p + dt * D * dot(grad(u_p), grad(v_p)) * dx_p
L_p = u_pn * v_p * dx_p + 2 * np.pi * r1 * b * dt * delta1 * v_p * dx_p
linearform_p = form (L_p)

A_p = assemble_matrix(form(a_p))
A_p.assemble()
b_p = create_vector(form(L_p))


t1_p = 2 * dt * a / r1 * u_p * dx_p(1) + dt * a / r1 * dot(grad(fxx),grad(u_p)) * dx_p(1) + dt * a / r1 * dot(grad(fyy),grad(u_p)) * dx_p(1) 
#t1_p = 2 * dt * a * fxi * u_p * dx_p + dt * a * fxi * dot(grad(fxx),grad(u_p)) * dx_p + dt * a * fxi * dot(grad(fyy),grad(u_p)) * dx_p 
q1_p = delta1 * v_p * dx_p
tt1_p = create_vector(form(t1_p))
qq1_p = create_vector(form(q1_p))
assemble_vector(tt1_p, form(t1_p))
assemble_vector(qq1_p, form(q1_p))


T1_p = PETSc.Mat().create()
T1_p.setSizes([tt1_p.getSize(), 1])
T1_p.setFromOptions()
T1_p.setUp()
T1_p.setValues(range(tt1_p.getSize()),0,tt1_p)
T1_p.assemble()
Q1_p = PETSc.Mat().create()
Q1_p.setSizes([qq1_p.getSize(), 1])
Q1_p.setFromOptions()
Q1_p.setUp()
Q1_p.setValues(range(qq1_p.getSize()),0,qq1_p)
Q1_p.assemble()
A1_p = Q1_p.matTransposeMult(T1_p)
A_p.axpy(1, A1_p)


solver_p = PETSc.KSP().create(domain_point_source.comm)
solver_p.setOperators(A_p)
#solver_p.setType(PETSc.KSP.Type.PREONLY)
#solver_p.getPC().setType(PETSc.PC.Type.LU)

u_s = Function(V_s)
u_p = Function(V_p)
u_p_res = Function(V_s)
w = Function(V_s)


grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V_s))
plotter = pyvista.Plotter()
plotter.open_gif("difference.gif", fps=10)
grid.point_data["w"] = w.x.array
warped = grid.warp_by_scalar("w", factor=1)
viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
             position_x=0.1, position_y=0.8, width=0.8, height=0.1)
renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
                            cmap=viridis, scalar_bar_args=sargs,
                            clim=[0, 2])
#xdmf = io.XDMFFile(domain_point_source.comm, "point_source.xdmf", "w")
#xdmf.write_mesh(domain_point_source)

if domain_spatial_exclusion.comm.rank == 0:
    e_w = np.zeros(num_steps, dtype=np.float64)
    t_e = np.zeros(num_steps, dtype=np.float64)
    i = 0


for i in range(num_steps):
    t += dt
    t_e[i] = t
    with b_s.localForm() as loc_b:
    	loc_b.set(0)
    assemble_vector(b_s, linearform_s)
    # Solve linear problem
    solver_s.solve(b_s, u_s.vector)
    u_s.x.scatter_forward()
    u_sn.x.array[:] = u_s.x.array
    with b_p.localForm() as loc_bb:
        loc_bb.set(0)
    assemble_vector(b_p, linearform_p)
    # Solve linear problem
    solver_p.solve(b_p, u_p.vector)
    u_p.x.scatter_forward()
    u_pn.x.array[:] = u_p.x.array
    #xdmf.write_function(u_p, t)
    u_p_res.interpolate(u_p, nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
        u_p_res.function_space.mesh._cpp_object,
        u_p_res.function_space.element,
        u_p.function_space.mesh._cpp_object,padding=0))
    u_p_res.x.scatter_forward()
    w.x.array[:] = np.abs(u_p_res.x.array[:] - u_s.x.array[:])
    new_warped = grid.warp_by_scalar("w", factor=1)
    warped.points[:, :] = new_warped.points
    warped.point_data["w"][:] = w.x.array
    plotter.write_frame()
    eL2_local = domain_spatial_exclusion.comm.allreduce(assemble_scalar(fem.form(w * w * dx_s)), op=MPI.SUM)
    e_w[i] = np.sqrt(np.abs(eL2_local))
    print(i+1,":",e_w[i])
    i += 1

       
fig = plt.figure(figsize=(35, 10))
l1 = plt.plot(t_e, e_w, label="L2-error", linewidth=2)
plt.title("L_2")
plt.grid()
plt.legend()
plt.savefig("Comparison.png")


#os.remove("spactial_exclusion_try.gif")
#plotter.close()
#xdmf.close()