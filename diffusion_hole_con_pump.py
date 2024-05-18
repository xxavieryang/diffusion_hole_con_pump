from mpi4py import MPI
import numpy as np
from dolfinx import mesh
import gmsh
from dolfinx.io import gmshio
from dolfinx.fem import functionspace


# Create the computation domain and geometric constant

gmsh.initialize()

L = W = 10
r = 0.5
c_x = 2.2
c_y = 2.6
gdim = 2
mesh_comm = MPI.COMM_WORLD
model_rank = 0

#Define the Petri dish and the cells

if mesh_comm.rank == model_rank:
	dish = gmsh.model.occ.addRectangle(0, 0, 0, L, W, tag=1)
	cell1 = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
	cell2 = gmsh.model.occ.addDisk(c_x+2, c_y+3, 0, r, r)
	cell3 = gmsh.model.occ.addDisk(c_x+6, c_y+6, 0, r, r)

#cut out for point source model

if  mesh_comm.rank == model_rank:
	point_source_domain = gmsh.model.occ.cut([(gdim,dish)],[(gdim,cell1)])
	point_source_domain = gmsh.model.occ.cut([(gdim,dish)],[(gdim,cell2)])
	point_source_domain = gmsh.model.occ.cut([(gdim,dish)],[(gdim,cell3)])
	gmsh.model.occ.synchronize()


liquid_marker = 1
if mesh_comm.rank == model_rank:
	volumes = gmsh.model.getEntities(dim=gdim)
	print(len(volumes))
	assert (len(volumes) == 1)
	gmsh.model.addPhysicalGroup(volumes[0][0],[volumes[0][1]],liquid_marker)
	gmsh.model.setPhysicalName(volumes[0][0],liquid_marker,"Liquid")

wall_marker, cell_marker = 2, 3

wall, cell1 = [], []

if mesh_comm.rank == model_rank:
	boundaries = gmsh.model.getBoundary(volumes, oriented = False)
	for boundary in boundaries:
		center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0],boundary[1])
		if np.allclose(center_of_mass,[L/2,0,0]) or np.allclose(center_of_mass,[0,W/2,0]) or \
		np.allclose(center_of_mass,[L,W/2,0]) or np.allclose(center_of_mass,[L/2,W,0]):
			wall.append(boundary[1])
		else:
			cell1.append(boundary[1])
	gmsh.model.addPhysicalGroup(1,wall,wall_marker)
	gmsh.model.setPhysicalName(1,wall_marker,"Wall")
	gmsh.model.addPhysicalGroup(1,cell1,cell_marker)
	gmsh.model.setPhysicalName(1,cell_marker,"Cell")

res_min = r / 3
if mesh_comm.rank == model_rank:
	distance_field = gmsh.model.mesh.field.add("Distance")
	gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", cell1)
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
	gmsh.model.mesh.setOrder(2)
	gmsh.model.mesh.optimize("Netgen")

domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
V = functionspace(domain, ("Lagrange", 1))

import pyvista
print(pyvista.global_theme.jupyter_backend)

from dolfinx import plot
topology, cell_types, geometry = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("fundamentals_mesh.png")