from __future__ import print_function
from fenics import *
from mshr import *
import time 
import matplotlib.pyplot as plt
#from dolfin import *
import numpy as np
from math import *
from copy import *
from scipy.optimize import fsolve
#from Line import *
import pandas as pd
import csv
from ufl import nabla_div


"""
File Description:
Domain: Computational domain (container) with the hole as a subdomain
Mesh: Fixed mesh
Method: Immersed boundary

BVP to be solved: du/dt - D $Delta$ u = int $phi$(x)n(x)$delta$(x) d$Gamma$, in $Omega$\
                  D\nabla u $cdot$ n = 0, on $Gamma$

"""

parameters['reorder_dofs_serial'] = False

#np.random.seed(2)
############################################################################
### Define computational domain and function space
############################################################################
x0, y0 = 10, 10 ### computational domain size

center_1 = np.array([-3.5, -3.8]) ### center of the hole domain
radius_1 = 0.5 ### radius of the hole domain

#xy: center_2 = np.array([3.5, 4])
#xy: radius_2 = 0.5

#### preparation for plotting
phi = np.arange(0, 2 * np.pi, 0.01)

domain = Rectangle(Point(-x0,-y0),Point(x0,y0))
hole_1 = Circle(Point(center_1[0],center_1[1]),radius_1)
#xy: hole_2 = Circle(Point(center_2[0],center_2[1]),radius_2)

resol = 30

##############################################################################
##############################################################################
##### the mesh of the HOLE approach
##############################################################################
##############################################################################
#xy: domain_h = domain - hole_1 - hole_2
domain_h = domain - hole_1
mesh_h = generate_mesh(domain_h, resol) ### the FEM mesh
subdomains_h = MeshFunction('size_t', mesh_h, mesh_h.topology().dim(), mesh_h.domains()) ### subdomains data

############################################################################
### Possibilities to refine the mesh
############################################################################
#mesh=refine(mesh)
#subdomains=adapt(subdomains, mesh)
#mesh=refine(mesh)
#subdomains=adapt(subdomains, mesh)
# mesh=refine(mesh)
# subdomains=adapt(subdomains,mesh)
# mesh=refine(mesh)
# subdomains=adapt(subdomains,mesh)

n_norm = FacetNormal(mesh_h) ### outward normal unit vector

V_h = FunctionSpace(mesh_h, "P", 1) ### Lagrange base function is used

meshpoints_h = mesh_h.coordinates() #coordinates of all the mesh, the order is corresponding to the index of vertices
dx_h = Measure('dx',subdomain_data = subdomains_h)
submesh_h = SubMesh(mesh_h, subdomains_h, 1)

print("mesh size h: ", (mesh_h.hmax()+mesh_h.hmin())/2)


#### preparation for plotting
#phi= np.arange(0, 2*np.pi, 0.01)
#plot(mesh)
#plt.plot(center_1[0] + radius_1 * np.cos(phi), center_1[1] + radius_1 * np.sin(phi), linewidth=2.5,color='blue')
#plt.plot(center_2[0] + radius_2 * np.cos(phi), center_2[1] + radius_2 * np.sin(phi), linewidth=2.5,color='blue')
#plt.show()

############################################################################
### Define two boundaries separately
############################################################################
class interior_boundaries_1(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary and (x[0]-center_1[0])**2+(x[1]-center_1[1])**2-radius_1**2 <= mesh_h.hmin()/3


#xy: class interior_boundaries_2(SubDomain):
#xy:    def inside(self,x,on_boundary):
#xy:        return on_boundary and (x[0]-center_2[0])**2+(x[1]-center_2[1])**2-radius_2**2 <= mesh_h.hmin()/3


class exterior_boundaries(SubDomain):
    def inside(self,x,on_boundary):
        return on_boundary and (near(x[0],-x0,DOLFIN_EPS) or near(x[0],x0,DOLFIN_EPS) or near(x[1],-y0,DOLFIN_EPS) or near(x[1],y0,DOLFIN_EPS))


bmesh_h = BoundaryMesh(mesh_h, "exterior")
boundaries_h = MeshFunction('size_t', mesh_h, mesh_h.topology().dim()-1)
boundaries_h.set_all(0)
exterior_boundaries().mark(boundaries_h,1)
interior_boundaries_1().mark(boundaries_h, 2)
#xy: interior_boundaries_2().mark(boundaries_h, 3)
ds_h = Measure('ds', domain = mesh_h, subdomain_data = boundaries_h)

# interior boundary segments coordinates to build up the polygon
#interior_bmeshpoints_coor=np.array([])
#for i in interior_bmeshpoints:
#    interior_bmeshpoints_coor = np.append(interior_bmeshpoints_coor,np.array(interior_bmeshpoints[i][0]),axis=0)
#    interior_bmeshpoints_coor = np.append(interior_bmeshpoints_coor, np.array(interior_bmeshpoints[i][1]), axis=0)
#interior_bmeshpoints_coor=interior_bmeshpoints_coor.reshape((-1,2))
#interior_bmeshpoints_coor=UniqueRow(interior_bmeshpoints_coor).T
#interior_bmeshpoints_coor=ClockwisePoints(interior_bmeshpoints_coor)
#
#interior_bmeshpoints_coor_index=[]
#for i in np.arange(interior_bmeshpoints_coor.shape[1]):
#    interior_bmeshpoints_coor_index.append(np.asscalar(np.where(np.all(meshpoints==interior_bmeshpoints_coor[:,i],axis=1))[0]))


##############################################################################
##############################################################################
##### the mesh of the DIRAC DELTA approach
##############################################################################
##############################################################################
domain.set_subdomain(1, hole_1)
#yx: domain.set_subdomain(2, hole_2)

mesh_ps = generate_mesh(domain, resol) ### the FEM mesh
subdomains_ps = MeshFunction('size_t',mesh_ps, mesh_ps.topology().dim(), mesh_ps.domains()) ### subdomains data

plot(mesh_h)
plt.axis('square')
plt.xlim(-x0, x0)
plt.ylim(-y0, y0)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.scatter(center_1[0], center_1[1], color = "blue", s=10)
#plt.scatter(center_2[0], center_2[1], color = "blue", s=10)
plt.plot(center_1[0] + radius_1 * np.cos(phi), center_1[1] + radius_1 * np.sin(phi), linewidth=2, color='blue')
#xy: plt.plot(center_2[0] + radius_2 * np.cos(phi), center_2[1] + radius_2 * np.sin(phi), linewidth=2, color='blue')
plt.show()

############################################################################
### Possibilities to refine the mesh
############################################################################
#mesh=refine(mesh)
#subdomains=adapt(subdomains, mesh)
#mesh=refine(mesh)
#subdomains=adapt(subdomains, mesh)
# mesh=refine(mesh)
# subdomains=adapt(subdomains,mesh)
# mesh=refine(mesh)
# subdomains=adapt(subdomains,mesh)


V_ps = FunctionSpace(mesh_ps, "P", 1) ### Langrage base function is used


#### edit the mesh to make sure the center is the nodal mesh point
meshpoints_ps = mesh_ps.coordinates() #coordinates of all the mesh, the order is corresponding to the index of vertices
center_mesh_index_1 = np.argmin(np.linalg.norm(meshpoints_ps - center_1, axis = 1))
# center_mesh_index_2 = np.argmin(np.linalg.norm(meshpoints_ps - center_2, axis = 1))

mesh_ps.coordinates()[center_mesh_index_1] = center_1
#xy: mesh_ps.coordinates()[center_mesh_index_2] = center_2

print("mesh size h: ", (mesh_ps.hmax()+mesh_ps.hmin())/2)
##############################################################################
### Get the index of same point in meshpoints_ps
##############################################################################
#points_index = []
#for i in range(meshpoints_h.shape[0]):
#    points_index += [np.argmin(np.linalg.norm(meshpoints_ps - meshpoints_h[i], axis = 1))]
#    mesh_ps.coordinates()[np.argmin(np.linalg.norm(meshpoints_ps - meshpoints_h[i], axis = 1))] = meshpoints_h[i]


dx_ps = Measure('dx',subdomain_data = subdomains_ps)
submesh_hole_1 = SubMesh(mesh_ps, subdomains_ps, 1)
#xy: submesh_hole_2 = SubMesh(mesh_ps, subdomains_ps, 2)

print("mesh size h: ", (mesh_ps.hmax()+mesh_ps.hmin())/2)

interior_bmeshpoints_1_index = []
#xy: interior_bmeshpoints_2_index = []


submesh_hole_1_vert_index = submesh_hole_1.data().array('parent_vertex_indices', 0)  # the vertices index in submesh corresponding to the global
interior_bmeshpoints_1=dict()
for f in edges(submesh_hole_1):
    p1=Vertex(submesh_hole_1,f.entities(0)[0])
    p2=Vertex(submesh_hole_1,f.entities(0)[1])
    if abs((p1.x(0)-center_1[0])**2+(p1.x(1)-center_1[1])**2-radius_1**2)<mesh_ps.hmin()/3 and abs((p2.x(0)-center_1[0])**2+(p2.x(1)-center_1[1])**2-radius_1**2)<mesh_ps.hmin()/3:
        interior_bmeshpoints_1[f.index()] = [[p1.x(0), p1.x(1)], [p2.x(0), p2.x(1)], [submesh_hole_1_vert_index[p1.index()], submesh_hole_1_vert_index[p2.index()]]]
        interior_bmeshpoints_1_index += [submesh_hole_1_vert_index[p1.index()], submesh_hole_1_vert_index[p2.index()]]

        
#xy: submesh_hole_2_vert_index=submesh_hole_2.data().array('parent_vertex_indices', 0)  # the vertices index in submesh corresponding to the global
#xy: interior_bmeshpoints_2=dict() ### keys are local index
#xy: for f in edges(submesh_hole_2):
#xy:    p1=Vertex(submesh_hole_2,f.entities(0)[0])
#xy:    p2=Vertex(submesh_hole_2,f.entities(0)[1])
#xy:    if abs((p1.x(0)-center_2[0])**2+(p1.x(1)-center_2[1])**2-radius_2**2)<mesh_ps.hmin()/3 and abs((p2.x(0)-center_2[0])**2+(p2.x(1)-center_2[1])**2-radius_2**2)<mesh_ps.hmin()/3:
#xy:        interior_bmeshpoints_2[f.index()] = [[p1.x(0), p1.x(1)], [p2.x(0), p2.x(1)], [submesh_hole_2_vert_index[p1.index()], submesh_hole_2_vert_index[p2.index()]]]
#xy:        interior_bmeshpoints_2_index += [submesh_hole_2_vert_index[p1.index()], submesh_hole_2_vert_index[p2.index()]]


interior_bmeshpoints_1_index = list(set(interior_bmeshpoints_1_index))
#xy: interior_bmeshpoints_2_index = list(set(interior_bmeshpoints_2_index))

V_vec_ps = VectorFunctionSpace(mesh_ps, "P", 1)
nh = Function(V_vec_ps)
nh_array = nh.vector().get_local().reshape((2, -1))
nh_array[:, interior_bmeshpoints_1_index] = -(meshpoints_ps.T[:, interior_bmeshpoints_1_index] - center_1[:, np.newaxis])/radius_1
#xy: nh_array[:, interior_bmeshpoints_2_index] = -(meshpoints_ps.T[:, interior_bmeshpoints_2_index] - center_2[:, np.newaxis])/radius_2
nh.vector().set_local(nh_array.flatten())    
    

############################################################################
### Define two boundaries separately
############################################################################
class interior_boundaries_1(SubDomain):
    def inside(self,x,on_boundary):
        return abs((x[0]-center_1[0])**2+(x[1]-center_1[1])**2-radius_1**2) <= mesh_ps.hmin()/3


#xy: class interior_boundaries_2(SubDomain):
#xy:    def inside(self,x,on_boundary):
#xy:        return abs((x[0]-center_2[0])**2+(x[1]-center_2[1])**2-radius_2**2) <= mesh_ps.hmin()/3



boundaries_ps = MeshFunction('size_t', mesh_ps, mesh_ps.topology().dim() - 1)
boundaries_ps.set_all(0)
interior_boundaries_1().mark(boundaries_ps, 1)
#xy: interior_boundaries_2().mark(boundaries_ps, 2)
dS_ps = Measure('dS', domain = mesh_ps, subdomain_data=boundaries_ps)

##############################################################
#### extract facet norm as a function
##############################################################
#def n_norm_to_nh():
#    V_vec = VectorFunctionSpace(mesh, "CG", 1)
#    u_vec = TrialFunction(V_vec)
#    v_vec = TestFunction(V_vec)
#    a_vec = inner(u_vec, v_vec)('+')* dS
#    l_vec = inner(n_norm, v_vec)('-')* dS
#    A_vec = assemble(a_vec, keep_diagonal = True)
#    L_vec = assemble(l_vec)
#    A_vec.ident_zeros()
#    nh = Function(V_vec)
#    solve(A_vec, nh.vector(), L_vec)
#    return nh
#
#nh = n_norm_to_nh()
#nh_vec = nh.vector().get_local().reshape((2, -1))
#nh_vec = np.nan_to_num(nh_vec/np.linalg.norm(nh_vec, axis = 0))
#nh.vector().set_local(np.squeeze(nh_vec.reshape((1, -1))))
###################################################################

plot(mesh_h)
plt.tick_params(which='major', labelsize = 30)
plt.xlabel("x-coordinate", fontsize = 30)
plt.ylabel("y-coordinate", fontsize = 30)
#plt.scatter(center_1[0], center_1[1], color = "red", s = 20)
#plt.scatter(center_2[0], center_2[1], color = "red", s = 20)
plt.plot(center_1[0] + radius_1 * np.cos(phi), center_1[1] + radius_1 * np.sin(phi), linewidth=2.5,color='blue')
#xy: plt.plot(center_2[0] + radius_2 * np.cos(phi), center_2[1] + radius_2 * np.sin(phi), linewidth=2.5,color='blue')
plt.show()

  

############################################################################
### Preparation for the loop
############################################################################
P = 1.0 ### density of the flux, can be a function too
#P = Expression('sin(atan((x[1] - center_1)/(x[0] - center_0 + eps)))', degree = 2, eps = 1e-5, center_0 = center[0], center_1 = center[1])

w_L2_sq_list = []
w_L2_list = []
w_H1_list = []
c_star_sqrt_list = []
c_star_1_list, c_star_2_list, c_star_list = [], [], []

T = 2030 ### final time
num_steps = 5000 ### number of time steps
dt = T / num_steps ### time step size

D = 0.1 ### diffusion rate
#D = Expression('pow(x[0] - a, 2)+pow(x[1] - b, 2)>=pow(radius_1,2)+tol && pow(x[0] - c, 2)+pow(x[1] - d, 2)>=pow(radius_2,2)+tol ? D_O:D_H', \
#               degree=2, tol = 0, a = center_1[0], b = center_1[1], c = center_2[0], d = center_2[1], radius_1 = radius_1, radius_2 = radius_2, D_H = 1E8, D_O = D)

####### Hole = 1, Other + Hole Bound = 0
#xy: hole_indicator = Expression('(pow(x[0] - a, 2)+pow(x[1] - b, 2)>=pow(radius_1,2)+tol || pow(x[0] - a, 2)+pow(x[1] - b, 2)>=pow(radius_1,2)-tol) && (pow(x[0] - c, 2)+pow(x[1] - d, 2)>=pow(radius_2,2)+tol || pow(x[0] - c, 2)+pow(x[1] - d, 2)>=pow(radius_2,2)-tol) ? D_O:D_H', \
#xy:               degree=2, tol = mesh_ps.hmin()/3, a = center_1[0], b = center_1[1], c = center_2[0], d = center_2[1], radius_1 = radius_1, radius_2 = radius_2, D_H = 1, D_O = 0)
hole_indicator = Expression('(pow(x[0] - a, 2)+pow(x[1] - b, 2)>=pow(radius_1,2)+tol || pow(x[0] - a, 2)+pow(x[1] - b, 2)>=pow(radius_1,2)-tol) ? D_O:D_H', \
               degree=2, tol = mesh_ps.hmin()/3, a = center_1[0], b = center_1[1], radius_1 = radius_1, D_H = 1, D_O = 0)

###### Hole + Hole Bound = 1, Other = 0
#xy:
#xy: hole_indicator_bnd = Expression('(pow(x[0] - a, 2)+pow(x[1] - b, 2)>pow(radius_1,2)+tol || pow(x[0] - a, 2)+pow(x[1] - b, 2)>pow(radius_1,2)-tol) && (pow(x[0] - c, 2)+pow(x[1] - d, 2)>pow(radius_2,2)+tol || pow(x[0] - c, 2)+pow(x[1] - d, 2)>pow(radius_2,2)-tol) ? D_O:D_H', \
#xy:              degree=2, tol = mesh_ps.hmin()/3, a = center_1[0], b = center_1[1], c = center_2[0], d = center_2[1], radius_1 = radius_1, radius_2 = radius_2, D_H = 1, D_O = 0)
hole_indicator_bnd = Expression('(pow(x[0] - a, 2)+pow(x[1] - b, 2)>pow(radius_1,2)+tol || pow(x[0] - a, 2)+pow(x[1] - b, 2)>pow(radius_1,2)-tol) ? D_O:D_H', \
              degree=2, tol = mesh_ps.hmin()/3, a = center_1[0], b = center_1[1], radius_1 = radius_1, D_H = 1, D_O = 0)




# =============================================================================
# Define variational problem
# =============================================================================
### hole approach

u_h_0 = Constant(0)
u_h_n = interpolate(u_h_0, V_h)

u_h = TrialFunction(V_h)
v_h = TestFunction(V_h)

#xy: F_h = u_h * v_h * dx_h + D * dt * dot(grad(u_h), grad(v_h)) * dx_h - dt * P * v_h * (ds_h(2) + ds_h(3)) - u_h_n * v_h * dx_h
F_h = u_h * v_h * dx_h + D * dt * dot(grad(u_h), grad(v_h)) * dx_h - dt * P * v_h * ds_h(2) - u_h_n * v_h * dx_h
a_h, L_h = lhs(F_h), rhs(F_h)

### dirac delta approach
#exec(open("flux_initial_opt.py").read())
P0 =  55.37856055
t0 = 3.65999141

###############################################################################
#t_list = np.linspace(0, T, num_steps+1)
#t_list_nonzero = np.delete(t_list, 0)
#
#def obj_func(x):
#    P0, t0 = x
#    return dt * (abs(P - P0 * radius_1/(2 * (t0))/(4 * pi * D * (t0)) * np.exp(-radius_1 ** 2/(4 * D * (t0))))\
#                 + np.sum(abs(P - P0 * radius_1/(2 * (t_list_nonzero+t0))/(4 * pi * D * (t_list_nonzero+t0)) * np.exp(-radius_1 ** 2/(4 * D * (t_list_nonzero+t0))) \
#                 - P * np.nan_to_num(np.exp(-radius_1 ** 2/(4 * D * t_list_nonzero))))))
#
#
#initial_guess=[8*pi*D* (radius_1**2/D) ** 2 /radius_1*exp(radius_1**2/(4*D*(radius_1**2/D))), radius_1 ** 2/(8*D)]
##cons = ({'type': 'eq', 'fun': lambda x:  x[0] - 8 * pi * D * x[1]**2 / radius_1 * exp(radius_1**2/(4*D*x[1]))})
#bnds = ((radius_1**2/(8*D), None), (0, None))
#res = optimize.minimize(obj_func, initial_guess, bounds=bnds)#, constraints = cons)
#P0, t0 = res.x
#################################################################################


#t0 = radius_1 ** 2/D
#P0 = P * 8 * pi * t0 ** 2 * D /radius_1 * exp(radius_1 ** 2/(4 * D * t0))
#exec(open("flux_initial.py").read())

u_ps_0 = Constant(0)
#u_ps_0 = Expression('P / (4*pi*eps*D)* exp(-(pow(x[0] - a, 2)+pow(x[1]-b, 2))/(4*D*eps)) + P / (4*pi*eps*D)* exp(-(pow(x[0] - c, 2)+pow(x[1]-d, 2))/(4*D*eps)) ', degree = 2, eps = t0, a = center_1[0], b = center_1[1], c = center_2[0], d = center_2[1], P = P0, D = D)


u_ps_n = project(u_ps_0 * hole_indicator, V_ps)

u_ps = TrialFunction(V_ps)
v_ps = TestFunction(V_ps)


#### derive the expression of the source term only considering the centre
#xy: ps = [(Point(center_1[0], center_1[1]), dt * P * np.pi * radius_1 * 2), (Point(center_2[0], center_2[1]), dt * P * np.pi * radius_2 * 2)]
ps = [(Point(center_1[0], center_1[1]), dt * P * np.pi * radius_1 * 2)]
ps = PointSource(V_ps, ps)


F_ps = u_ps * v_ps * dx_ps + D * dt * dot(grad(u_ps), grad(v_ps))*dx_ps - (u_ps_n) * v_ps * dx_ps
a_ps, L_ps = lhs(F_ps), rhs(F_ps)


#vtkfile = File('diffusion_hole_simu/solution_1.pvd')

# Time-stepping
u_h = Function(V_h)
u_ps = Function(V_ps)
t = 0
for n in range(num_steps):
    print('*************************************', '\n', n, '\n')
    
#    if n == 0:
#        w = Expression("P / (4*pi*D*t) * exp(-(pow(x[0] - a, 2)+pow(x[1]-b, 2))/(4*D*t))+P / (4*pi*D*t) * exp(-(pow(x[0] - c, 2)+pow(x[1]-d, 2))/(4*D*t))", degree = 2, t = t0, a = center_1[0], b = center_1[1], c = center_2[0], d = center_2[1], P = P0, D = D)
#        w = project(w * hole_indicator_bnd, V_h)
#        w_L2_list += [sqrt(assemble(inner(w, w)*dx_h))]
#        w_L2_sq_list += [assemble(inner(w, w)*dx_h)]
#        w_H1_list += [sqrt(assemble((inner(w, w) + inner(grad(w), grad(w)))*dx_h))]

    t += dt
    
    # Compute solution
    solve(a_h == L_h, u_h)
    
    A_ps, b_ps = assemble_system(a_ps, L_ps)
    ps.apply(b_ps)
    
# Compute solution
#   solve(a == L, u)
    solve(A_ps, u_ps.vector(), b_ps)
    
    # Update previous solution
    u_h_n.assign(u_h)
    u_ps_n.assign(u_ps)
    
    # Calculate the difference of the two solutions
    u_ps_ad = Function(V_h)
    for i in range(meshpoints_h.shape[0]):
        ps_u = PointSource(V_h, Point(meshpoints_h[i,0], meshpoints_h[i,1]), u_ps(meshpoints_h[i, 0], meshpoints_h[i, 1]))
        ps_u.apply(u_ps_ad.vector())
#    u_ps_ad.vector().set_local(u_ps.vector().get_local()[points_index])
    
    w = Function(V_h)
    w = project((u_h - u_ps_ad), V_h)
    print(sqrt(assemble(inner(w, w)*dx_h)))
    
# Save to file and plot solution
#    vtkfile << (u, t)
#    plt.clf()
#    plt.ion()
#
#    fig = plot(u)
#    plt.title(n)
#    plot(mesh)
#    plt.plot(center_1[0] + radius_1 * np.cos(phi), center_1[1] + radius_1 * np.sin(phi), linewidth=1,color='blue')
#    plt.plot(center_2[0] + radius_2 * np.cos(phi), center_2[1] + radius_2 * np.sin(phi), linewidth=1,color='blue')
#    fig_cb = plt.colorbar(fig)
#    fig_cb.set_clim(vmin = 0,vmax = 1)
#    
#    plt.pause(0.001)
#    plt.show()

    c_t = project(Constant(P) - D * inner(grad(u_ps), nh))
    c_L2_1 = dt * assemble(inner(c_t, c_t) * dS_ps(1)) 
    c_L2_2 = dt * assemble(inner(c_t, c_t) * dS_ps(2))
    c_L2 = c_L2_1 + c_L2_2
#    
    if len(c_star_list) == 0:
        c_star_sqrt_list += [sqrt(c_L2)]
        c_star_1_list += [c_L2_1]
        c_star_2_list += [c_L2_2]
        c_star_list += [c_L2]
    else:
        c_star_sqrt_list += [sqrt(c_L2 + c_star_list[-1])]
        c_star_1_list += [c_L2_1 + c_star_1_list[-1]]
        c_star_2_list += [c_L2_2 + c_star_2_list[-1]]
        c_star_list += [c_L2 + c_star_list[-1]]
    
    w_L2_sq_list += [assemble(inner(w, w)*dx_h)]    
    w_L2_list += [sqrt(assemble(inner(w, w)*dx_h))]
    w_H1_list += [sqrt(assemble((inner(w, w) + inner(grad(w), grad(w)))*dx_h))]

    print("||u_h||_{L2}: ", sqrt(assemble(inner(u_h, u_h)*dx_h)))
    print("||u_ps||_{L2}: ", sqrt(assemble(inner(u_ps, u_ps)*dx_ps(0))))
    print(u_ps_ad(0.5, 0.0), u_ps(0.5, 0.0))
    print("||w||_{L2}: ", sqrt(assemble(inner(w, w)*dx_h)))
    print("||u_h||_{L2} - ||u_ps||_{L2}:", abs(norm(u_h, 'l2') - norm(u_ps_ad, 'l2')))
    print("||u_h||_{L2} - ||u_ps||_{L2}:", abs(norm(u_h, 'l2') - sqrt(assemble(inner(u_ps, u_ps)*dx_ps(0)))))
    print("||w||_{H1}: ", sqrt(assemble((inner(w, w) + inner(grad(w), grad(w)))*dx_h)))
    print("c_star_sqrt:", c_star_sqrt_list[-1])
    print("c_star:", c_star_list[-1])



t_list = np.linspace(0, T, num_steps+1)

plt.plot(w_L2_list, linewidth = 2.5, label = "w_L2_norm")
plt.legend(loc = "best")
plt.show()

#c1 = 1
#eps = 0.15   
#plt.plot(w_L2_sq_list, linewidth = 2.5, label = "w_L2_norm^2")
#plt.plot(c1 * np.array(c_star_list) * np.exp(eps * t_list), label = "c_star_bnd")
#plt.legend(loc = "best")
#plt.show()

with open('number_of_holes/w_homo_zero_two_asys_holes.csv', 'w') as f:
    # create the csv writer
    writer = csv.writer(f)
#    writer.writerow(["max", "L2_norm", "H1_norm", "c_star", "c_star_sqrt"])
    writer.writerow(["w_L2_norm", "w_H1_norm", "c_star_1", "c_star_2", "c_star"])
    
    # write a row to the csv file
    for i in range(num_steps):    
#        writer.writerow([u_max_list[i], u_L2_list[i], u_H1_list[i], c_star_list[i], c_star_sqrt_list[i]])
        writer.writerow([w_L2_list[i], w_H1_list[i], c_star_1_list[i], c_star_2_list[i], c_star_list[i]])
