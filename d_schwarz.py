import numpy as np
from stl import mesh
import math


siderefnode0 = [[( 0, 0, 0),2],
				[( 0, 0, 0),1],
				[(-1, 0, 0),3],
				[(-1, 0,-1),2],
				[( 0,-1,-1),1],
				[( 0,-1, 0),3]]

siderefnode1 = [[( 0, 0, 0),0],
				[( 0, 0, 0),3],
				[( 0, 1, 0),2],
				[( 0, 1, 1),0],
				[(-1, 0, 1),3],
				[(-1, 0, 0),2]]

siderefnode2 = [[( 0, 0, 0),3],
				[( 0, 0, 0),0],
				[( 0,-1, 0),1],
				[( 0,-1, 1),3],
				[( 1, 0, 1),0],
				[( 1, 0, 0),1]]

siderefnode3 = [[( 0, 0, 0),1],
				[( 0, 0, 0),2],
				[( 1, 0, 0),0],
				[( 1, 0,-1),1],
				[( 0, 1,-1),2],
				[( 0, 1, 0),0]]

rot_60_tform = np.array([[0.5, -0.5*math.sqrt(3), 0.0],
						 [0.5*math.sqrt(3), 0.5, 0.0],
						 [0.0, 0.0, 1.0]])

def align_points(meshobj,threshold):
	allvec = meshobj.vectors
	for fi, face in enumerate(allvec):
		for vi,vector in enumerate(face):
			for ofi,otherface in enumerate(allvec):
				if ofi != fi:
					for ovi, othervector in enumerate(otherface):
						if np.linalg.norm(othervector-vector) < thresh:
							meshobj.vectors[fi][vi] = (othervector+vector)/2.0
							meshobj.vectors[ofi][ovi] = (othervector+vector)/2.0


def scale(meshobj,factor):
	vects = meshobj.vectors
	scalemat = np.diag(np.array(factor))
	for vsi,vectset in enumerate(vects):
		for vi,vector in enumerate(vectset):
			meshobj.vectors[vsi][vi] = np.dot(vector,scalemat)

def translate(meshobj,tvect):
	vects = meshobj.vectors
	for vsi,vectset in enumerate(vects):
		for vi,vector in enumerate(vectset):
			meshobj.vectors[vsi][vi] = vector+tvect

# We should have a matrix that tells us whether a voxel has material, and
# a matrix that tells us what nodes in that voxel are present.
def nodeneighbors(mat_matrix, invol, cl):
	#node 1
	node0sides = np.zeros(6)
	node1sides = np.zeros(6)
	node2sides = np.zeros(6)
	node3sides = np.zeros(6)

	for i in range(6):
		node0sides[i] = -1*(invol[cl[0]+siderefnode0[i][0][0]][cl[1]+siderefnode0[i][0][1]][cl[2]+siderefnode0[i][0][2]][siderefnode0[i][1]]-1)
		node1sides[i] = -1*(invol[cl[0]+siderefnode1[i][0][0]][cl[1]+siderefnode1[i][0][1]][cl[2]+siderefnode1[i][0][2]][siderefnode1[i][1]]-1)
		node2sides[i] = -1*(invol[cl[0]+siderefnode2[i][0][0]][cl[1]+siderefnode2[i][0][1]][cl[2]+siderefnode2[i][0][2]][siderefnode2[i][1]]-1)
		node3sides[i] = -1*(invol[cl[0]+siderefnode3[i][0][0]][cl[1]+siderefnode3[i][0][1]][cl[2]+siderefnode3[i][0][2]][siderefnode3[i][1]]-1)
		
	return np.vstack((node0sides,node1sides,node2sides,node3sides))

def node(beam_width, chamfer_factor, side_code):
	topcap = np.zeros(12,dtype=mesh.Mesh.dtype)
	side = np.zeros(12,dtype=mesh.Mesh.dtype)
	chamside = np.zeros(12,dtype=mesh.Mesh.dtype)

	center = np.array([0.0, 0.0, beam_width/2.0])
	
	topbound = np.array([[ 0.5*chamfer_factor*beam_width, beam_width+0.5*math.sqrt(3)*chamfer_factor*beam_width, beam_width/2.0],
						 [-0.5*chamfer_factor*beam_width, beam_width+0.5*math.sqrt(3)*chamfer_factor*beam_width, beam_width/2.0]])
	for i in range(10):
		vec = topbound[-2]
		topbound = np.vstack((topbound,np.dot(rot_60_tform,vec)))

	botbound = np.copy(topbound)
	botbound.T[2] = -botbound.T[2]

	for i in range(12):
		topcap['vectors'][i] = np.vstack((topbound[i],center,topbound[i-1]))
	
	botcap = mesh.Mesh(topcap.copy())
	botcap.rotate([0.0,1.0,0.0],math.radians(180))
	

	for i in range(0,12,2):
		if side_code[int(i/2)] == 1:
			side['vectors'][i] = np.vstack((botbound[i-1],botbound[i],topbound[i]))
			side['vectors'][i+1]= np.vstack((topbound[i],topbound[i-1],botbound[i-1]))

	for i in range(-1,11,2):
		chamside['vectors'][i] = np.vstack((botbound[i-1],botbound[i],topbound[i]))
		chamside['vectors'][i+1]= np.vstack((topbound[i],topbound[i-1],botbound[i-1]))

	nodecapbase = mesh.Mesh(np.concatenate([
	    topcap.copy(),
	    botcap.data.copy(),
	    side.copy(),
	    chamside.copy()
	]))

	return nodecapbase



strut_width = 0.6 #mm
lattice_pitch = 3.5 #mm
unit_cell = lattice_pitch*np.sqrt(2)
cham_factor = 0.3
thresh = 0.01 

mat_matrix = np.zeros((3,3,3))
mat_matrix[1][1][1] = 1
invol = np.zeros((3,3,3,4))
invol[1][1][1] = [1,1,1,1]

sides = nodeneighbors(mat_matrix,invol,(1,1,1))
#node(strut_width,cham_factor)

data = np.zeros(2,dtype=mesh.Mesh.dtype)

#Top of the strut
data['vectors'][0] = np.array([[ 0.5     ,    0, 0.5     ],
                                [ 0.304738,  1.0, 0.638071],
                                [-0.638071,  1.0, 0.304638]])
data['vectors'][1] = np.array([[ 0.5     ,    0, 0.5     ],
                                [-0.638071,  1.0, 0.304638],
                                [-0.5     ,    0, 0.5     ]])

strut = mesh.Mesh(data.copy())

for i in range(3):
	tmpside = mesh.Mesh(data.copy())
	tmpside.rotate([0.0,1.0,0.0],math.radians(90*(i+1)))

	strut = mesh.Mesh(np.concatenate([
	    strut.data,
	    tmpside.data.copy()
	]))

align_points(strut,thresh)
scale(strut,[strut_width,lattice_pitch,strut_width])



node0 = node(strut_width,cham_factor,sides[0])
node0.rotate([0.0,0.0,1.0],math.radians(45))
node1 = node(strut_width,cham_factor,sides[1])
node1.rotate([0.0,0.0,1.0],math.radians(45))
node2 = node(strut_width,cham_factor,sides[2])
node2.rotate([0.0,0.0,1.0],math.radians(45))
node3 = node(strut_width,cham_factor,sides[3])
node3.rotate([0.0,0.0,1.0],math.radians(45))

node0.rotate([1.0,-1.0,0.0],-math.atan(math.sqrt(2)))

node3.rotate([1.0,-1.0,0.0],-math.atan(math.sqrt(2)))
node3.rotate([0.0,0.0,1.0],math.radians(180))

node1.rotate([1.0,-1.0,0.0],math.atan(math.sqrt(2)))
node1.rotate([0.0,0.0,1.0],math.radians(90))

node2.rotate([1.0,-1.0,0.0],math.atan(math.sqrt(2)))
node2.rotate([0.0,0.0,1.0],math.radians(90))
node2.rotate([0.0,0.0,1.0],math.radians(180))


translate(node0,np.array([unit_cell*0.25]*3))
translate(node3,np.array([0.75,0.75,0.25])*unit_cell)
translate(node1,np.array([0.25,0.75,0.75])*unit_cell)
translate(node2,np.array([0.75,0.25,0.75])*unit_cell)


# Optionally render the rotated cube faces
from matplotlib import pyplot
from mpl_toolkits import mplot3d

# Create a new plot
figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)

axes.set_xlim(0,unit_cell)
axes.set_ylim(0,unit_cell)
axes.set_zlim(0,unit_cell)

# Render the cube
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(node0.vectors))
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(node1.vectors))
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(node2.vectors))
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(node3.vectors))
#axes.add_collection3d(mplot3d.art3d.Poly3DCollection(strut2.vectors))


# Show the plot to the screen
pyplot.show()

