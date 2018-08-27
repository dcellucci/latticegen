import stl
from stl import mesh
import numpy as np
import math

rot_60_tform = np.array([[0.5, -0.5*math.sqrt(3), 0.0],
						 [0.5*math.sqrt(3), 0.5, 0.0],
						 [0.0, 0.0, 1.0]])

rot_90_tform = np.array([[0.0, -1.0, 0.0],
						 [1.0, 0.0, 0.0],
						 [0.0, 0.0, 1.0]])

nodessideref = 	[
				[[ 0, 0, 1],  #node 0
				 [-1, 0, 0],
				 [-1, 0, 0],
				 [ 0,-1, 0],
				 [ 0,-1, 0],
				 [ 0, 0, 1]],
				[[ 0, 0, 1],  #node 1
				 [ 0, 1, 0],
				 [ 0, 1, 0],
				 [-1, 0, 0],
				 [-1, 0, 0],
				 [ 0, 0, 1]],
				[[ 0, 0, 1],  #node 2
				 [ 1, 0, 0],
				 [ 1, 0, 0],
				 [ 0, 1, 0],
				 [ 0, 1, 0],
				 [ 0, 0, 1]],
				[[ 0, 0, 1],  #node 3
				 [ 0,-1, 0],
				 [ 0,-1, 0],
				 [ 1, 0, 0],
				 [ 1, 0, 0],
				 [ 0, 0, 1]],
				[[ 0, 0,-1],  #node 4
				 [-1, 0, 0],
				 [-1, 0, 0],
				 [ 0,-1, 0],
				 [ 0,-1, 0],
				 [ 0, 0,-1]],
				[[ 0, 0,-1],  #node 5
				 [ 0, 1, 0],
				 [ 0, 1, 0],
				 [-1, 0, 0],
				 [-1, 0, 0],
				 [ 0, 0,-1]],
				[[ 0, 0,-1],  #node 6
				 [ 1, 0, 0],
				 [ 1, 0, 0],
				 [ 0, 1, 0],
				 [ 0, 1, 0],
				 [ 0, 0,-1]],
				[[ 0, 0,-1],  #node 7
				 [ 0,-1, 0],
				 [ 0,-1, 0],
				 [ 1, 0, 0],
				 [ 1, 0, 0],
				 [ 0, 0,-1]]
				]

beamNeighCode = [[ 0,-1, 0],
				 [ 0, 1, 0],
				 [-1, 0, 0],
				 [ 1, 0, 0],
				 [ 0, 0,-1],
				 [ 0, 0, 1]]

hexcapref001 = [[1,0,0,0,0,1],
			 	[1,0,0,0,0,1],
			 	[1,0,0,0,0,1],
			 	[1,0,0,0,0,1],
			 	[0,0,0,0,0,0],
			 	[0,0,0,0,0,0],
			 	[0,0,0,0,0,0],
			 	[0,0,0,0,0,0]]

hexcapref00n1 = [[0,0,0,0,0,0],
			 	 [0,0,0,0,0,0],
			 	 [0,0,0,0,0,0],
			 	 [0,0,0,0,0,0],
			 	 [1,0,0,0,0,1],
			 	 [1,0,0,0,0,1],
			 	 [1,0,0,0,0,1],
			 	 [1,0,0,0,0,1]]

hexcapref010 = [[0,0,0,0,0,0],
			 	[0,1,1,0,0,0],
			 	[0,0,0,1,1,0],
			 	[0,0,0,0,0,0],
			 	[0,0,0,0,0,0],
			 	[0,1,1,0,0,0],
			 	[0,0,0,1,1,0],
			 	[0,0,0,0,0,0]]

hexcapref0n10 = [[0,0,0,1,1,0],
			 	 [0,0,0,0,0,0],
			 	 [0,0,0,0,0,0],
			 	 [0,1,1,0,0,0],
			 	 [0,0,0,1,1,0],
			 	 [0,0,0,0,0,0],
			 	 [0,0,0,0,0,0],
			 	 [0,1,1,0,0,0]]

hexcapref100 = [[0,0,0,0,0,0],
			 	[0,0,0,0,0,0],
			 	[0,1,1,0,0,0],
			 	[0,0,0,1,1,0],
			 	[0,0,0,0,0,0],
			 	[0,0,0,0,0,0],
			 	[0,1,1,0,0,0],
			 	[0,0,0,1,1,0]]

hexcaprefn100 = [[0,1,1,0,0,0],
			 	 [0,0,0,1,1,0],
			 	 [0,0,0,0,0,0],
			 	 [0,0,0,0,0,0],
			 	 [0,1,1,0,0,0],
			 	 [0,0,0,1,1,0],
			 	 [0,0,0,0,0,0],
			 	 [0,0,0,0,0,0]]

hexcaprefs = [hexcapref100,hexcaprefn100,hexcapref010,hexcapref0n10,hexcapref001,hexcapref00n1]

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

def rev_z(meshobj):
	for vectset in meshobj.vectors:
		tval = vectset[1].copy()
		vectset[1] = vectset[2].copy()
		vectset[2] = tval

def sideNeighbors(mat_matrix, nodenum, cellcoord):
	sidecode = [0]*6
	for side in range(6):
		if mat_matrix[cellcoord[0]+nodessideref[nodenum][side][0]][cellcoord[1]+nodessideref[nodenum][side][1]][cellcoord[2]+nodessideref[nodenum][side][2]] == 0:
			sidecode[side] = 1
	return sidecode

def beamNeighbors(mat_matrix,cellcoord):
	beamcode = [1]*6
	for i in range(6):
		if mat_matrix[cellcoord[0]+beamNeighCode[i][0]][cellcoord[1]+beamNeighCode[i][1]][cellcoord[2]+beamNeighCode[i][2]] == 0:
			beamcode[i] = 0
	return beamcode

def strut(strut_width,lattice_pitch,cham_factor):
	data = np.zeros(2,dtype=mesh.Mesh.dtype)
	#Top of the strut
	
	data['vectors'][0] = np.array([ [ 0.5     ,   0, 0.5     ],
	                                [-0.696872, 1.0, 0.11987 ],
	                                [ 0.11987,  1.0, 0.696872]])
	data['vectors'][1] = np.array([ [ 0.5     ,   0, 0.5     ],
	                                [-0.5     ,   0, 0.5     ],
	                                [-0.696872, 1.0, 0.11987 ]])
	

	strut = mesh.Mesh(data.copy())

	for i in range(3):
		tmpside = mesh.Mesh(data.copy())
		tmpside.rotate([0.0,1.0,0.0],math.radians(90*(i+1)))

		strut = mesh.Mesh(np.concatenate([
		    strut.data,
		    tmpside.data.copy()
		]))

	hex_node_width = (0.5*math.sqrt(3)+cham_factor)*strut_width
	square_node_width = (0.5+cham_factor/math.sqrt(2))*strut_width
	scale(strut,[strut_width,lattice_pitch-hex_node_width-square_node_width,strut_width])
	translate(strut,(0,square_node_width,0))
	strut.rotate([0.0,0.0,1.0],math.radians(45))
	return strut

def squarenode(beam_width, lattice_pitch, chamfer_factor, side_code):
	topcap = np.zeros(12,dtype=mesh.Mesh.dtype)
	side = np.zeros(12,dtype=mesh.Mesh.dtype)
	chamside = np.zeros(12,dtype=mesh.Mesh.dtype)

	center = np.array([0.0, 0.0, beam_width/2.0])

	topbound = np.array([[ 0.5*chamfer_factor*beam_width, beam_width/math.sqrt(2)+chamfer_factor*beam_width/2.0, beam_width/2.0],
						 [-0.5*chamfer_factor*beam_width, beam_width/math.sqrt(2)+chamfer_factor*beam_width/2.0, beam_width/2.0]])
	for i in range(6):
		vec = topbound[-2]
		topbound = np.vstack((topbound,np.dot(rot_90_tform,vec)))

	botbound = np.copy(topbound)
	botbound.T[2] = -botbound.T[2]

	for i in range(8):
		topcap['vectors'][i] = np.vstack((topbound[i],center,topbound[i-1]))
	
	botcap = mesh.Mesh(topcap.copy())
	botcap.rotate([0.0,1.0,0.0],math.radians(180))
	
	struts = []
	for i in range(0,8,2):
		if side_code[i/2] == 1:
			side['vectors'][i] = np.vstack((botbound[i-1],botbound[i],topbound[i]))
			side['vectors'][i+1]= np.vstack((topbound[i],topbound[i-1],botbound[i-1]))
		else:
			tstrut = strut(beam_width,lattice_pitch,chamfer_factor)
			if(i/2 == 1):
				tstrut.vectors.T[0] = -tstrut.vectors.T[0]
			if(i/2 == 2):
				tstrut.rotate([0.0,0.0,1.0],math.radians(180))
			if(i/2 == 3):
				tstrut.vectors.T[1] = -tstrut.vectors.T[1]
			struts.append(mesh.Mesh(tstrut.data.copy()))


	for i in range(-1,8,2):
		chamside['vectors'][i] = np.vstack((botbound[i-1],botbound[i],topbound[i]))
		chamside['vectors'][i+1]= np.vstack((topbound[i],topbound[i-1],botbound[i-1]))

	nodedata = [
	    topcap.copy(),
	    botcap.data.copy(),
	    side.copy(),
	    chamside.copy()
	]

	for struto in struts:
		nodedata.append(struto.data.copy())

	nodecapbase = mesh.Mesh(np.concatenate(nodedata))


	#nodecapbase.rotate([0.0,0.0,1.0],math.radians(45))

	return nodecapbase





def hexnode(beam_width, chamfer_factor, side_code):
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
		if side_code[i/2] == 1:
			side['vectors'][i] = np.vstack((botbound[i-1],botbound[i],topbound[i]))
			side['vectors'][i+1]= np.vstack((topbound[i],topbound[i-1],botbound[i-1]))

	for i in range(-1,11,2):
		chamside['vectors'][i] = np.vstack((botbound[i-1],botbound[i],topbound[i]))
		chamside['vectors'][i+1]= np.vstack((topbound[i],topbound[i-1],botbound[i-1]))

	nodedata = [
	    topcap.copy(),
	    botcap.data.copy(),
	    side.copy(),
	    chamside.copy()
	]

	nodecapbase = mesh.Mesh(np.concatenate(nodedata))


	#nodecapbase.rotate([0.0,0.0,1.0],math.radians(45))

	return nodecapbase

def pitch_from_relden(relden, cham_factor, strut_width):
	bsv = 0.74658 #mm^3
	#calculate the volume of the struts
	hex_node_width = (0.5*math.sqrt(3)+cham_factor)*strut_width
	hex_node_vol = ((1+cham_factor)*1.5*math.sqrt(3)+6*cham_factor)*strut_width*strut_width*strut_width
	square_node_width = (0.5+cham_factor/math.sqrt(2))*strut_width
	square_node_vol = (1.0+4.0*cham_factor+2.0*cham_factor*cham_factor)*strut_width*strut_width*strut_width

	c1 = relden*16.0*np.sqrt(2.0)
	c2 = -24*bsv*strut_width*strut_width
	c3 = -8*hex_node_vol-12*square_node_vol
	#print(np.roots[c1,0,c2,c3])
	return max(np.roots([c1,0,c2,c3]))

##
## IMPORTANT LATTICE CHARACTERISTICS
##

# chamfer factor (so the failure points isn't the nodes)
cham_factor = 0.3 
# distance between the nodes
# width of the struts
min_width = 0.62
strut_width = min_width/0.6504 #mm
relden = 0.05

lattice_pitch = pitch_from_relden(relden,cham_factor,strut_width)
unit_cell = 2*np.sqrt(2)*lattice_pitch


###
### MATERIAL MATRIX ET AL
###

# This is where we store our triangles for output
latticedata = []

# 0 means there is no material there
# usually a mat matrix is 1s surrounded on all sides by
# non-ones

size = 10
matmatrix = np.zeros((size,size,size))

for i in range(1,size-1):
	for j in range(1,size-1):
		for k in range(1,size-1):
			matmatrix[i][j][k] = 1

for i in range(1,size-1):
	for j in range(1,size-1):
		matmatrix[i][j][0] = -1
		matmatrix[i][j][size-1] = -1




latticedata = []
for xdex in range(len(matmatrix)):
	for ydex in range(len(matmatrix[0])):
		for zdex in range(len(matmatrix[0][0])):
			if matmatrix[xdex][ydex][zdex] == 1:

				## First checking for caps on the hex nodes
				## a -1 in the mat matrix implies fixturing
				## therefore we want a clean cut plane
				hexcapmap = [0,0,0,0,0,0]
				whlcapmap = [0,0,0,0,0,0]
				halfcapmap =[0,0,0,0,0,0]

				if matmatrix[xdex+1][ydex][zdex] == -1:
					hexcapmap[0] = 1				
				if matmatrix[xdex-1][ydex][zdex] == -1:
					hexcapmap[1] = 1
				if matmatrix[xdex][ydex+1][zdex] == -1:
					hexcapmap[2] = 1
				if matmatrix[xdex][ydex-1][zdex] == -1:
					hexcapmap[3] = 1
				if matmatrix[xdex][ydex][zdex+1] == -1:
					hexcapmap[4] = 1				
				if matmatrix[xdex][ydex][zdex-1] == -1:
					hexcapmap[5] = 1


				if matmatrix[xdex+1][ydex][zdex] == 0:
					halfcapmap[0] = 1
				if matmatrix[xdex][ydex+1][zdex] == 0:
					halfcapmap[1] = 1
				if matmatrix[xdex-1][ydex][zdex] == 0:
					halfcapmap[2] = 1
				if matmatrix[xdex][ydex-1][zdex] == 0:
					halfcapmap[3] = 1
				if matmatrix[xdex][ydex][zdex+1] == 0:
					halfcapmap[4] = 1
				if matmatrix[xdex][ydex][zdex-1] == 0:
					halfcapmap[5] = 1

				if matmatrix[xdex+1][ydex][zdex] == 1:
					whlcapmap[0] = 1
				if matmatrix[xdex][ydex+1][zdex] == 1:
					whlcapmap[1] = 1
				if matmatrix[xdex][ydex][zdex+1] == 1:
					whlcapmap[4] = 1
				
				
				
				## Generating the side codes for the hex nodes
				## see the above matrices for a sense of how they
				## map
				## NB: probably a more efficient way to do that.
				hexcapvals = np.zeros((8,6))
				for hexdex in range(8):
					for facedex in range(6):
						for hexcapdex in range(len(hexcaprefs)):
							hexcapvals[hexdex][facedex] = hexcapvals[hexdex][facedex]+hexcapmap[hexcapdex]*hexcaprefs[hexcapdex][hexdex][facedex]
				
				## Generating the hex nodes from the side codes
				hexnodes = []
				for i in range(8):
					hexnodes.append(hexnode(strut_width,cham_factor,hexcapvals[i]))
					hexnodes[i].rotate([0.0,0.0,1.0],math.radians(30))
					hexnodes[i].rotate([0.0,1.0,0.0],math.radians(54.736))
					hexnodes[i].rotate([0.0,0.0,1.0],math.radians(-45))
					translate(hexnodes[i],[-0.25*unit_cell,-0.25*unit_cell,0.25*unit_cell])

				for i in range(3):
					hexnodes[i+1].rotate([0.0,0.0,1.0],math.radians(90*(i+1)))
					hexnodes[i+5].rotate([0.0,0.0,1.0],math.radians(90*(i+1)))

				for i in range(4):
					hexnodes.append(mesh.Mesh(hexnodes[i].data.copy()))
					hexnodes[i+4].vectors.T[2] = -hexnodes[i+4].vectors.T[2]

				newhexnodes = []

				for thexnode in hexnodes:
					newhexnodes.append(thexnode.data.copy())

				hexnodes = newhexnodes
				#hexnodes.append(hex1.data.copy())
				hexnodes = np.concatenate(hexnodes)
				hexnodes = stl.mesh.Mesh(hexnodes)

				## Square node sets
				## one half set and one whole set
				partialxset = []
				partialx = mesh.Mesh(squarenode(strut_width,lattice_pitch,cham_factor,[1,1,0,0]).data.copy())
				partialx.rotate([0.0,0.0,1.0],math.radians(90))
				translate(partialx,[unit_cell/2.0,0,unit_cell/4.0])
				for i in range(4):
					partialx.rotate([1.0,0.0,0.0],math.radians(90))
					partialxset.append(partialx.data.copy())
				
				partialxset = np.concatenate(partialxset)
				partialxset = stl.mesh.Mesh(partialxset)

				wholexset = []
				wholex = mesh.Mesh(squarenode(strut_width,lattice_pitch,cham_factor,[0,0,0,0]).data.copy())
				wholex.rotate([0.0,0.0,1.0],math.radians(90))
				translate(wholex,[unit_cell/2.0,0,unit_cell/4.0])
				for i in range(4):
					wholex.rotate([1.0,0.0,0.0],math.radians(90))
					wholexset.append(wholex.data.copy())
				
				wholexset = np.concatenate(wholexset)
				wholexset = stl.mesh.Mesh(wholexset)

				xs = []
				for i in range(4):
					if halfcapmap[i] == 1:
						xs.append(partialxset.data.copy())
					if whlcapmap[i] == 1:
						xs.append(wholexset.data.copy())
					partialxset.rotate([0.0,0.0,1.0],math.radians(-90))
					wholexset.rotate([0.0,0.0,1.0],math.radians(-90))

				partialxset.rotate([0.0,1.0,0.0],math.radians(90))
				wholexset.rotate([0.0,1.0,0.0],math.radians(90))
				if halfcapmap[4] == 1:
					xs.append(partialxset.data.copy())
				if whlcapmap[4] == 1:
					xs.append(wholexset.data.copy())

				partialxset.rotate([0.0,1.0,0.0],math.radians(180))
				if halfcapmap[5] == 1:
					xs.append(partialxset.data.copy())

				xs.append(hexnodes.data.copy())
				xs = np.concatenate(xs)
				xs = stl.mesh.Mesh(xs)
				translate(xs,[xdex*unit_cell,ydex*unit_cell,zdex*unit_cell])
				latticedata.append(xs.data.copy())

latticedata = np.concatenate(latticedata)
latticedata = stl.mesh.Mesh(latticedata)


latticedata.save('{0}_{0}_{0}_psch_{1}rd_{2}sw.stl'.format(size-2,relden,min_width))

'''

# Optionally render the rotated cube faces
from matplotlib import pyplot
from mpl_toolkits import mplot3d

# Create a new plot
figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)

axes.set_xlim(-unit_cell/2,unit_cell/2)
axes.set_ylim(-unit_cell/2,unit_cell/2)
axes.set_zlim(-unit_cell/2,unit_cell/2)

#for i in range(6):
	#axes.add_collection3d(mplot3d.art3d.Poly3DCollection(struts[i].vectors))

# Render the cube

#for ucmeshes in meshes:
#	for meshset in ucmeshes:
#		for mesh in meshset:
#			axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))

axes.add_collection3d(mplot3d.art3d.Poly3DCollection(latticedata.vectors))

# Show the plot to the screen
pyplot.show()
'''