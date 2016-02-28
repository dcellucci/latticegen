import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

######  #######  #####   #####  ######  ### ######  ####### ### ####### #     # 
#     # #       #     # #     # #     #  #  #     #    #     #  #     # ##    # 
#     # #       #       #       #     #  #  #     #    #     #  #     # # #   # 
#     # #####    #####  #       ######   #  ######     #     #  #     # #  #  # 
#     # #             # #       #   #    #  #          #     #  #     # #   # # 
#     # #       #     # #     # #    #   #  #          #     #  #     # #    ## 
######  #######  #####   #####  #     # ### #          #    ### ####### #     #

# (111) Lattice generation takes a hexagon size and height and outputs
# a material matrix in cubic coordinates that, when transformed to the
# (111) coordinate system, fills this hexagonal volume.

# The (111) coordinate system is defined by the cubic coordinate system
# vectors listed in the columns of the transform matrix shown below. 

tform111 = np.array([[ 1.0/np.sqrt(2) , -1.0/np.sqrt(6), 1/np.sqrt(3)],
					 [-1.0/np.sqrt(2) , -1.0/np.sqrt(6), 1/np.sqrt(3)],
					 [ 			  0.0 ,np.sqrt(2.0/3.0), 1/np.sqrt(3)]])

tform111 = np.linalg.inv(tform111)

# The hexagonal volume is defined as being a certain number of unit cells
# XY plane view
#        /\y
#      __||__
#     /\ || /\
#    /  \||/  \
#   /____\/O___\__>x
#   \ r  /\    /
#    \  /  \  /
#     \/____\/

# A unit cell in the hexagon is a triangular unit, whose side length
# is related to the cube dimension by a factor of sqrt(2). 
# Since the unit cell has side length 1:

hex_triangle_width = np.sqrt(2)

# In the z-direction, the units are in cube-diagonal lengths, or

hex_base_height = np.sqrt(3)

# Two utilities simplify checking whether a point is in this
# hexagonal volume, a 30 and 60 degree rotation about the [0,0,1] axis

# Transform matrix that rotates around the [0,0,1] axis by 30 degrees
init_hex_rot_tform = 0.5*np.array([[np.sqrt(3),        -1,0],
						           [         1,np.sqrt(3),0],
						           [         0,         0,2]])

# Transform matrix that rotates around the [0,0,1] axis by 60 degrees
hex_rot_tform = 0.5*np.array([[         1,-np.sqrt(3),0],
						      [np.sqrt(3),          1,0],
						      [         0,          0,2]])


hex_radius = 3
hex_height = 3




#
#creation of the hex volume wireframe
#
#first node
hexnode1 = np.array([hex_radius*hex_triangle_width,0,0])

hex_nodes = np.array([hexnode1])

for i in range(5):
	hexnode1 = np.dot(hex_rot_tform,hexnode1)
	hex_nodes = np.append(hex_nodes,[np.copy(hexnode1)],axis=0)

hex_nodes = np.append(hex_nodes,hex_nodes+np.array([0,0,hex_base_height*hex_height]),axis=0)

hex_frames = [[ 0, 1],
			  [ 0, 6],
			  [ 1, 2],
			  [ 1, 7],
			  [ 2, 3],
			  [ 2, 8],
			  [ 3, 4],
			  [ 3, 9],
			  [ 4, 5],
			  [ 4,10],
			  [ 5, 0],
			  [ 5,11],
			  [ 6, 7],
			  [ 7, 8],
			  [ 8, 9],
			  [10,11],
			  [11, 6]]



cube = [[0.0,0.0,0.0], #0
		[1.0,0.0,0.0], #1
		[0.0,1.0,0.0], #2
		[0.0,0.0,1.0], #3
		[1.0,1.0,0.0], #4
		[0.0,1.0,1.0], #5
		[1.0,0.0,1.0], #6
		[1.0,1.0,1.0]] #7

frames = [[0,1],
		  [0,2],
		  [0,3],
		  [1,4],
		  [1,6],
		  [2,4],
		  [2,5],
		  [3,5],
		  [3,6],
		  [4,7],
		  [5,7],
		  [6,7]]





nodes = np.transpose(np.dot(tform111,np.transpose(cube)))

#print(frames,np.shape(nodes))
#Set up figure plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#nodes = np.array(nodes)
xs = np.array(nodes.T[0])
ys = np.array(nodes.T[1])
zs = np.array(nodes.T[2])

hxs = np.array(hex_nodes.T[0])
hys = np.array(hex_nodes.T[1])
hzs = np.array(hex_nodes.T[2])

#This maintains proper aspect ratio for the 3d plot
max_range = np.array([hxs.max()-hxs.min(), hys.max()-hys.min(), hzs.max()-hzs.min()]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(hxs.max()+hxs.min())
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(hys.max()+hys.min())
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(hzs.max()+hzs.min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
   ax.plot([xb], [yb], [zb], 'w')

#plot all of the frames
for i,frame in enumerate(frames):
		nid1 = int(frame[0])
		nid2 = int(frame[1])
		start = [xs[nid1],ys[nid1],zs[nid1]]
		end   = [xs[nid2],ys[nid2],zs[nid2]]
		ax.plot([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],color='r', alpha=0.1)

#plot all of the frames
for i,frame in enumerate(hex_frames):
		nid1 = int(frame[0])
		nid2 = int(frame[1])
		start = [hxs[nid1],hys[nid1],hzs[nid1]]
		end   = [hxs[nid2],hys[nid2],hzs[nid2]]
		ax.plot([start[0],end[0]],[start[1],end[1]],[start[2],end[2]],color='b', alpha=0.1)


#plot the nodes
ax.scatter(xs,ys,zs)
ax.scatter(hxs,hys,hzs,color='r')

#show it
plt.show()