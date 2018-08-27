import stl
from stl import mesh
import numpy as np
import math

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