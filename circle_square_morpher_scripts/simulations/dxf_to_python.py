import numpy as np
import ezdxf
import matplotlib.pyplot as plt 
import sys 

filename = r'./mechanism_shapemorpher_open_v3_clean_dxf.dxf'

doc = ezdxf.readfile(filename)
msp = doc.modelspace()
numcolumns = 8

# iterate over all entities in modelspace
msp = doc.modelspace()
block_dict = {}
key = 0
for e in msp:
    if e.dxftype() == "LWPOLYLINE":
        block_corners = np.array(e.get_points()).T[:2].T
        #order block corners counterclockwise, starting from rightmost point. Does not work for strongly distorted lattices.
        centroid = np.mean(block_corners, axis=0)
        rel_corner_vecs = np.array([bc - centroid for bc in block_corners])
        rel_corner_angles = np.array([np.arctan2(rc[1], rc[0]) for rc in rel_corner_vecs])
        # rel_corner_angles[rel_corner_angles < 0] +=2*np.pi 
        centroids_sorted_angle = np.argsort(rel_corner_angles)
        rightmost_idx = np.argmax(rel_corner_vecs.T[0])
        centroids_sorted_angle_startright = np.roll(centroids_sorted_angle, -np.where(centroids_sorted_angle==rightmost_idx)[0][0])
        block_corners_sorted = block_corners[centroids_sorted_angle_startright]
        block_dict.update({key:{'vertices': block_corners_sorted}})
        block_dict[key].update({'centroid': centroid})
        key += 1
        
#sort dictionary by order of patch centroids, starting from bottom left, 
#(x0, y0), (x0, y1).... (x1, y0), (x1, y1),.... et cetera
centroids = np.array([val['centroid'] for key, val in block_dict.items()])

sort_idx = []
centroids_sorted_x = np.argsort(centroids.T[0])
for column_idx in np.split(centroids_sorted_x,numcolumns):
    column_sorted_y = np.argsort(centroids[column_idx].T[1])
    sort_idx.extend(column_idx[column_sorted_y])
sort_idx = np.array(sort_idx)

sorted_block_dict = {i: block_dict[sort_idx[i]] for i in range(len(sort_idx))}

check_sorted_centroids = np.array([val['centroid'] for key, val in sorted_block_dict.items()])
fig, ax = plt.subplots(1,1)
ax.scatter(check_sorted_centroids.T[0], check_sorted_centroids.T[1], c=np.linspace(0, 1, len(centroids)), 
           cmap='Greys', vmin=-0.1, vmax=1.1)
for i, dd in sorted_block_dict.items():
    ordered_corners = dd['vertices']
    center = dd['centroid']
    ax.scatter(ordered_corners.T[0], ordered_corners.T[1], c=np.linspace(0, 0.5, 4),
                                                                         cmap='jet', vmin=-0.1, vmax=1.1)
    ax.plot(ordered_corners.T[0], ordered_corners.T[1])
    ax.text(center[0], center[1], i)

np.save('ordered_blockpatch_dict', sorted_block_dict)

filename = r'./mechanism_shapemorpher_open_v3_clean_dxf_nohinge_reference.dxf'

doc = ezdxf.readfile(filename)
msp = doc.modelspace()

# iterate over all entities in modelspace
msp = doc.modelspace()
block_dict = {}
key = 0
for e in msp:
    if e.dxftype() == "LWPOLYLINE":
        block_corners = np.array(e.get_points()).T[:2].T
        #order block corners counterclockwise, starting from rightmost point. Does not work for strongly distorted lattices.
        centroid = np.mean(block_corners, axis=0)
        rel_corner_vecs = np.array([bc - centroid for bc in block_corners])
        rel_corner_angles = np.array([np.arctan2(rc[1], rc[0]) for rc in rel_corner_vecs])
        # rel_corner_angles[rel_corner_angles < 0] +=2*np.pi 
        centroids_sorted_angle = np.argsort(rel_corner_angles)
        rightmost_idx = np.argmax(rel_corner_vecs.T[0])
        centroids_sorted_angle_startright = np.roll(centroids_sorted_angle, -np.where(centroids_sorted_angle==rightmost_idx)[0][0])
        block_corners_sorted = block_corners[centroids_sorted_angle_startright]
        block_dict.update({key:{'vertices': block_corners_sorted}})
        block_dict[key].update({'centroid': centroid})
        key += 1
        # sys.exit()
        
#sort dictionary by order of patch centroids, starting from bottom left, 
#(x0, y0), (x0, y1).... (x1, y0), (x1, y1),.... et cetera
centroids = np.array([val['centroid'] for key, val in block_dict.items()])

sort_idx = []
centroids_sorted_x = np.argsort(centroids.T[0])
for column_idx in np.split(centroids_sorted_x,numcolumns):
    column_sorted_y = np.argsort(centroids[column_idx].T[1])
    sort_idx.extend(column_idx[column_sorted_y])
sort_idx = np.array(sort_idx)

sorted_block_dict = {i: block_dict[sort_idx[i]] for i in range(len(sort_idx))}

check_sorted_centroids = np.array([val['centroid'] for key, val in sorted_block_dict.items()])
fig, ax = plt.subplots(1,1)
ax.scatter(check_sorted_centroids.T[0], check_sorted_centroids.T[1], c=np.linspace(0, 1, len(centroids)), 
           cmap='Greys', vmin=-0.1, vmax=1.1)
for i, dd in sorted_block_dict.items():
    center = dd['centroid']
    ordered_corners = dd['vertices']
    ax.scatter(ordered_corners.T[0], ordered_corners.T[1], c=np.linspace(0, 0.5, 4),
                                                                         cmap='jet', vmin=-0.1, vmax=1.1)
    ax.plot(ordered_corners.T[0], ordered_corners.T[1])
    ax.text(center[0], center[1], i)

np.save('ordered_reference_blockpatch_dict', sorted_block_dict)

