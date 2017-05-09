import numpy as np
import Point
import Limb



limbSeq  = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
               [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
               [1,16], [16,18], [3,17], [6,18]]

original_limbs = [[1,2],[1,6],[2,3],[3,4],[6,7],[7,8],[1,9],[9,10],[10,11],\
                   [1,13],[13,14],[14,15],[1,0],[0,16],[16,18],[0,17],[17,19]]


def get_bodies_from_picture(painting, p_id):
    """get all the bodies from a painting and return them as an array of arrays of points and corresponding limbs
    which represent each body.
    
    the format of output is:
    list(bodies)
    body = (points, limbs)"""
    subset   = painting[2]
    peaks    = painting[4]
    ignored = [5, 12, 16, 17, 18, 19]
    
    #get all the points in an image
    points = list();
    for i in range(18):
        for j in range(len(peaks[i])):
            p = Point(peaks[i][j][0], peaks[i][j][1])
            points.append(p)
    

    #get all the corresponding limbs in an image
    limbs = np.empty((len(subset), 17), dtype=Limb)
    for n in range(len(subset)):
        for  i in range(17):
            index = subset[n][np.array(limbSeq[i])-1]
            if -1 in index:
                limbs[n][i] = Limb(Point(0,0).invalidate, Point(0,0).invalidate)
            else:
                limbs[n][i] = Limb(points[index[0]], points[index[1]])
            
    

    #construct all the bodies
    bodies = list()
    for b in limbs:
        body_points = np.empty(20, dtype=Point)
        body_limbs = np.empty(17, dtype =Limb)
        seen_points = list()
        for i in range(len(b)):
            body_limbs[i] = b[i]
            pts = b[i].get_points()
            for j in range(2):
                if not pts[j] in seen_points:
                    body_points[original_limbs[i][j]] = pts[j]
                    seen_points.append(pts[j])
        
        bodies.append((body_points, body_limbs, p_id, ignored))
    
    return bodies