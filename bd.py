import numpy as np
from Point import Point
from Limb import Limb
from Body import Body


limbSeq  = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
               [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
               [1,16], [16,18], [3,17], [6,18]]

original_limbs = [[1,2],[1,6],[2,3],[3,4],[6,7],[7,8],[1,9],[9,10],[10,11],\
                   [1,13],[13,14],[14,15],[1,0],[0,16],[16,18],[0,17],[17,19]]


def get_bodies_from_picture(painting, p_id):
    """get all the bodies from a painting and return them as an array of arrays
    of points and corresponding limbs which represent each body."""
    subset   = painting[2]
    peaks    = painting[4]
    ignored = [5, 12, 16, 17, 18, 19]
    
    #get all the points in an image
    points = list();
    for i in range(18):
        for j in range(len(peaks[i])):
            p = Point(peaks[i][j][0], peaks[i][j][1])
            if p == None:
                p = Point(0,0).invalidate()
            points.append(p)
    
    
    #get all the corresponding limbs in an image
    limbs = [[Limb(Point(0,0).invalidate(), Point(0,0).invalidate())]*17]*len(subset)
    for n in range(len(subset)):
        for  i in range(17):
            index = subset[n][np.array(limbSeq[i])-1]
            p1, p2 = Point(0,0).invalidate(), Point(0,0).invalidate()
            if not int(index[0]) == -2:
                p1 = points[int(index[0])]
            if not int(index[1]) == -2:
                p2 = points[int(index[1])]
            limbs[n][i] = Limb(p1,p2)
    
    #construct all the bodies
    bodies = list()
    for b in limbs:
        body_points = [Point(0,0).invalidate()]*20
        body_limbs = list()
        seen_points = list()
        for i in range(len(b)):
            body_limbs.append(b[i])
            pts = b[i].get_points()
            for j in range(2):
                if not pts[j] in seen_points:
                    body_points[original_limbs[i][j]] = pts[j]
                    seen_points.append(pts[j])
        
        bodies.append(Body(body_points, body_limbs, p_id, ignored))
    
    return bodies

def construct_body_list(n, paintings):
    """Construct a list with all the bodies of the n first paintings in the database."""
    bodies = list()
    i = 1
    while i <= n: 
        new_bodies = get_bodies_from_picture(paintings[i], i)
        for b in new_bodies:
            bodies.append(b)
        i += 1
    return bodies


def all_bodies_mean_limb_length(bodies):
    """compute the mean length of each limb of body (i.e. the mean length of a right leg)"""
    n_limbs = np.zeros(17, np.int16)
    limb_length_sum = np.zeros(17)
    for b in bodies:
        l = b.limbs
        for i in range(len(l)):
            if l[i].valid():
                n_limbs[i] += 1
                limb_length_sum[i] += l[i].length()
    mean_lengths = np.zeros(17)
    for i in range(17):
        if not n_limbs[i] == 0:
            mean_lengths[i] = limb_length_sum[i] / n_limbs[i]
    return mean_lengths

