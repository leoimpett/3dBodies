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
    """get all the bodies from a painting and return them as an array of arrays of points and corresponding limbs
    which represent each body."""
    subset   = painting[2]
    peaks    = painting[4]
    ignored = [5, 12, 16, 17, 18, 19]
    
    #get all the points in an image
    points = list();
    for i in range(18):
        for j in range(len(peaks[i])):
            points.append(peaks[i][j][0:2])
            
    #get all the corresponding limbs
    limbs = np.zeros((len(subset), 17, 2))
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i])-1]
            if -1 in index:
                continue
            limbs[n][i] = index
    
    bodies = list()
    for b in limbs:
        body_points = np.zeros((20,2), np.int16)
        body_limbs  = np.zeros((17,2), np.int16)
        seen_points = list()
        for i in range(len(b)):
            if not b[i][0] == b[i][1]:
                body_limbs[i] = original_limbs[i]
                if not b[i][0] in seen_points:
                    body_points[body_limbs[i][0]] = points[int(b[i][0])]
                    seen_points.append(b[i][0])
                if not b[i][1] in seen_points:
                    body_points[body_limbs[i][1]] = points[int(b[i][1])]
                    seen_points.append(b[i][1])
        
        
        pts = []
        for p in body_points:
            pts.append(Point(p[0], p[1]))
        lbs = []
        for l in body_limbs:
            lbs.append(Limb(pts[l[0]], pts[l[1]]))
        
        bodies.append(Body(pts, lbs, p_id, ignored))
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


def resize_bodies(bodies, mean):
    """return all the bodies resized to a mean size"""
    res = list()
    for b in bodies:
        scale = b.compute_scale(mean)
        res.append(b.scale(scale))
    return res

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

def middle_mean_length(bodies):
    n = 0
    s = 0.0
    for b in bodies:
        l = b.middle_length()
        if not l == 0:
            n+= 1
            s += l
    return s/n

def compute_dev(bodies):
    count = 0
    tot = 0.0
    for b in bodies:
        l1 = b.limbs[6]
        l2 = b.limbs[9]
        if l1.valid() and l2.valid():
            tot += (l2.angle() - l1.angle())
            count += 1
    return tot/count
    
def all_relative_angles(bodies):
    """compute relative angles of all the bodies amd return a list with them"""
    angles = list()
    dev = compute_dev(bodies)
    for b in bodies:
        angles.append(b.relative_angles(dev))
    return angles