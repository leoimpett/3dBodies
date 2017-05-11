import numpy as np
import math
import sklearn
import sklearn.neighbors

from Point import Point
from Limb import Limb
from Body import Body


limbSeq  = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
               [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
               [1,16], [16,18], [3,17], [6,18]]

original_limbs = [[1,2],[1,6],[2,3],[3,4],[6,7],[7,8],[1,9],[9,10],[10,11],\
                   [1,13],[13,14],[14,15],[1,0],[0,16],[16,18],[0,17],[17,19]]

base_angles = [180., 0., 180., 180., 0., 0., 105., 90.,\
               90., 75., 90., 90., -90., -135., -170., -45., -10.]

ignored = [5, 12, 16, 17, 18, 19]

def angles():
    return base_angles

def bound(a):
    """bounds an angle between -180 and +180"""
    while a > 180:
        a -= 360
    while a <= -180:
        a += 360
    return a


def get_bodies_from_picture(painting, p_id):
    """get all the bodies from a painting and return them as an array of arrays of points and corresponding limbs
    which represent each body."""
    subset   = painting[2]
    peaks    = painting[4]
    
    
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


def angles_to_body(angles, ll, center):
    """constructs a body from limb length and angles of limbs"""
    pts = [Point(0,0).invalidate()] * 20
    pts[1] = center
    lbs = original_limbs
    limbs = list()
    for i in range(17):
        p1 = pts[lbs[i][0]]
        l = ll[i]
        a = angles[i]
        dx, dy = int(math.cos(math.pi * a/180) * l), int(math.sin(math.pi * a/180) * l)
        p2 = p1.translate(dx,dy)
        pts[lbs[i][1]] = p2
        limbs.append(Limb(pts[lbs[i][0]], pts[lbs[i][1]]))
    return Body(pts, limbs, 0, ignored)
    
def mean_angle(angles):
    """compute the mean angle of an angle list."""
    s = 0.0
    c = 0.0
    n = 0
    for a in angles:
        s += math.sin(math.radians(a))
        c += math.cos(math.radians(a))
        n += 1
    return math.degrees(math.atan2(s,c))


def compute_mean_relative_angles(relative_angles):
    """compute the mean relative angle of each limb"""
    angles = np.transpose(relative_angles)
    mean = np.zeros(len(angles))
    for i in range(len(angles)):
        mean[i] = mean_angle(angles[i])
    return mean


def compute_std_deviation(relative_angles):
    """compute the standard deviation of an array of relative angles"""
    
    means = compute_mean_relative_angles(relative_angles)
    angles = np.transpose(relative_angles)
    std = np.zeros(len(angles))
    for i in range(len(angles)):
        m = means[i]
        n = 0
        s = 0.0
        for a in angles[i]:
            if a != 360:
                s += abs(a-m)**2
                n+=1
        std[i] = math.sqrt(s/n)
    return std


def angles_distance(a1, a2, deviation):
    """compute the distance between two arrays of relative angles. a1 has only valid values but a2
    may have invalid values (360)"""
    if len(a1) != len(a2):
        return -1
    s = 0.0
    for i in range(len(a1)):
        if a2[i] == 360:
            s += deviation[i]**2
        else:
            s += bound(a2[i]-a1[i]) * bound(a2[i]-a1[i])
    return math.sqrt(s)

def keep_angles(angles, to_keep):
    """keeps only angles of interest"""
    keep_angles = list()
    if not len(to_keep) == 6:
        return angles
    tr = np.transpose(angles)
    if to_keep[0]:
        keep_angles.append(tr[0])
        keep_angles.append(tr[1])
    if to_keep[1]:
        keep_angles.append(tr[2])
        keep_angles.append(tr[3])
    if to_keep[2]:
        keep_angles.append(tr[4])
        keep_angles.append(tr[5])
    if to_keep[3]:
        keep_angles.append(tr[6])
        keep_angles.append(tr[7])
    if to_keep[4]:
        keep_angles.append(tr[8])
    if to_keep[5]:
        keep_angles.append(tr[9])
    
    return np.transpose(keep_angles)
    

def get_n_nearest_neighbor(angles, body, deviation, to_keep, n=100, dist=50):
    """gets the n nearest neighbors of a body"""
    angles = keep_angles(angles, to_keep)
    body_angles = keep_angles([body.relative_angles(deviation)], to_keep)[0]
    def angles_dist(a1, a2):
        """compute the distance between two arrays of relative angles. a1 has only valid values but a2
        may have invalid values (360)"""
        return angles_distance(a1,a2,deviation)
    
    
    NN = sklearn.neighbors.NearestNeighbors(n_neighbors=n, radius=dist, leaf_size=30,
                                             metric=angles_dist, algorithm='auto')
    NN.fit(angles)
    return NN.kneighbors([body_angles])