import numpy as np
import math
import cv2 as cv

import bqplot as bqp
from ipywidgets import interact


limbSeq  = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
               [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
               [1,16], [16,18], [3,17], [6,18]]
    
original_limbs = [[1,2],[1,6],[2,3],[3,4],[6,7],[7,8],[1,9],[9,10],[10,11],\
                   [1,13],[13,14],[14,15],[1,0],[0,16],[16,18],[0,17],[17,19]]
    

def limb_valid(limb):
    """Control that a single limb is valid"""
    p1 = int(limb[0])
    p2 = int(limb[1])
    if(p1 == p2):
        return False
    return True

def dist_points(a, b):
    """computes the distance between points a and b"""
    return math.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2)


def length_of_limb(points, limb):
    """Compute the length of the ith limb of a body (list of points)"""
    if not limb_valid(limb):
        return 0
    return dist_points(points[int(limb[0])], points[int(limb[1])])


def length_of_middle(body):
    """return the length of the neck of a skeleton"""
    points = body[0]
    limbs = body[1]
    lor = length_of_limb(points, limbs[9])
    lol = length_of_limb(points, limbs[6])
    if lor == 0:
        return lol
    if lol == 0:
        return lor
    return (lor + lol)/2.


def limbs_all_valid(limbs):
    """Control that all the limbs in arguments are valid limbs. i.e they exists in the paint
    non-existing points are written as [0  0] so if the two values of a limb are equals the limbs is not valid"""
    for l in limbs:
        if not limb_valid(l):
            return False
    return True


def get_body_limbs(limbs):
    """return the body limbs: shoulder limbs and middle limbs"""
    return (limbs[0], limbs[1], limbs[9], limbs[6])


def translate_skeleton(points, ref_point):
    """translate a skeleton from its position to a reference position"""
    p1 = points[1]
    t_vector = (ref_point[0]-p1[0], ref_point[1]-p1[1])
    new_points = list()
    for p in points:
        new_points.append(np.add(p, t_vector))
    return new_points


def scale_skeleton(points, scale, center_point):
    """scale a skeleton"""
    origin_points = translate_skeleton(points, (0,0))
    scaled_points = np.zeros((len(points), 2), np.int16)
    for i in range(len(points)):
        scaled_points[i] = (np.int16(scale * origin_points[i][0]), np.int16(scale * origin_points[i][1]))
    return translate_skeleton(scaled_points, center_point)


def totuple(a):
    """transforms a np array to a tuple for cv.line"""
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


def find_ignored_points(pts):
    """finds the points we have to ignore when computing the mean error between two bodies"""
    ignored = list()
    for i in range(len(pts)):
        #We don't care about the head to compute errors so let ignored the head points
        if np.all(pts[i]==0) or i > 15:
            ignored.append(i)
    return ignored


def get_bodies_from_picture(painting):
    """get all the bodies from a painting and return them as an array of arrays of points and corresponding limbs
    which represent each body.
    
    the format of output is:
    list(bodies)
    body = (points, limbs)"""
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
            if limb_valid(b[i]):
                body_limbs[i] = original_limbs[i]
                if not b[i][0] in seen_points:
                    body_points[body_limbs[i][0]] = points[int(b[i][0])]
                    seen_points.append(b[i][0])
                if not b[i][1] in seen_points:
                    body_points[body_limbs[i][1]] = points[int(b[i][1])]
                    seen_points.append(b[i][1])
        
        bodies.append((body_points, body_limbs, find_ignored_points(body_points)))
    return bodies
                
        


def construct_body_list(n, paintings):
    """Construct a list with all the bodies of the n first paintings in the database."""
    bodies = list()
    i = 1
    while i <= n: 
        bodies = bodies + get_bodies_from_picture(paintings[i])
        i += 1
    return bodies


def all_bodies_mean_limb_length(bodies):
    """compute the mean length of each limb of body (i.e. the mean length of a right leg)"""
    n_limbs = np.zeros(17, np.int16)
    limb_length_sum = np.zeros(17)
    for b in bodies:
        p = b[0]
        l = b[1]
        for i in range(len(l)):
            if limb_valid(l[i]):
                n_limbs[i] += 1
                limb_length_sum[i] += length_of_limb(p, l[i])
    mean_lengths = np.zeros(17)
    for i in range(17):
        if not n_limbs[i] == 0:
            mean_lengths[i] = limb_length_sum[i] / n_limbs[i]
    return mean_lengths


def find_absolute_angles(body):
    """find the absolute angle of each of all the limbs of a body compared 
    to horizontal axis and return a list containing all those angles"""
    points = body[0]
    limbs = body[1]
    angles = list();
    for l in limbs:
        x0 = points[int(l[0])][0]
        y0 = points[int(l[0])][1]
        x1 = points[int(l[1])][0]
        y1 = points[int(l[1])][1]
        angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
        angles.append(angle)
    return angles


def points_to_skeleton(body, color, image):
    """draw a skeleton from a set of points and connections between them"""
    points = body[0]
    limbs = body[1]
    ignored = body[2]
    for i in range(13):
        if not(i == 6 or i == 9): 
            p1 = int(limbs[i][0])
            p2 = int(limbs[i][1])
            #if a point of the limb is ignored, we do not plot it
            if (not p1 in ignored) and (not p2 in ignored):
                cv.line(image, totuple(points[p1]) , totuple(points[p2]), color, 2)
        if(i == 12):
            head_radius = length_of_limb(points, limbs[i]) * 0.6
            center = totuple(points[int(limbs[i][1])])
            cv.circle(image, center, int(head_radius), color, 2)
            
    body_limbs = get_body_limbs(limbs)
    if(limbs_all_valid(body_limbs)):
        body_points = list()
        for bl in body_limbs:
            body_points.append(points[int(bl[1])])
        cv.polylines(image, np.int32([body_points]), True, color, 2)
        
    return image


def angles_to_body(agl, limbs, limbs_length, center_point, ignored):
    """computes the points of a skeleton, given the base point, the limbs' lengths and orientation(angle)"""
    pts = np.zeros((20, 2), np.int16)
     
    pts[int(limbs[0][0])] = center_point
    
    for i in range(17):
        p1 = pts[int(limbs[i][0])]
        l = limbs_length[i]
        a = agl[i]
        delta = (int(math.cos(math.pi * a/180) * l),int(math.sin(math.pi * a/180) * l))
        p2 = (p1[0] + delta[0], p1[1] + delta[1])
        pts[int(limbs[i][1])] = p2
        
        
    return (pts, limbs, ignored)


def mean_error_between_bodies(body1, body2):
    """compute the mean error/distance in pixels between points of two bodies"""
    p1 = body1[0]
    p2 = body2[0]
    ignored_points = body1[2]
    if not len(p1) == len(p2):
        print("bodies must have the same number of points")
        return -1
    total_diff = 0.
    taken_points = 0
    for i in range(len(p1)):
        if i not in ignored_points:
            total_diff += dist_points(p1[i], p2[i])
            taken_points += 1
    return total_diff / taken_points


def compute_scale(body, mll):
    lom = length_of_middle(body)
    if lom == 0:
        return 1
    return ((mll[6]+mll[9])/2) / lom

def resize_all_bodies(bodies, center_point, mean_limbs_lengths):
    """translate and scale all the bodies"""
    u_bodies = list()
    n = 0
    for b in bodies:
        p = b[0]
        l = b[1]
        i = b[2]
        scale = compute_scale(b, mean_limbs_lengths)
        if scale == 1:
            n += 1
        u_bodies.append((scale_skeleton(p, scale, center_point), l, i))
    return u_bodies


def interactive_skeleton(body):
    """Plot an interactive body with which we can play. Only call this function on a complete body (for now)"""
    def refresh(_):
        lines.x, lines.y = [[0, scat.x[1], scat.x[0]],[0, scat.y[1], scat.y[0]]]
    scales = {'x': bqp.LinearScale(min= 0, max= 1000),
             'y' : bqp.LinearScale(min = 1000, max = 0)}
    scat = bqp.Scatter(scales = scales, enable_move = True, update_on_move = True)
    lines = bqp.Lines(scales=scales)
    scat.x , scat.y = [[1000, 0],[1000, 0]]
    lines.x, lines.y = [scat.x,scat.y]
    scat.observe(refresh, names=['x', 'y'])
    return (bqp.Figure(marks=[scat, lines], padding_y = 0., min_height = 750, min_width = 750),1)