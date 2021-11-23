import numpy as np
import math
import cv2 as cv
import matplotlib
from matplotlib import pyplot as plt
import scipy
from scipy import stats
import scipy.spatial
from sklearn.neighbors import NearestNeighbors
import sklearn

from collections import Counter
from collections import OrderedDict
import pandas
import operator

from PIL import Image
from io import StringIO
from skimage import io
import requests

import bqplot as bqp
from ipywidgets import interact


limbSeq  = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
               [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
               [1,16], [16,18], [3,17], [6,18]]
    
original_limbs = [[1,2],[1,6],[2,3],[3,4],[6,7],[7,8],[1,9],[9,10],[10,11],\
                   [1,13],[13,14],[14,15],[1,0],[0,16],[16,18],[0,17],[17,19]]

    

    
def base_angles():
    return [180., 0., 180., 180., 0., 0., 105., 90., 90., 75., 90., 90., -90., -135., -170., -45., -10.]

def base_ignored():
    return [5, 12, 16, 17, 18, 19]

def base_limbs():
    return [[ 1,  2], [ 1,  6], [ 2,  3], [ 3,  4], [ 6,  7], [ 7 , 8], [ 1,  9], [ 9, 10], [10, 11], [ 1, 13], [13, 14],
              [14, 15], [ 1,  0], [ 0, 16], [16, 18], [ 0, 17], [17, 19]]




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


def get_bodies_from_picture(painting, p_id):
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
        
        bodies.append((body_points, body_limbs, find_ignored_points(body_points), p_id))
    return bodies
                
        


def construct_body_list(n, paintings):
    """Construct a list with all the bodies of the n first paintings in the database."""
    bodies = list()
    i = 1
    while i <= n: 
        bodies = bodies + get_bodies_from_picture(paintings[i], i)
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
        
        
    return (pts, limbs, ignored, -1)


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
    for b in bodies:
        p = b[0]
        l = b[1]
        i = b[2]
        b_id = b[3]
        scale = compute_scale(b, mean_limbs_lengths)
        u_bodies.append((scale_skeleton(p, scale, center_point), l, i, b_id))
    return u_bodies

def bound(a):
    """bounds an angle between -180 and +180"""
    while a > 180:
        a -= 360
    while a <= -180:
        a += 360
    return a

def member_relative_angles(body):
    """get the relative angles of the 9 member limbs. Arms first, Left first.
    The arms have relative angles to the clavicle, and the legs to the pelvis (= bassin in french)"""
    angles = find_absolute_angles(body)
    points = body[0]
    limbs = body[1]
    
    #rel_angles = ['?','?','?','?','?','?','?','?']
    rel_angles = np.zeros(9)+360
    
    #left arm 1
    if limb_valid(limbs[2]):
        rel_angles[0] = bound(angles[2]-angles[0])
    #left arm 2
    if limb_valid(limbs[3]):
        rel_angles[1] = bound(angles[3]-angles[2])
    #right arm 1
    if limb_valid(limbs[4]):
        rel_angles[2] = bound(angles[4]-angles[1])
    #right arm 2
    if limb_valid(limbs[5]):
        rel_angles[3] = bound(angles[5]-angles[4])
    
    x0 = points[9][0]
    y0 = points[9][1]
    x1 = points[13][0]
    y1 = points[13][1]
    pelvis_angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
    
    #left leg 1
    if limb_valid(limbs[7]):
        rel_angles[4] = bound(angles[7]-pelvis_angle)
    #left leg 2
    if limb_valid(limbs[8]):
        rel_angles[5] = bound(angles[8]-angles[7])
    #right leg 1
    if limb_valid(limbs[10]):
        rel_angles[6] = bound(angles[10]-pelvis_angle)
    #right leg 2
    if limb_valid(limbs[11]):
        rel_angles[7] = bound(angles[11]-angles[10])
    
    x2 = points[2][0]
    y2 = points[2][1]
    x3 = points[6][0]
    y3 = points[6][1]
    shoulder_angle = math.degrees(math.atan2(y3 - y2, x3 - x2))
    #neck
    if limb_valid(limbs[12]):
        rel_angles[8] = bound(angles[12]-shoulder_angle)
            
    return rel_angles

def get_all_relative_angles(bodies):
    """returns a list with all the relative angles of the list of bodies."""
    relative_angles = list()
    for b in bodies:
        relative_angles.append(member_relative_angles(b))
    return relative_angles

def interactive_body(body, angles, bodies, r_arm = True, l_arm = True, r_leg = True, l_leg = True, neck_p = True):
    """Plot an interactive body with which we can play. Only call this function on a complete body (for now)"""
    def refresh(_):
        i = 0
        to_update = []
        if l_arm:
            left_arm.x, left_arm.y = [[p[2][0], scat.x[i], scat.x[i+1]],[p[2][1], scat.y[i], scat.y[i+1]]]
            to_update.append(3)
            to_update.append(4)
            i+=2
        if r_arm:
            right_arm.x, right_arm.y = [[p[6][0], scat.x[i], scat.x[i+1]],[p[6][1], scat.y[i], scat.y[i+1]]]
            to_update.append(7)
            to_update.append(8)
            i+=2
        if l_leg:
            left_leg.x, left_leg.y = [[p[9][0], scat.x[i], scat.x[i+1]],[p[9][1], scat.y[i], scat.y[i+1]]]
            to_update.append(10)
            to_update.append(11)
            i+=2
        if r_leg:
            right_leg.x, right_leg.y = [[p[13][0], scat.x[i], scat.x[i+1]],[p[13][1], scat.y[i], scat.y[i+1]]]
            to_update.append(14)
            to_update.append(15)
            i+=2
        if neck_p:
            to_update.append(0)
            neck.x, neck.y = [[p[1][0], scat.x[i]],[p[1][1], scat.y[i]]]
        
        head.x = np.cos(np.linspace(0, 2*np.pi, 100))*60+neck.x[1]
        head.y = np.sin(np.linspace(0, 2*np.pi, 100))*65+neck.y[1]
        
        #update body
        for i in range(len(to_update)):
            body[0][to_update[i]] = [scat.x[i], scat.y[i]]
        
        body_angles = []
        b_angles = member_relative_angles(body)
        
        if l_arm:
            body_angles.append(b_angles[0])
            body_angles.append(b_angles[1])
        if r_arm:
            body_angles.append(b_angles[2])
            body_angles.append(b_angles[3])
        if l_leg:
            body_angles.append(b_angles[4])
            body_angles.append(b_angles[5])
        if r_leg:
            body_angles.append(b_angles[6])
            body_angles.append(b_angles[7])
        if neck_p:
            body_angles.append(b_angles[8])
        
        
        b =  get_nearest_neighbor(np.transpose(all_angles), np.transpose(body_angles), bodies)[0]
        p2 = b[0]
        nchest.x, nchest.y = np.transpose([p2[1], p2[2], p2[9], p2[13], p2[6], p2[1], p2[0]])
        
        nhead.x = np.cos(np.linspace(0, 2*np.pi, 100))*60+p2[0][0]
        nhead.y = np.sin(np.linspace(0, 2*np.pi, 100))*65+p2[0][1]
        
        nleft_arm.x, nleft_arm.y = [[p2[2][0], p2[3][0], p2[4][0]],[p2[2][1], p2[3][1], p2[4][1]]]
        nright_arm.x, nright_arm.y = [[p2[6][0], p2[7][0], p2[8][0]],[p2[6][1], p2[7][1], p2[8][1]]]
        nleft_leg.x, nleft_leg.y = [[p2[9][0], p2[10][0], p2[11][0]],[p2[9][1], p2[10][1], p2[11][1]]]
        nright_leg.x, nright_leg.y = [[p2[13][0], p2[14][0], p2[15][0]],[p2[13][1], p2[14][1], p2[15][1]]]
        
        
        
    
    scales = {'x': bqp.LinearScale(min= 0, max= 1000),
             'y' : bqp.LinearScale(min = 1000, max = 0)}
    
    #initialization of the nearest neighbor of the interactive body.
    body_angles = []
    all_angles = []
    b_angles = member_relative_angles(body)
    a = np.transpose(angles)
    
    if l_arm:
        body_angles.append(b_angles[0])
        body_angles.append(b_angles[1])
        all_angles.append(a[0])
        all_angles.append(a[1])
    if r_arm:
        body_angles.append(b_angles[2])
        body_angles.append(b_angles[3])
        all_angles.append(a[2])
        all_angles.append(a[3])
    if l_leg:
        body_angles.append(b_angles[4])
        body_angles.append(b_angles[5])
        all_angles.append(a[4])
        all_angles.append(a[5])
    if r_leg:
        body_angles.append(b_angles[6])
        body_angles.append(b_angles[7])
        all_angles.append(a[6])
        all_angles.append(a[7])
    if neck_p:
        body_angles.append(b_angles[8])
        all_angles.append(a[8])
        
    nbody = get_nearest_neighbor(np.transpose(all_angles), np.transpose(body_angles), bodies)[0]
    
    #points of the interactive and neighbor bodies
    p = body[0]
    p2 = nbody[0]
    
    marks = []
    #Constructions of the two bodies: the interactive one and its nearest neighbor. body part beginnig with n... are part of the neighbor.
    #draw the chest
    chest = bqp.Lines(scales=scales)
    chest.x, chest.y = np.transpose([p[1], p[2], p[9], p[13], p[6], p[1]])
    marks.append(chest)
    
    nchest = bqp.Lines(scales=scales,  colors=['red'])
    nchest.x, nchest.y = np.transpose([p2[1], p2[2], p2[9], p2[13], p2[6], p2[1], p2[0]])
    marks.append(nchest)
    
    #draw the head
    head = bqp.Lines(scales=scales)
    head.x = np.cos(np.linspace(0, 2*np.pi, 100))*60+p[0][0]
    head.y = np.sin(np.linspace(0, 2*np.pi, 100))*65+p[0][1]
    marks.append(head)
    
    
    nhead_x = np.cos(np.linspace(0, 2*np.pi, 100))*60+p2[0][0]
    nhead_y = np.sin(np.linspace(0, 2*np.pi, 100))*65+p2[0][1]
    nhead = bqp.Lines(x=nhead_x, y=nhead_y, scales=scales,  colors=['red'])
    marks.append(nhead)
    
    to_keep=list()
    if l_arm:
        to_keep.append(p[3])
        to_keep.append(p[4])
    if r_arm:
        to_keep.append(p[7])
        to_keep.append(p[8])
    if l_leg:
        to_keep.append(p[10])
        to_keep.append(p[11])
    if r_leg:
        to_keep.append(p[14])
        to_keep.append(p[15])
    if neck_p:
        to_keep.append(p[0])
    #movable points: arms first, left side first
    scat = bqp.Scatter(scales = scales, enable_move = True, update_on_move = True, stroke_width = 7)
    scat.x , scat.y = np.transpose(to_keep)
    marks.append(scat)
    
    i = 0
    left_arm = bqp.Lines(scales=scales)
    if l_arm:
        left_arm.x, left_arm.y = [[p[2][0], scat.x[i], scat.x[i+1]],[p[2][1], scat.y[i], scat.y[i+1]]]
        i +=2
    else:
        left_arm.x, left_arm.y = [[p[2][0], p[3][0], p[4][0]],[p[2][1], p[3][1],p[4][1]]]
    marks.append(left_arm)
    
    nleft_arm = bqp.Lines(scales=scales, colors=['red'])
    nleft_arm.x, nleft_arm.y = [[p2[2][0], p2[3][0], p2[4][0]],[p2[2][1], p2[3][1], p2[4][1]]]
    marks.append(nleft_arm)
    
    right_arm = bqp.Lines(scales=scales)
    if r_arm:
        right_arm.x, right_arm.y = [[p[6][0], scat.x[i], scat.x[i+1]],[p[6][1], scat.y[i], scat.y[i+1]]]
        i+=2
    else:
        right_arm.x, right_arm.y = [[p[6][0], p[7][0], p[8][0]],[p[6][1], p[7][1],p[8][1]]]
    
    marks.append(right_arm)
    
    nright_arm = bqp.Lines(scales=scales,  colors=['red'])
    nright_arm.x, nright_arm.y = [[p2[6][0], p2[7][0], p2[8][0]],[p2[6][1], p2[7][1], p2[8][1]]]
    marks.append(nright_arm)
    
    
    left_leg = bqp.Lines(scales=scales)
    if l_leg:
        left_leg.x, left_leg.y = [[p[9][0], scat.x[i], scat.x[i+1]],[p[9][1], scat.y[i], scat.y[i+1]]]
        i+=2
    else:
        left_leg.x, left_leg.y = [[p[9][0], p[10][0], p[11][0]],[p[9][1], p[10][1],p[11][1]]]
    marks.append(left_leg)
    
    nleft_leg = bqp.Lines(scales=scales,  colors=['red'])
    nleft_leg.x, nleft_leg.y = [[p2[9][0], p2[10][0], p2[11][0]],[p2[9][1], p2[10][1], p2[11][1]]]
    marks.append(nleft_leg)
    
    
    right_leg = bqp.Lines(scales=scales)
    if r_leg:
        right_leg.x, right_leg.y = [[p[13][0], scat.x[i], scat.x[i+1]],[p[13][1], scat.y[i], scat.y[i+1]]]
        i+=2
    else:
        right_leg.x, right_leg.y = [[p[13][0], p[14][0], p[15][0]],[p[13][1], p[14][1], p[15][1]]]
    marks.append(right_leg)
    
    
    nright_leg = bqp.Lines(scales=scales,  colors=['red'])
    nright_leg.x, nright_leg.y = [[p2[13][0], p2[14][0], p2[15][0]],[p2[13][1], p2[14][1], p2[15][1]]]
    marks.append(nright_leg)
    
    
    neck =  bqp.Lines(scales=scales)
    if neck_p:
        neck.x, neck.y = [[p[1][0], scat.x[i]],[p[1][1], scat.y[i]]]
        i+=1
    else:
        neck.x, neck.y = [[p[1][0], p[0][0]],[p[1][1], p[0][1]]]
    marks.append(neck)
    scat.observe(refresh, names=['x', 'y'])
    

    return bqp.Figure(marks=marks, 
                      padding_y = 0., min_height = 750, min_width = 750)




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
    invalid = 0
    s = 0.0
    for i in range(len(a1)):
        if a2[i] == 360:
            s += deviation[i]**2
        else:
            s += bound(a2[i]-a1[i]) * bound(a2[i]-a1[i])
    return math.sqrt(s)


def plot_skeleton(body):
    """plot a body in a 1000X1000 image"""
    img =  np.ones((1000, 1000, 3))
    img =  points_to_skeleton(body, (0,0,0), img)
    plt.imshow(img)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 10)
    
def plot_nearest_neighbor(angles, body_angles, bodies):
    """plot the bodies collection's nearest neighbor of a body"""
    plot_skeleton(get_nearest_neighbor(angles, body_angles, bodies)[0])
    
def get_nearest_neighbor(angles, body_angles, bodies):
    """Get the bodies collection's nearest neighbor of a body"""
    tree = scipy.spatial.cKDTree(angles, leafsize=100)
    distance = tree.query(body_angles, k=1, distance_upper_bound=1000)
    return (bodies[distance[1]], distance[1])

def get_n_nearest_neighbor(angles, body_angles, deviation, n=100, dist=50):
    
    def angles_dist(a1, a2):
        """compute the distance between two arrays of relative angles. a1 has only valid values but a2
        may have invalid values (360)"""
        return angles_distance(a1,a2,deviation)
    
    
    NN = sklearn.neighbors.NearestNeighbors(n_neighbors=n, radius=dist, leaf_size=30,
                                             metric=angles_dist, algorithm='auto')
    NN.fit(angles)
    return NN.kneighbors([body_angles])


def get_distant_neighbors(angles, body_angles, deviation, dist=50):
    """get all the bodies id of bodies that are nearer than dist from a certain body"""
    ids = list()
    for i in range(len(angles)):
        if angles_distance(body_angles, angles[i], deviation) < dist:
            ids.append(i)
    return ids


def pose_rarity(body_angles, angles, dist=50):
    """return the rarity ratio of a certain pose in the collection"""
    n = len(get_distant_neighbors(angles, body_angles, dist=dist))
    return float(n)/len(angles)


def plot_n_nearest_neighbors(angles, body_angles, bodies, paintings, deviation, n=5):
    """plot the n nearest neighbor's paintings with skeleton drawn on them with some info on the painting"""
    distances, b = get_n_nearest_neighbor(angles, body_angles, deviation, n)
    bd_i = b[0]
    f, ax = plt.subplots(n,2, figsize=(24,n * 15))
    for i in range(n):
        p_id = bodies[bd_i[i]][3]
        # response = requests.get(paintings[p_id][1])
        # img = np.array(Image.open(StringIO(response.content)))
        img = io.imread(paintings[p_id][1])
        img = points_to_skeleton(bodies[bd_i[i]], (255, 0, 0), img)
        ax[i, 0].imshow(img)
        ax[i, 1].text(0.2,0.5,'Time period: ' + str(paintings[p_id][15]) + '(' +  str(paintings[p_id][8]) + ')', fontsize = 20)
        ax[i, 1].text(0.2,0.3,'Type: ' + str(paintings[p_id][13]), fontsize = 20)
        ax[i, 1].text(0.2,0.7,'School: ' + str(paintings[p_id][14]), fontsize = 20)
        ax[i, 1].text(0.2,0.9,'Form: ' + str(paintings[p_id][12]), fontsize = 20)
        ax[i, 0].axis('off')
        ax[i, 1].axis('off')
    
    plt.show()
    f.set_size_inches(10, 10)
    
    
def get_paintings_from_bodies(bodies_id, bodies, paintings):
    """retrieve the paintings which contains the sublist of bodies"""
    p = list()
    for b in bodies_id:
        p.append(paintings[bodies[b][3]])
    return p

def histograms(paintings, subpaintings, Form=False, Type=False, School=False, Timeline=False):
    """plot an histogram of the paintings' form, type, school and timeline, specify explicitely which you want,
    default values are False
    returns the p-values of each graph"""
    zipped = zip(*paintings)
    subzipped = zip(*subpaintings)
    
    p_values = list()
    
    i = list()
    if Form:
        i.append(12)
    if Type:
        i.append(13)
    if School:
        i.append(14)
    if Timeline:
        i.append(15)
    
    for k in i:
        f = Counter(zipped[k])
        sf = Counter(subzipped[k])
        for key in f.keys():
            f[key] /= float(len(paintings))
            sf[key] /= float(len(subpaintings))
            
        
        f = OrderedDict(sorted(f.items(), key=lambda t: t[0]))
        sf = OrderedDict(sorted(sf.items(), key=lambda t: t[0]))
        
        v1 = f.values()
        v2 = sf.values()
        
        
        p_values.append(stats.ks_2samp(v1, v2)[1])
            
        df = pandas.DataFrame.from_dict(f, orient='index')
        sdf = pandas.DataFrame.from_dict(sf, orient='index')
        ax = df.plot(kind='bar', legend=False, colormap='ocean', position=0.1)
        sdf.plot(kind='bar', ax=ax, legend=False, position = 0.9)
    return p_values