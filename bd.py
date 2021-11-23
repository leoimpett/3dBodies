import numpy as np
import math
import sklearn
import sklearn.neighbors
from sklearn.neighbors import LSHForest
import time

from collections import Counter
from collections import OrderedDict
import pandas
from scipy import stats
import random
import bqplot as bqp

from PIL import Image
from io import StringIO
import requests

import matplotlib
from matplotlib import pyplot as plt

from Point import Point
from Limb import Limb
from body import Body


limbSeq  = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
               [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
               [1,16], [16,18], [3,17], [6,18]]

original_limbs = [[1,2],[1,6],[2,3],[3,4],[6,7],[7,8],[1,9],[9,10],[10,11],\
                   [1,13],[13,14],[14,15],[1,0],[0,16],[16,18],[0,17],[17,19]]

base_angles = [180., 0., 180., 180., 0., 0., 105., 90.,\
               90., 75., 90., 90., -90., -135., -170., -45., -10.]

ignored = [5, 12, 16, 17, 18, 19]

n_valid_points = 8

def angles():
    return base_angles

def bound(a):
    """bounds an angle between -180 and +180"""
    while a > 180:
        a -= 360
    while a <= -180:
        a += 360
    return a

def filter_paintings(paintings):
    """remove from paintings all the paintings that does not have at least one valid body"""
    new_p = list()
    for p in paintings:
        n = 0
        for b in p[2]:
            if b[-1] >= n_valid_points:
                n+= 1
        if n > 0:
            new_p.append(p)
    return new_p


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

def filter_bodies(bodies):
    """remove from the body list all the bodies that have not at least a certain number of valid points"""
    new_b = list()
    for b in bodies:
        if b.count_valid_points() >= n_valid_points:
            new_b.append(b)
    return new_b


def resize_bodies(bodies, mean):
    """return all the bodies resized to a mean size"""
    res = list()
    for b in bodies:
        scale = b.compute_scale(mean)
        res.append(b.scale(scale).translate(Point(500,300)))
    return res

def all_bodies_mean_limb_length(bodies):
    """compute the mean length of each limb of body (i.e. the mean length of a right leg)"""
    n_limbs = np.zeros(17, np.int32)
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
    body_angles = keep_angles([body.relative_angles(deviation[9])], to_keep)[0]
    def angles_dist(a1, a2):
        """compute the distance between two arrays of relative angles. a1 has only valid values but a2
        may have invalid values (360)"""
        return angles_distance(a1,a2,deviation)
    
    
    NN = sklearn.neighbors.NearestNeighbors(n_neighbors=n, radius=dist, leaf_size=30,
                                             metric=angles_dist, algorithm='auto')
    
    t = time.time()
    NN.fit(angles)
    print time.time()-t
    t = time.time()
    ne = NN.kneighbors([body_angles])
    print time.time()-t
    return ne

def pre_fit(angles, deviation, n=100, dist=50):
    """returns the angles fitted in a NearestNeighbors object"""
    #angles = keep_angles(angles, to_keep)
    def angles_dist(a1, a2):
        """compute the distance between two arrays of relative angles. a1 has only valid values but a2
        may have invalid values (360)"""
        return angles_distance(a1,a2,deviation)
    
    NN = sklearn.neighbors.NearestNeighbors(n_neighbors=n, radius=dist, leaf_size=30,
                                             metric=angles_dist, algorithm='auto')
   
    nn = NN.fit(angles)
    return nn


def nearest_neighbors(n, body, deviation, nn):
    """gets the n nearest neighbors of a body"""
    
    body_angles = body.relative_angles(deviation[9])
    
    ne = nn.kneighbors([body_angles])[:n]
    return ne

def approximate_nearest_neighbors(n, body, deviation, lshf):
    """compute an approximation of the nearest neighbors"""
    b = body.relative_angles(deviation[9])
    return lshf.kneighbors([b], n_neighbors=n)[1]


def write_p_values(ps, approx):
    """write a list of 4 p_values in a file with a painting_id and the number of neighbors"""
    def to_string(l):
        """transforms a list of numbers to a list of string"""
        ls = list()
        for i in l:
            ls.append(str(i))
        return ls
    if approx:
        f = open('approx_p_values.txt', 'a')
    else:
        f = open('p_values.txt', 'a')
    if len(ps) == 0:
        f.close()
        return
    for l in ps:
        l = to_string(l)
        f.write('\t'.join(l))
        f.write('\n')
    f.close()

def read_p_values(approx=False):
    """read a file with p_values stored in it"""
    def to_int_float(l):
        """transform a list of string to a list of (int, int, float, float, ...)"""
        ls = list()
        for i in range(len(l)):
            if i == 0 or i == 1:
                ls.append(int(l[i]))
            else:
                ls.append(float(l[i]))

        return ls
    if approx:
        f = open('approx_p_values.txt', 'r')
    else:
        f = open('p_values.txt', 'r')
    ps = list()
    l = f.readline()
    while l:
        ls = to_int_float(l.split('\t'))
        ps.append(ls)
        l = f.readline()
    f.close()
    return ps


def p_values_from_random(paintings, bodies, angles, deviation, tree, approx):
    """computes 4 p-values for a set of neighbors of a random body from the paintings dataset
    Return n: the number of neighbors, b: the index of the random body, p_values, the 4 p_values computed"""
    #generate random n
    n = random.randint(20,100)
    
    #as base body we want a body that is complete
    body_complete = False
    while not body_complete:
        #get a random body from all the bodies
        b = random.randint(0, len(bodies)-1)
        base_body = bodies[b]
        body_complete = base_body.complete()
    
    
    #get the neighbors and link them to their painting
    if approx:
        neighbors = approximate_nearest_neighbors(n, base_body, deviation, tree)[0]
    else:
        neighbors = nearest_neighbors(n, base_body, deviation, tree)[1][0]
    subpaintings = list()
    for i in neighbors:
        subpaintings.append(paintings[bodies[i].painting])
    
    
    zipped = zip(*paintings)
    subzipped = zip(*subpaintings)
    
    p_values = list()
    
    #indexes of metadata we want
    i = [12,13,14,15]
    
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
    return n,b,p_values


def p_values_random_hypothesis(nb, paintings, bodies, angles, deviation, approx=False):
    """create nb random hypothesis and store them into a text file to keep them"""
    samples = list()
    i = 0
    if approx:
        tree = LSHForest().fit(angles)
    else:
        tree = pre_fit(angles, deviation, n=100, dist=50)
    while i < nb:
        n,b,p = p_values_from_random(paintings, bodies, angles, deviation, tree, approx)
        m = min(p)
        samples.append([n] + [b] + p + [m])
        i += 1
    write_p_values(samples, approx)
    
def plot_hypothesis(b_id, n, bodies, resized_bodies, angles, paintings, deviation, approx=False):
    """plot the bodies found in a hypothesis"""
    body = bodies[b_id]
    if approx:
        tree = LSHForest().fit(angles)
        neighbors = approximate_nearest_neighbors(n, body, deviation, tree)[0]
    else:
        tree = pre_fit(angles, deviation, n=100, dist=50)
        neighbors = nearest_neighbors(n, body, deviation, tree)[1][0]
    
    subpaintings = list()
    
    for i in neighbors:
        subpaintings.append(paintings[bodies[i].painting])
    
    f, ax = plt.subplots(n,1, figsize=(24,n * 9))
    for i in range(n):
        if i < 10:
            pid = subpaintings[i][1]
            response = requests.get(pid)
            try:
                img = np.array(Image.open(StringIO(response.content)))
                bodies[neighbors[i]].draw(img, (255,0,0))
            except:
                print 'painting '+str(pid)+' failed being retreived, its id is: '+str(bodies[neighbors[i]].painting)
                img = np.ones((1000,1000,3))
                resized_bodies[neighbors[i]].draw(img, (1,0,0))
        else:
            img = np.ones((1000,1000,3))
            resized_bodies[neighbors[i]].draw(img, (1,0,0))
        ax[i].imshow(img)
        ax[i].axis('off')
        
    plt.show()
    f.set_size_inches(10, 10)
    

def plot_research(body, n, bodies, resized_bodies, angles, paintings, deviation, approx=False):
    """plot the bodies found in a hypothesis"""
    if approx:
        tree = LSHForest().fit(angles)
        neighbors = approximate_nearest_neighbors(n, body, deviation, tree)[0]
    else:
        tree = pre_fit(angles, deviation, n=100, dist=50)
        neighbors = nearest_neighbors(n, body, deviation, tree)[1][0]
    
    subpaintings = list()
    for i in neighbors:
        subpaintings.append(paintings[bodies[i].painting])
    
    f, ax = plt.subplots(n,1, figsize=(24,n * 9))
    for i in range(n):
        pid = subpaintings[i][1]
        response = requests.get(pid)
        try:
            img = np.array(Image.open(StringIO(response.content)))
            bodies[neighbors[i]].draw(img, (255,0,0))
        except:
            print 'painting '+str(pid)+' failed being retreived, its id is: '+str(bodies[neighbors[i]].painting)
            img = np.ones((1000,1000,3))
            resized_bodies[neighbors[i]].draw(img, (1,0,0))
        ax[i].imshow(img)
        ax[i].axis('off')
        
    plt.show()
    f.set_size_inches(10, 10)

    
    


def histograms(paintings, neighbors, bodies, wrt_paintings=True, Form=True, Type=True, School=True, Timeline=True):
    """plot an histogram of the paintings' form, type, school and timeline, specify explicitely which you want,
    default values are False
    returns the p-values of each graph"""
    dup_paintings = list()
    if wrt_paintings:
        dup_paintings = paintings[1:]
        
    else:
        for i in bodies:
            pid = i.painting
            if not pid == 0:
                dup_paintings.append(paintings[i.painting])
    
    
    subpaintings = list()
    for i in neighbors:
        subpaintings.append(paintings[bodies[i].painting])
    
    
    zipped = zip(*dup_paintings)
    subzipped = zip(*subpaintings)
    
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
            f[key] /= float(len(dup_paintings))
            sf[key] /= float(len(subpaintings))
            
        
        f = OrderedDict(sorted(f.items(), key=lambda t: t[0]))
        sf = OrderedDict(sorted(sf.items(), key=lambda t: t[0]))
        
            
        df = pandas.DataFrame.from_dict(f, orient='index')
        sdf = pandas.DataFrame.from_dict(sf, orient='index')
        ax = df.plot(kind='bar', legend=False, colormap='ocean', position=0.1)
        sdf.plot(kind='bar', ax=ax, legend=False, position = 0.9)
    return    


def interactive_body(body, l_arm = True, r_arm = True, l_leg = True, r_leg = True, neck_p = True, general= True):
    """Plot an interactive body with which we can play."""
    
    middle = Limb(body.pts[9], body.pts[13]).middle()
    to_keep=list()
    to_keep2 = list()
    if l_arm:
        to_keep.append(body.pts[3].to_array())
        to_keep.append(body.pts[4].to_array())
    if r_arm:
        to_keep.append(body.pts[7].to_array())
        to_keep.append(body.pts[8].to_array())
    if l_leg:
        to_keep.append(body.pts[10].to_array())
        to_keep.append(body.pts[11].to_array())
    if r_leg:
        to_keep.append(body.pts[14].to_array())
        to_keep.append(body.pts[15].to_array())
    if neck_p:
        to_keep.append(body.pts[0].to_array())
    if general:
        to_keep2.append(middle.to_array())
        
    def refresh_rot(_):
        middle = Limb(body.pts[9], body.pts[13]).middle()
        if (not abs(scat2.x[0] - scat3.x[0]) < 15) or not (abs(scat2.y[0] - scat3.y[0])< 15):
            o = body.pts[1]
            a = Point(scat2.x[-1], scat2.y[-1]).angle(o) - middle.angle(o)
            body.rotate(a)
            
            middle = Limb(body.pts[9], body.pts[13]).middle()
            
            i = 0
            if l_arm:
                to_keep[i] = body.pts[3].to_array()
                to_keep[i+1] = body.pts[4].to_array()
                i+=2
            if r_arm:
                to_keep[i] = body.pts[7].to_array()
                to_keep[i+1] = body.pts[8].to_array()
                i+=2
            if l_leg:
                to_keep[i] = body.pts[10].to_array()
                to_keep[i+1] = body.pts[11].to_array()
                i+=2
            if r_leg:
                to_keep[i] = body.pts[14].to_array()
                to_keep[i+1] = body.pts[15].to_array()
                i+=2
            if neck_p:
                to_keep[i] = body.pts[0].to_array()
                i+=1
            if general:
                to_keep2[0] = [scat2.x[0], scat2.y[0]]
            
            scat.x , scat.y = np.transpose(to_keep)
            scat2.x, scat2.y = np.transpose(to_keep2)
            scat3.x, scat3.y = scat2.x, scat2.y
            
            
    def refresh(_):
        
        
        chest.x, chest.y = np.transpose([body.pts[1].to_array(), body.pts[2].to_array(), body.pts[9].to_array(), \
                                     body.pts[13].to_array(), body.pts[6].to_array(), body.pts[1].to_array()])
        
        i = 0
        to_update = []
        if l_arm:
            left_arm.x, left_arm.y = [[body.pts[2].x, scat.x[i], scat.x[i+1]],[body.pts[2].y, scat.y[i], scat.y[i+1]]]
            to_update.append(3)
            to_update.append(4)
            i+=2
        if r_arm:
            right_arm.x, right_arm.y = [[body.pts[6].x, scat.x[i], scat.x[i+1]],[body.pts[6].y, scat.y[i], scat.y[i+1]]]
            to_update.append(7)
            to_update.append(8)
            i+=2
        if l_leg:
            left_leg.x, left_leg.y = [[body.pts[9].x, scat.x[i], scat.x[i+1]],[body.pts[9].y, scat.y[i], scat.y[i+1]]]
            to_update.append(10)
            to_update.append(11)
            i+=2
        if r_leg:
            right_leg.x, right_leg.y = [[body.pts[13].x, scat.x[i], scat.x[i+1]],[body.pts[13].y, scat.y[i], scat.y[i+1]]]
            to_update.append(14)
            to_update.append(15)
            i+=2
        if neck_p:
            to_update.append(0)
            neck.x, neck.y = [[body.pts[1].x, scat.x[i]],[body.pts[1].y, scat.y[i]]]
            
        
        head.x = np.cos(np.linspace(0, 2*np.pi, 100))*60+body.pts[0].x
        head.y = np.sin(np.linspace(0, 2*np.pi, 100))*65+body.pts[0].y
        
        
        #update body
        for i in range(len(to_update)):
            body.pts[to_update[i]].x = scat.x[i]
            body.pts[to_update[i]].y =  scat.y[i]
        

        
        return
    
    
    
    scales = {'x': bqp.LinearScale(min= 0, max= 1000),
             'y' : bqp.LinearScale(min = 1000, max = 0)}
    
    marks = []
    
   
    
    #movable points: arms first, left side first
    scat = bqp.Scatter(scales = scales, enable_move = True, update_on_move = True, stroke_width = 7)
    scat.x , scat.y = np.transpose(to_keep)
    if general:
        scat2 = bqp.Scatter(scales = scales, enable_move = True, update_on_move = True, stroke_width = 7)
        scat2.x , scat2.y = np.transpose(to_keep2)
        scat3 = bqp.Scatter(scales = scales, enable_move = False, update_on_move = False, stroke_width = 0)
        scat3.x , scat3.y = scat2.x, scat2.y
    
    marks.append(scat)
    if general:
        marks.append(scat2)
    
    #draw the chest
    chest = bqp.Lines(scales=scales)
    chest.x, chest.y = np.transpose([body.pts[1].to_array(), body.pts[2].to_array(), body.pts[9].to_array(), \
                                     body.pts[13].to_array(), body.pts[6].to_array(), body.pts[1].to_array()])
    marks.append(chest)
    
    
    #draw the head
    head = bqp.Lines(scales=scales)
    head.x = np.cos(np.linspace(0, 2*np.pi, 100))*60+body.pts[0].x
    head.y = np.sin(np.linspace(0, 2*np.pi, 100))*65+body.pts[0].y
    marks.append(head)
    
    i = 0
    #draw the left arm
    left_arm = bqp.Lines(scales=scales)
    if l_arm:
        left_arm.x, left_arm.y = [[body.pts[2].x, scat.x[i], scat.x[i+1]],[body.pts[2].y, scat.y[i], scat.y[i+1]]]
        i +=2
    else:
        left_arm.x, left_arm.y = [[body.pts[2].x, body.pts[3].x, body.pts[4].x],[body.pts[2].y, body.pts[3].y, body.pts[4].y]]
    marks.append(left_arm)
    
    
    #draw the right arm
    right_arm = bqp.Lines(scales=scales)
    if r_arm:
        right_arm.x, right_arm.y = [[body.pts[6].x, scat.x[i], scat.x[i+1]],[body.pts[6].y, scat.y[i], scat.y[i+1]]]
        i+=2
    else:
        right_arm.x, right_arm.y = [[body.pts[6].x, body.pts[7].x, body.pts[8].x],[body.pts[6].y, body.pts[7].y,body.pts[8].y]]
    
    marks.append(right_arm)
    
    
    #draw the left leg
    left_leg = bqp.Lines(scales=scales)
    if l_leg:
        left_leg.x, left_leg.y = [[body.pts[9].x, scat.x[i], scat.x[i+1]],[body.pts[9].y, scat.y[i], scat.y[i+1]]]
        i+=2
    else:
        left_leg.x, left_leg.y = [[body.pts[9].x, body.pts[10].x, body.pts[11].x],[body.pts[9].y, body.pts[10].y,body.pts[11].y]]
    marks.append(left_leg)
    
    
    #draw the right leg
    right_leg = bqp.Lines(scales=scales)
    if r_leg:
        right_leg.x, right_leg.y = [[body.pts[13].x, scat.x[i], scat.x[i+1]],[body.pts[13].y, scat.y[i], scat.y[i+1]]]
        i+=2
    else:
        right_leg.x, right_leg.y = [[body.pts[13].x, body.pts[14].x, body.pts[15].x],[body.pts[13].y, body.pts[14].y, body.pts[15].y]]
    marks.append(right_leg)
    
    
    #draw the neck
    neck =  bqp.Lines(scales=scales)
    if neck_p:
        neck.x, neck.y = [[body.pts[1].x, scat.x[i]],[body.pts[1].y, scat.y[i]]]
        i+=1
    else:
        neck.x, neck.y = [[body.pts[1].x, body.pts[0].x],[body.pts[1].y, body.pts[0].y]]
    marks.append(neck)
    
    scat.observe(refresh, names=['x', 'y'])
    if general:
        scat2.observe(refresh_rot, names=['x', 'y'])
    
    
    return bqp.Figure(marks=marks, padding_y = 0., min_height = 750, min_width = 750)