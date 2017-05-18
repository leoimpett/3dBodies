from Limb import Limb
from Point import Point
import cv2 as cv
import numpy as np


nb_p = 20
nb_l = 17

original_limbs = [[1,2],[1,6],[2,3],[3,4],[6,7],[7,8],[1,9],[9,10],[10,11],\
                   [1,13],[13,14],[14,15],[1,0],[0,16],[16,18],[0,17],[17,19]]

def bound(a):
    """bounds an angle between -180 and +180"""
    while a > 180:
        a -= 360
    while a <= -180:
        a += 360
    return a

class Body:
    
    
    
    def __init__(self, points, limbs, painting, ignored_points):
        
        if not len(points) == nb_p:
            raise ValueError('there must be exactly '+str(nb_p)+' points in a body, found: '+str(len(points)))
        if not len(limbs) == nb_l:
            raise ValueError('there must be exactly '+str(nb_l)+' limbs in a body, found: ' + str(len(limbs)))
        
        self.pts = points
        self.limbs = limbs
        self.painting = painting
        self.ignored = ignored_points
        
        #a point can be Null, if so we ignore it
        for i in range(nb_p):
            if self.pts[i] == None:
                self.pts[i] = Point(0,0)
                self.ignored.append(i)
        
        #invalidate all ignored points
        for i in self.ignored:
            self.pts[i].invalidate()
            
    
    def get_body_limbs(self):
        """return the breast limbs"""
        b_limbs = list()
        for i in [0,1,9,6]:
            b_limbs.append(self.limbs[i])
        return b_limbs
    
    def all_limbs_valid(self, limbs):
        """return whether all the limbs of this body are valid or not"""
        for l in limbs:
            if not l.valid():
                return False
        return True
    
    def has_right_arm(self):
        return self.all_limbs_valid([self.limbs[4], self.limbs[5]])
    
    def has_right_leg(self):
        return self.all_limbs_valid([self.limbs[10], self.limbs[11]])

    def has_left_arm(self):
        return self.all_limbs_valid([self.limbs[2], self.limbs[3]])
    
    def has_left_leg(self):
        return self.all_limbs_valid([self.limbs[7], self.limbs[8]])
    
    def has_neck(self):
        return self.all_limbs_valid([self.limbs[12]])
        
    def draw(self, image, color):
        """Draw a body skeleton on an image in a color"""
        for i in range(nb_l):
            if i not in [6,9]:
                l = self.limbs[i]
                if l.valid():
                    cv.line(image, tuple([l.p1.x, l.p1.y]), tuple([l.p2.x, l.p2.y]), color, 2)
                    if i == 12:
                        radius = l.length() * 0.6
                        center = tuple([self.pts[0].x, self.pts[0].y])
                        cv.circle(image, center, int(radius), color, 2)
        
        b_limbs = self.get_body_limbs()
        if self.all_limbs_valid(b_limbs):
            b_points = list()
            for bl in b_limbs:
                b_points.append([bl.p2.x, bl.p2.y])
            cv.polylines(image, np.int32([b_points]), True, color, 2)
        
    
    def link_limbs(self, pts):
        new_limbs = list()
        for l in original_limbs:
            p1 = pts[l[0]]
            p2 = pts[l[1]]
            new_limbs.append(Limb(p1, p2))
        return new_limbs
    
    
    def translate(self, destination):
        """translate the whole body to have the neck at the destination point"""
        #compute the difference vector between the actual neck and the point the neck will be
        vector = self.pts[1].diff(destination)
        dx = vector[0]
        dy = vector[1]
        new_pts = list()
        for p in self.pts:
            new_pts.append(p.translate(dx,dy))
        new_limbs = self.link_limbs(new_pts)
        return Body(new_pts, new_limbs, self.painting, self.ignored)
        
    def scale(self, scale):
        """scales a body by a scale with the neck staying at its position."""
        original_neck = Point(self.pts[1].x, self.pts[1].y)
        origin = self.translate(Point(0,0))
        new_pts = list()
        for p in origin.pts:
            new_pts.append(p.scale(scale))
        new_limbs = self.link_limbs(new_pts)
        return Body(new_pts, new_limbs, self.painting, self.ignored).translate(original_neck)
        
    
    def middle_length(self):
        """returns the length of the breast"""
        lol = self.limbs[6].length()
        lor = self.limbs[9].length()
        if lor == 0:
            return lol
        if lol == 0:
            return lor
        return (lor + lol)/2.0
    
    def find_absolute_angles(self):
        """find the absolute angle of each of all the limbs of a body compared 
        to horizontal axis and return a list containing all those angles"""
        angles = list();
        for l in self.limbs:
            angles.append(l.angle())
        return angles
    
    def compute_scale(self, mean):
        """computes the scale that we will apply to our body to make it have a mean size"""
        lom = self.middle_length()
        if lom == 0:
            return 1
        return mean / lom
    
    def relative_angles(self, dev):
        angles = self.find_absolute_angles()
        rel = np.zeros(10)+360
        
        al = self.limbs[6].angle()
        ar = self.limbs[9].angle()
        if (not al) and (not ar):
            gen_angle = 90.0
        elif not al:
            gen_angle = ar + dev
        elif not ar:
            gen_angle = al - dev
        else:
            gen_angle = Limb(self.pts[1], Limb(self.pts[9], self.pts[13]).middle()).angle()
                      
                      
        shoulder = Limb(self.pts[2], self.pts[6]).angle()
        if not shoulder:
            shoulder = 0.0
        
        
        if self.limbs[2].valid():
            rel[0] = bound(angles[2]-(shoulder + 180))
        if self.limbs[3].valid():
            rel[1] = bound(angles[3]-angles[2])
        if self.limbs[4].valid():
            rel[2] = bound(angles[4]-shoulder)
        if self.limbs[5].valid():
            rel[3] = bound(angles[5]-angles[4])
        if self.limbs[7].valid():
            rel[4] = bound(angles[7]-gen_angle)
        if self.limbs[8].valid():
            rel[5] = bound(angles[8]-angles[7])
        if self.limbs[10].valid():
            rel[6] = bound(angles[10]-gen_angle)
        if self.limbs[11].valid():
            rel[7] = bound(angles[11]-angles[10])
            
        if self.limbs[12].valid():
            rel[8] = bound(angles[12]-shoulder)
        rel[9] = gen_angle
                      
        return rel
    
    def rotate(self, angle):
        o = self.pts[1]
        for p in self.pts:
            p.rotate(o,angle)
        return self
            
    