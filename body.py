from Limb import Limb
from Point import Point
import cv2 as cv
import numpy as np


nb_p = 20
nb_l = 17

original_limbs = [[1,2],[1,6],[2,3],[3,4],[6,7],[7,8],[1,9],[9,10],[10,11],\
                   [1,13],[13,14],[14,15],[1,0],[0,16],[16,18],[0,17],[17,19]]


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
        
        
        
          
    