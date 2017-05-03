import Limb
import Point
import cv2 as cv
import numpy as np


nb_p = 18
nb_l = 17

class Body:
    
    
    
    def __init__(self, points, limbs, painting, ignored_points):
        
        if not len(points) == nb_p:
            raise ValueError('there must be exactly '+str(nb_p)+' points in a body')
        if not len(limbs) == nb_l:
            raise ValueError('there must be exactly '+str(nb_l)+' points in a body')
        
        self.pts = points
        self.limbs = limbs
        self.painting = painting
        self.ignored = ignored_points
        
        #invalidate all ignored points
        for i in self.ignored:
            self.pts[i].invalidate()
            
    def get_body_limbs(self):
        b_limbs = list()
        for i in [0,1,9,6]:
            b_limbs.append(self.limbs[i])
        return b_limbs
    
    def all_limbs_valid(limbs):
        for l in limbs:
            if not l.valid():
                return False
        return True
        
        
        
    def draw(self, color, image):
        for i in range(nb_l):
            if i not in [6,9]:
                l = self.limbs[i]
                if l.valid():
                    cv.line(image, tuple(l.p1.x, l.p1.y), tuple(l.p2.x, l.p2.y), color, 2)
                if i == 12:
                    radius = l.length() * 0.6
                    center = l.p1
                    cv.circle(image, center, int(radius), color, 2)
        b_limbs = self.get_body_limbs()
        if self.all_limbs_valid(b_limbs):
            b_points = list()
            for bl in b_limbs:
                b_points.append(bl.p2)
            cv.polylines(image, np.int32([b_points]), True, color, 2)
        return image
        
        
          
    