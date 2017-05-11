from Point import Point

class Limb:
    def __init__(self, p1, p2):
        
        if type(p1) == type(None):
            self.p1 = Point(0,0, False)
        else:
            self.p1 = p1
        if type(p2) == type(None):
            self.p2 = Point(0,0, False)
        else:
            self.p2 = p2
        
        self.val = True
        if self.p1.eq(self.p2):
            self.val = False
            
    
    def valid(self):
        return self.p1.valid and self.p2.valid and self.val
    
    def length(self):
        return self.p1.dist(self.p2)
    
    def angle(self):
        if self.valid():
            return self.p1.angle(self.p2)
        return False
    
    def get_points(self):
        return [self.p1, self.p2]
    
    def set_p1(self, p):
        self.p1 = p
        
    def set_p2(self, p):
        self.p2 = p
        
    def middle(self):
        if self.valid():
            return Point((self.p1.x +self.p2.x)/2, (self.p1.y +self.p2.y)/2)
        return Point(0,0).invalidate()
        
    
    def to_string(self):
        """creates a string representation of a Limb"""
        if not self.val:
            return 'ivalid'
        s = 'p1 :\n'
        s += self.p1.to_string()
        s += '\np2 :\n'
        s += self.p2.to_string()
        return s