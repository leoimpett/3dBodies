import math

class Point:
    
    def __init__(self,x,y,valid):
        self.x = x
        self.y = y
        self.valid = valid
        
    def dist(self, p):
        """computes the distance between this point and another point p"""
        return math.sqrt((p.x - self.x)**2+(p.y-self.y)**2)

    def translate(self, dx, dy):
        """translates a point of a vector (dx,dy)"""
        self.x += dx
        self.y += dy
        
    def scale(self, s):
        """scale the point of s"""
        self.x *= s
        self.Y *= s
        
    def angle(self, p):
        """compute the absolute angle with the horizontal of the line passing by two points"""
        return math.degrees(math.atan2(p.x - self.x, p.y - self.y))
    
    def valid(self):
        """"return True if the point is valid"""
        return valid
    
    def invalidate(self):
        self.valid = False
    
    def validate(self):
        self.valid = True