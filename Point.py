import math

class Point:
    
    def __init__(self,x,y,valid=True):
        self.x = x
        self.y = y
        self.valid = valid
        if self.x == 0 and self.y == 0:
            self.valid = False
    

    def eq(self, p):
        return self.x == p.x and self.y == p.y
    
    
    def dist(self, p):
        """computes the distance between this point and another point p"""
        if not(self.valid and p.valid):
            return 0
        return math.sqrt((p.x - self.x)**2+(p.y-self.y)**2)

    def translate(self, dx, dy):
        """translates a point of a vector (dx,dy)"""
        if self.valid:
            return Point(self.x + dx, self.y + dy)
        return self
        
    def scale(self, s):
        """scale the point of s"""
        return Point(int(self.x * s),int(self.y * s) )
    
    def rotate(self, o, angle):
        if self.valid and o.valid:
            a = math.radians(angle)
            
            s = math.sin(a)
            c = math.cos(a)
            
            #translate to origin
            xo = self.x - o.x
            yo = self.y - o.y
            
            #compute rotation
            nx = xo * c - yo * s
            ny = xo * s + yo * c
            
            #translate point back
            self.x = int(nx + o.x)
            self.y = int(ny + o.y)
        
    def angle(self, p):
        """compute the absolute angle with the horizontal of the line passing by two points"""
        return math.degrees(math.atan2(p.y - self.y, p.x - self.x))
        
    
    
    def invalidate(self):
        """invalidare that point"""
        self.valid = False
    
    def validate(self):
        """validate that point"""
        self.valid = True
        
    def diff(self, point):
        if not (self.valid and point.valid):
            return [0,0]
        return [point.x - self.x, point.y - self.y]
    
    def to_string(self):
        """creates a string representation of a point"""
        if not self.valid:
            return 'invalid'
        s = ''
        s += 'x: '
        s += str(self.x)
        s += '\ty: '
        s += str(self.y)
        return s
    
    def to_array(self):
        if not self.valid:
            return None
        return [self.x, self.y]