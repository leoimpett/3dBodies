import Point

class Limb:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        
    def length(self):
        return self.p1.dist(self.p2)
    
    def angle(self):
        return self.p1.angle(self.p2)
    
    def valid(self):
        return self.p1.valid() and self.p2.valid()
    
    def get_points(self):
        return [self.p1, self.p2]