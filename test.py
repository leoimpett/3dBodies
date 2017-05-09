# -*- coding: utf-8 -*-
"""
Created on Tue May 09 16:24:35 2017

@author: bruno
"""

from Point import Point
from Limb import Limb
from Body import Body
import bd

pts = list()
limbs = list()
for i in range(18):
    pts.append(Point(i,0))
for i in range(17):
    limbs.append(Limb(pts[i],pts[i+1]))


b1 = Body(pts, limbs, 0, list())

pts = list()
limbs = list()
for i in range(18):
    pts.append(Point(2*i,0))
for i in range(17):
    limbs.append(Limb(pts[i],pts[i+1]))


b2 = Body(pts, limbs, 0, list())

bodies = [b1, b2]
print bd.all_bodies_mean_limb_length(bodies)