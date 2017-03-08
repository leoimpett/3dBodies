# 3dBodies
BW's semester project 2017


read an image directly from a url:

http://stackoverflow.com/questions/7391945/how-do-i-read-image-data-from-a-url-in-python 

### What I've done so far

Week1: 23.02 - 02.03
- Find absolute angles of limbs of a body.
- Plot a skeleton given points and limbs.
- Plot a skeleton given absolute limbs absolute angles length.
- Calculate error between two bodies.

Week2: 02.03 - 09.03
- Skeletons with some limbs lacking can be plotted as well
- Make skeleton look more human: the chest is now plotted as a 4-polygon and the head is a circle.
- Two skeletons are shown in a same plot so that differences between them are more visible.
- Compute the list of all bodies of all the 7500 (first?) paintings. body = (points, limbs, ignored points ids). Points are the list of points and are for all bodies in the same order (i.e. the 8th point will be the same body-part in each body). Limbs are the bones of the skeleton, also always in the same order. Ignored points are the points which we do not take care of: points of the head and points that are "null" (some bodies are incomplete).
- Scale and translate all the bodies for them to all have the same order of size and have the neck placed at the same point (for error computation to be consistent).
- Compute the average size of each limb. Scaling of bodies is done in order to have all the bodies to have the same chest size
- Improve error computation to not taking ignored points into account anymore. They were leading to erroneous results.