# 3dBodies
BW's semester project 2017


read an image directly from a url:

http://stackoverflow.com/questions/7391945/how-do-i-read-image-data-from-a-url-in-python 




### For next time (written 21st march 17) : 
- interactive searching produces a grid with original image, and skeleton, and some metadata (print the time-period). 
- start to read a little about CLustering, especially: the K-means algorithm, and try (but don't implement yourself - use the library) T-SNE on the angles 
http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html 

NOT FOR NEXT TIME - but to keep in mind:
- be able to search with positive and negative examples (online training an SVM). 
- once a cluster is identified manually, to be able to visualise metadata (e.g. relationship through time, places, and keywords in the title). 


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
