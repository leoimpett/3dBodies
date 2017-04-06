# 3dBodies
BW's semester project 2017



### Meeting 6 April

- Previous work: PRINTART project:
http://printart.isr.ist.utl.pt/paper_eccv12_final.pdf (read if you like)
http://printart.isr.ist.utl.pt/my_article.pdf (most important)

- how we got the data: https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation 


### Meeting 28 March 
- This tool shows us that there's a big correlation between *some* poses and the respective metadata (title, year, place, ...)
- midway through the project, we can think of identifying patterns automatically
- now, we want to augment the interactive tool, to show us: 
- how rare is that pose? e.g. number of poses that are at fixed distance D from the skeleton
- turn on and off individual points and add the neck in the interaction (e.g. we may not be interested in legs)
- for the nearest N points (or within distance D?), we want to know the bias in the distributions of metadata (school, time period, type, etc etc). 
- dual histogram (normalised by %, not frequency) with one transparent, over the variables FORM, TYPE, SCHOOL, TIMELINE. Add a p-value for each one to check if they are significantly distinct (e.g. https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test  ). More meaningful for distance D. 
- What information can you get from the bodies themselves, other than the metadata? Heatmap of positions, average size, average number of people in the painting, etc etc. 



***
NN_test = sklearn.neighbors.NearestNeighbors(n_neighbors=5, radius=1.0, leaf_size=30,
                                             metric=our_distance_function, algorithm='auto')
                                             def our_distance_function(a,  b):
    return np.mean(a - b)

our_distance_function(all_relative_angles[30],all_relative_angles[34])

NN_test.fit(all_relative_angles)

NN_test.kneighbors(all_relative_angles[60])
***



### For next time (written 21st march 17) : 
- change lambda of the error - use the scikit.neighbors.learn: 
http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.kneighbors  
(by using your own distance function, using the flag metric=’precomputed’ )

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
