# 3dBodies
BW's semester project 2017

### Meeting 18 May

General issues
- sort out global rotation 

For p-value computation. 
- precompute NN KD-tree
- try approximate LSH forest: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LSHForest.html#sklearn.neighbors.LSHForest 
- hybrid approach: LSH approx. generate list of p-values, then check the hypotheses again with precise KD-tree. 
- look at getting skeleton much more real-time; if displaying the URLs takes too long, display just skeleton and metadata. 
- in interactive script: instead of searching every 'refresh', every time a point is changed by e.g. more than 5PX
- for hypotheses: remove artworks that don't have *any* bodies in them. Maybe you want also to remove all the bodies that don't have enough limbs (eg. remove all that don't have at least X=6 points)
- Implement both 'metadata WRT bodies' and 'metadata WRT paintings'. 
- way to display all the skeletons of the hypothesis, and the metadata. Also original images for the first 10. 

LATER; - include timing graphs for varying n nearest neighbours and N dataset size. 


SVM : query by metadata!
- details on SVM are here: http://scikit-learn.org/stable/modules/svm.html 
- query by metadata (time, place, maybe 'i dont care' for some)
- classify dataset y=1 if meets metadata-conditions, y=0 otherwise
- X = [sin(bodies-angles),cosine(bodies-angles)]
- train SVM(X,y)
- calculate *PROBABILITIES* for X (using the SVM)
- which are the most probable in query? 
- Display results as if you had some automatic hypothesis. 
- do this inside a 4-fold cross validation, i.e. separate into 4 groups and then Train or Calculate-probailities:
TTTC
TTCT
TCTT
CTTT... 







### Meeting 4 May

Updates 
- Possibility to select different limbs
- Search by neck


To do (this week)
- Implement deselection in angles
- Search by global rotation
- check p-values - check they're appropriate (right hypothesis - check right test for 'A subset of B')
- check for random datapoints, with N (>20) nearest neighbors, see what the p-values look like. 
- multiple comparison problem (multiple hypothesis testing) 
- make a list of random hypotheses: skeleton, covariate (i.e. which dimension of metadata), p-value: then sort by p-value. 


For the rest of the semester
- Move from supervised (search) to unsupervised (clustering). 
- generate automatic hypotheses
- evaluate search system with art-historian



### Meeting 6 April

- Previous work: 

1. PRINTART project:
http://printart.isr.ist.utl.pt/paper_eccv12_final.pdf (read if you like)
http://printart.isr.ist.utl.pt/my_article.pdf (most important)
(doing something slightly different, and not very well)

2. Activity Recognition (https://www.cs.utexas.edu/~chaoyeh/web_action_data/dataset_list.html) in 'still' images!! And motivator for pose estimation. 

3. All backed up by Johansson's 'Light Spots Model' , which is also important in psychology, e.g. : http://journals.sagepub.com/doi/pdf/10.1068/p5096


- evaluation (precision @ N, user studies)



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
