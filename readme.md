# Python implementation of Cheng-Church algorithm based on Numpy

Data Mining and Big Data project work about python implementation of Cheng-Church algorithm based on Numpy for fast perfomance. 

## Overview
Let us consider a table in which each row represent an individual and each column an aspect of the individual. 
Using a traditional clustering algorithm would allow us to obtain the natural partition in which we can consider the entire population, in few words each cluster would represent all those individuals who are similar under all of the available aspects; that's the point, if some individuals are pretty similar but only under some conditions, these would be never included in the same cluster and sometimes that's wrong.
While a biclustering algorithm it's capable not only of grouping individuals but also the conditions.
For a better comprehension, the result of a clustering algorithm, so a cluster, it's a subset of the rows of the matrix. On the other hand, the result of a biclustering algorithm it's a subset of both of rows and columns.

![A comparison of clustering and biclustering](https://ars.els-cdn.com/content/image/1-s2.0-S0169260713002605-gr1.jpg)

Cheng and Church were interested in finding biclusters with a small mean squared residue, which is a measure of bicluster homogeneity. 
In particular the aim was to knowledge discovery from expression gene data in order to find in a well optimized way co-regulation patterns in yeast and humans.

### (readme in progess) ###
