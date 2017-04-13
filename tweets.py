import sys
import urllib2
import random
import math
import json


list_of_tweets = []
list_of_initial_tweets = []


class Tweet:

	text = ""
	id_str = ""


def tweet_define(ind_line):
	obj = Tweet()
	obj.__dict__.update(ind_line)
	return obj


class Cluster:


	def __init__(self, points):

		# The points that belong to this cluster
		self.points = points

		# Set up the initial centroid (this is usually based off one point)
		self.centroid = self.calculateCentroid()

	def __repr__(self):

		return str(self.points)

	def update(self, points):

		old_centroid = self.centroid
		self.points = points
		self.centroid = self.calculateCentroid()
		shift = getJaccardDistance(old_centroid, self.centroid)
		return shift

	def calculateCentroid(self):

		numPoints = len(self.points)
		least_value = 0.0
        # Get a list of all coordinates in this cluster
		for p in self.points:
			new_centroid=p

			sum_values=0.0

			for tp in self.points:
				sum_values+=getJaccardDistance(p,tp)#summation

			if(sum_values<=least_value):
				least_value=sum_values
				new_centroid=p

		return new_centroid



def validation_sse(list_of_clusters):
	list_sse = []
	itr = 0
	total_sse = 0.0
	for i in list_of_clusters:#list_of_clusters - has list of points belonging to each individual cluster;its a list of lists
		sum1 = 0.0
		for j in i:#j refers to individual point in i
			term1 = (getJaccardDistance(list_of_centroids[itr].centroid,j)**2)
			sum1+=term1
		itr+=1
		list_sse.append(sum1)
	total_sse = sum(list_sse)
	print "Number of clusters - ",int(sys.argv[1])
	print "Validation SSE - ",total_sse

def kmeans_clustering(tweets, default_centroids,cutoff):
	iteration=1
	initial = []
    # Pick out k random points to use as initial centroids

	for id in default_centroids:

		for t in tweets:
			if(t.id_str==id):
				initial.append(t)

    # Create k clusters using those centroids
	clusters = [Cluster([p]) for p in initial]

    # Loop through the dataset until the clusters stabilize
	loopCounter = 0

	while (iteration<=25):
        # Create a list of lists to hold the points in each cluster
		lists = [ [] for c in clusters]
		clusterCount = len(clusters)

        # Start counting loops
		loopCounter += 1
        # For every point in the dataset
		for p in tweets:
            # Get the distance between that point and the centroid of the first
            # cluster.
			smallest_distance = getJaccardDistance(p, clusters[0].centroid)

            # Set the cluster this point belongs to
			clusterIndex = 0

            # For the remainder of the clusters
			for i in range(clusterCount - 1):
                # calculate the distance of that point to each other cluster's
                # centroid.
				distance = getJaccardDistance(p, clusters[i+1].centroid)


                # if the distance of the point from the centroid is lesser than
                # the previous distance, set the data point as belonging to the current cluster
				if distance < smallest_distance:
					smallest_distance = distance
					clusterIndex = i+1
			lists[clusterIndex].append(p)

        # Set maxShift to zero for this iteration
		maxShift = 0.0

        # As many times as there are clusters
		for i in range(clusterCount):
            # Calculate how far the centroid moved in this iteration
			shift = clusters[i].update(lists[i])
            # Keep track of the largest move from all cluster centroid updates
			maxShift = max(maxShift, shift)

        # centroids have not moved much
		if maxShift < cutoff:
			print "Converged after %s iterations" % loopCounter
			break
		iteration+=1

	global list_of_centroids
	list_of_centroids=clusters
	return lists

def getJaccardDistance(str1, str2):

	str1=str1.text
	str2=str2.text
	str1 = str1.split()
	str2 = str2.split()
	union = list(set(str1+str2))
	intersection = list(set(str1) - (set(str1)-set(str2)))

	jaccard_dist = float(len(intersection))/len(union)
	return 1-jaccard_dist

#EXECUTION BEGINS HERE
if(len(sys.argv)!=5):
    print("Invalid number of arguments: ")
    print("Correct order:tweets-k-means.py <numberOfClusters> <initialSeedsFile> <TweetsDataFile> <outputFile>")
    exit(0)

#reading list of all tweets and mapping them into Tweet objects
url_data = urllib2.urlopen(sys.argv[3])
url_data = url_data.readlines()

for d in url_data:
	obj=json.loads(d,object_hook=tweet_define)

	list_of_tweets.append(obj)#list_of_tweets is the list of json objects

#reading list of initial tweets
file= urllib2.urlopen(sys.argv[2])

for line in file:

	line= line.replace(",\n","")
	list_of_initial_tweets.append(line)#list_of_initial_tweets are the initial centroids; it has list of IDs of the 25 initial centroids


num_clusters = int(sys.argv[1])
opt_cutoff = 0.01
if(num_clusters>len(list_of_tweets)):
	print "The total number of tweets is ",len(list_of_tweets)
	print "Please input k value lesser than number of tweets"
	exit(0)

if(num_clusters>25):
	print "Please input k value <= 25"
	exit(0)

list_of_clusters = kmeans_clustering(list_of_tweets, list_of_initial_tweets,opt_cutoff)


f = open(sys.argv[4], 'w+')
f.write("Cluster-ID    List of points\n")
for i in range(num_clusters):
	str1=[]
	f.seek(0,2)
	str0=str(i+1)

	str1=str([str(c.id_str) for c in list_of_clusters[i]])

	str2=str0+"	     "+str1

	f.write(str2)
	f.write("\n")
f.close()

validation_sse(list_of_clusters)
