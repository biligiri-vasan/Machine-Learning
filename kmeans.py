import sys
import urllib2
import random
import math

coordinate_points = []

centroids = []


class CoordinatePoint:

    def __init__(self, label, x, y):

        self.label = label
        self.x = x
        self.y = y


class Cluster:


    def __init__(self, points):


        # The points that belong to this cluster
        self.points = points

        # Set up the initial centroid
        self.centroid = self.calculateCentroid()

    def __repr__(self):

        return str(self.points)

    def update(self, points):

        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        shift = getDistance(old_centroid, self.centroid)
        return shift

    def calculateCentroid(self):

        numPoints = len(self.points)
        # Get a list of all coordinates in this cluster
        label = [p.label for p in self.points]
        x = [p.x for p in self.points]
        y = [p.y for p in self.points]
        # Reformat that so all x's are together, all y'z etc.
        zipped = zip(x,y)
        unzipped = zip(*zipped)

        # Calculate the mean for each dimension
        centroid_coords = [math.fsum(dList)/numPoints for dList in unzipped]
        #return the mean centroid of all points in this cluster
        return CoordinatePoint(label, centroid_coords[0], centroid_coords[1])



def getDistance(a, b):

    ret = ((a.x-b.x)**2)+((a.y-b.y)**2)
    return math.sqrt(ret)


def kmeans_clustering(points, k, cutoff):
	iteration=1
    # choose k random points to use as initial centroids
	initial = random.sample(points, k)

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
		for p in points:
            # Get the distance between that point and the centroid of the first
            # cluster.
			smallest_distance = getDistance(p, clusters[0].centroid)

            # Set the cluster this point belongs to
			clusterIndex = 0

            # For the remainder of the clusters
			for i in range(clusterCount - 1):
                # calculate the distance of that point to each other cluster's
                # centroid.
				distance = getDistance(p, clusters[i+1].centroid)

                # if the distance of the point from the centroid is lesser than
                # the previous distance, set the data point as belonging to the current cluster
				if distance < smallest_distance:
					smallest_distance = distance
					clusterIndex = i+1

			lists[clusterIndex].append(p)


        # Set maxShift to zero for this iteration
		maxShift = 0.0

        # for every cluster
		for i in range(clusterCount):
            # Calculate how far the centroid moved in this iteration
			shift = clusters[i].update(lists[i])

            # Keep track of the largest move from all cluster centroid updates
			maxShift = max(maxShift, shift)


        # If the centroids have stopped moving much, break from the loop
		if maxShift < cutoff:
			print "Converged after %s iterations" % loopCounter
			break

		iteration+=1

	global centroids
	centroids=clusters
	return lists


def validation_sse(list_of_clusters):
	#stores dist squared values of every point in a cluster to its centroid
	list_sse = []

	itr = 0
	total_sse = 0.0
	#for each cluster list(cluster list has a list of points belonging to individual cluster
	for i in list_of_clusters:
		sum1 = 0.0
		for j in i:#j refers to individual point in i

			term1 = (getDistance(centroids[itr].centroid,j)**2)

			sum1+=term1

		itr+=1
		list_sse.append(sum1)
	total_sse = sum(list_sse)
	print "Number of clusters - ",int(sys.argv[1])
	print "Validation SSE - ",total_sse

#EXECUTION BEGINS HERE
def main():
	if(len(sys.argv)!=4):

		print("Invalid number of arguments: ")
		print("Correct order:<numberOfClusters> <input-file-name> <output-file-name>")


	url_data = urllib2.urlopen(sys.argv[2])
	url_data = url_data.readline()
	split_data = url_data.split("\r")
	all_points = []
	i = 0

#storing the read coordinate points in a list 'a' and then every element in that list into an object CoordinatePoint
	for j in split_data:

		all_points.append(j.split("\t")) #a contains list of lists; individual list contains id,x,y values

		if(i!=0):


			point1 = CoordinatePoint(int(all_points[i][0]), float(all_points[i][1]),float(all_points[i][2]))

			coordinate_points.append(point1) #label,x,y,cluster

		i+=1


	num_clusters = int(sys.argv[1])
	opt_cutoff = 0.0
	#exit if given k value is greater than number of coordinate points
	if(num_clusters>len(coordinate_points)):
		print "The number of coordinate points is ",len(coordinate_points)
		print "Please input k value lesser than number of coordinate points"
		exit(0)
	list_of_clusters = kmeans_clustering(coordinate_points, num_clusters, opt_cutoff)

	f = open(sys.argv[3], 'w+')
	f.write("Cluster-ID    List of points\n")
	for i in range(num_clusters):
		f.seek(0,2)
		str0=str(i+1)
		str1=str([c.label for c in list_of_clusters[i]])
		str2=str0+"	     "+str1
		f.write(str2)
		f.write("\n")
	f.close()

	validation_sse(list_of_clusters)

if __name__== "__main__":
	main()
