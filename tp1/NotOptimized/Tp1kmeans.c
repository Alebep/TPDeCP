//#define _USE_MATH_DEFINES /* required for MS Visual C */
#include <float.h>  /* DBL_MAX, DBL_MIN */
#include <math.h>   /* PI, sin, cos */
#include <stdio.h>  /* printf */
#include <stdlib.h> /* rand */
#include <string.h> /* memset */
#include <time.h>   /* time */

typedef struct observation
{
    float x;   /**< abscissa of 2D data point */
    float y;   /**< ordinate of 2D data point */
    int group; /**< the group no in which this observation would go */
} observation;

typedef struct cluster
{
    float x;      /**< abscissa centroid of this cluster */
    float y;      /**< ordinate of centroid of this cluster */
    size_t count; /**< count of observations present in this cluster */
} cluster;

int it = 0;

int calculateNearst(observation *o, cluster clusters[], int k)
{
    float minD = DBL_MAX;
    float dist = 0;
    int index = -1;
    int i = 0;
    for (; i < k; i++)
    {
        /* Calculate Squared Distance*/
        dist = (clusters[i].x - o->x) * (clusters[i].x - o->x) + (clusters[i].y - o->y) * (clusters[i].y - o->y);
        if (dist < minD)
        {
            minD = dist;
            index = i;
        }
    }
    return index;
}

/*!
 * Calculate centoid and assign it to the cluster variable
 *
 * @param observations  an array of observations whose centroid is calculated
 * @param size  size of the observations array
 * @param centroid  a reference to cluster object to store information of
 * centroid
 */
void calculateCentroid(observation observations[], size_t size,
                       cluster *centroid)
{
    size_t i = 0;
    centroid->x = 0;
    centroid->y = 0;
    centroid->count = size;
    for (; i < size; i++)
    {
        centroid->x += observations[i].x;
        centroid->y += observations[i].y;
        observations[i].group = 0;
    }
    centroid->x /= centroid->count;
    centroid->y /= centroid->count;
}

/*!
 *    --K Means Algorithm--
 * 1. Assign each observation to one of k groups
 *    creating a random initial clustering
 * 2. Find the centroid of observations for each
 *    cluster to form new centroids
 * 3. Find the centroid which is nearest for each
 *    observation among the calculated centroids
 * 4. Assign the observation to its nearest centroid
 *    to create a new clustering.
 * 5. Repeat step 2,3,4 until there is no change
 *    the current clustering and is same as last
 *    clustering.
 *
 * @param observations  an array of observations to cluster
 * @param size  size of observations array
 * @param k  no of clusters to be made
 *
 * @returns pointer to cluster object
 */
cluster *kMeans(observation observations[], size_t size, int k)
{  
    int flag = 0;
    cluster *clusters = NULL;
    if (k <= 1)
    {
        /*
        If we have to cluster them only in one group
        then calculate centroid of observations and
        that will be a ingle cluster
        */
        clusters = (cluster *)malloc(sizeof(cluster));
        memset(clusters, 0, sizeof(cluster));
        calculateCentroid(observations, size, clusters);
    }
    else if (k < size)
    {
        clusters = malloc(sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));
        /* STEP 1 */
        size_t j = 0;
        for (j = 0; j < size; j++)
        {
            observations[j].group = rand() % k;
        }
        size_t changed = 0;
        size_t minAcceptedError =
            size /
            10000; // Do until 99.99 percent points are in correct cluster
        int t = 0;
        int i = 0;
        for (i = 0; i < k; i++)
        {
            clusters[i].x = observations[i].x;
            clusters[i].y = observations[i].y;
        }
        do
        {
            /* Initialize clusters */
            size_t j = 0;
            if (flag != 0)
            {
                for (i = 0; i < k; i++)
                {
                    clusters[i].x = 0;
                    clusters[i].y = 0;
                    clusters[i].count = 0;
                }
                /* STEP 2*/
                for (j = 0; j < size; j++)
                {
                    t = observations[j].group;
                    clusters[t].x += observations[j].x;
                    clusters[t].y += observations[j].y;
                    clusters[t].count++;
                }
                for (i = 0; i < k; i++)
                {
                    clusters[i].x /= clusters[i].count;
                    clusters[i].y /= clusters[i].count;
                }
            }
            /* STEP 3 and 4 */
            changed = 0; // this variable stores change in clustering
            for (j = 0; j < size; j++)
            {
                t = calculateNearst(observations + j, clusters, k);
                if (flag == 0)
                {
                    observations[j].group = t;
                    changed++;
                }
                else if (t != observations[j].group)
                {
                    changed++;
                    observations[j].group = t;
                }
            }
            if(flag != 0)
                it++;
        } while (changed != 0); // Keep on grouping until we have
                                // got almost best clustering
    }
    else
    {
        /* If no of clusters is more than observations
           each observation can be its own cluster
        */
        clusters = (cluster *)malloc(sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));
        int j = 0;
        for (j = 0; j < size; j++)
        {
            clusters[j].x = observations[j].x;
            clusters[j].y = observations[j].y;
            clusters[j].count = 1;
            observations[j].group = j;
        }
    }
    return clusters;
}

void impri(cluster *cl, size_t s, int k)
{
    printf("N = %d k = 4\n", s, k);
    int i;
    for (i = 0; i < k; i++)
    {
        printf("Center: (%.3f, %.3f) : Size: %d\n", cl[i].x, cl[i].y, cl[i].count);
    }
    printf("\nIterations: %d\n", it);
}

static void test()
{
    size_t size = 10000000L;
    observation *observations = (observation *)malloc(sizeof(observation) * size);
    size_t i = 0;
    srand(10);
    for (; i < size; i++)
    {
        observations[i].x = (float)rand() / RAND_MAX;
        observations[i].y = (float)rand() / RAND_MAX;
    }
    int k = 4; // No of clusters
    cluster *clusters = kMeans(observations, size, k);
    impri(clusters, size, k);
    free(observations);
    free(clusters);
}


/*!
 * This function calls the test
 * function
 */
int main()
{
    test();
    return 0;
}
