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
            if (it != 0)
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
                if (it == 0)
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

/*void printEPS(observation pts[], size_t len, cluster cent[], int k)
{
    int W = 400, H = 400;
float min_x = DBL_MAX, max_x = DBL_MIN, min_y = DBL_MAX, max_y = DBL_MIN;
float scale = 0, cx = 0, cy = 0;
float* colors = float*)malloc(sizeoffloat) * (k * 3));
    int i;
    size_t j;
float kd = k * 1.0;
    for (i = 0; i < k; i++)
    {
        *(colors + 3 * i) = (3 * (i + 1) % k) / kd;
        *(colors + 3 * i + 1) = (7 * i % k) / kd;
        *(colors + 3 * i + 2) = (9 * i % k) / kd;
    }

    for (j = 0; j < len; j++)
    {
        if (max_x < pts[j].x)
        {
            max_x = pts[j].x;
        }
        if (min_x > pts[j].x)
        {
            min_x = pts[j].x;
        }
        if (max_y < pts[j].y)
        {
            max_y = pts[j].y;
        }
        if (min_y > pts[j].y)
        {
            min_y = pts[j].y;
        }
    }
    scale = W / (max_x - min_x);
    if (scale > (H / (max_y - min_y)))
    {
        scale = H / (max_y - min_y);
    };
    cx = (max_x + min_x) / 2;
    cy = (max_y + min_y) / 2;

    printf("%%!PS-Adobe-3.0 EPSF-3.0\n%%%%BoundingBox: -5 -5 %d %d\n", W + 10,
           H + 10);
    printf(
        "/l {rlineto} def /m {rmoveto} def\n"
        "/c { .25 sub exch .25 sub exch .5 0 360 arc fill } def\n"
        "/s { moveto -2 0 m 2 2 l 2 -2 l -2 -2 l closepath "
        "	gsave 1 setgray fill grestore gsave 3 setlinewidth"
        " 1 setgray stroke grestore 0 setgray stroke }def\n");
    for (i = 0; i < k; i++)
    {
        printf("%g %g %g setrgbcolor\n", *(colors + 3 * i),
               *(colors + 3 * i + 1), *(colors + 3 * i + 2));
        for (j = 0; j < len; j++)
        {
            if (pts[j].group != i)
            {
                continue;
            }
            printf("%.3f %.3f c\n", (pts[j].x - cx) * scale + W / 2,
                   (pts[j].y - cy) * scale + H / 2);
        }
        printf("\n0 setgray %g %g s\n", (cent[i].x - cx) * scale + W / 2,
               (cent[i].y - cy) * scale + H / 2);
    }
    printf("\n%%%%EOF");

    // free accquired memory
    free(colors);
}*/

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

/*!
 * A function to test the kMeans function
 * Generates 100000 points in a circle of
 * radius 20.0 with center at (0,0)
 * and cluster them into 5 clusters
 *
 * <img alt="Output for 100000 points divided in 5 clusters" src=
 * "https://raw.githubusercontent.com/TheAlgorithms/C/docs/images/machine_learning/k_means_clustering/kMeansTest1.png"
 * width="400px" heiggt="400px">
 * @returns None
 */
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
    // printEPS(observations, size, clusters, k);
    //  Free the accquired memory
    free(observations);
    free(clusters);
}

/*!
 * A function to test the kMeans function
 * Generates 1000000 points in a circle of
 * radius 20.0 with center at (0,0)
 * and cluster them into 11 clusters
 *
 * <img alt="Output for 1000000 points divided in 11 clusters" src=
 * "https://raw.githubusercontent.com/TheAlgorithms/C/docs/images/machine_learning/k_means_clustering/kMeansTest2.png"
 * width="400px" heiggt="400px">
 * @returns None
 */
/*void test2()
{
    size_t size = 10000000L;
    observation* observations =
        (observation*)malloc(sizeof(observation) * size);
float maxRadius = 20.00;
float radius = 0;
float ang = 0;
    size_t i = 0;
    for (; i < size; i++)
    {
        radius = maxRadius * (float)rand() / RAND_MAX);
        ang = 2 * M_PI * (float)rand() / RAND_MAX);
        observations[i].x = radius * cos(ang);
        observations[i].y = radius * sin(ang);
    }
    int k = 11;  // No of clusters
    cluster* clusters = kMeans(observations, size, k);
    printEPS(observations, size, clusters, k);
    // Free the accquired memory
    free(observations);
    free(clusters);
}*/

/*!
 * This function calls the test
 * function
 */
int main()
{
    // srand(time(NULL));
    // srand(10);
    test();
    /* test2(); */
    return 0;
}
