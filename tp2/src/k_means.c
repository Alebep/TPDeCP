//#define _USE_MATH_DEFINES /* required for MS Visual C */
#include <float.h>  /* DBL_MAX, DBL_MIN */
#include <math.h>   /* PI, sin, cos */
#include <stdio.h>  /* printf */
#include <stdlib.h> /* rand */
#include <string.h> /* memset */
#include <time.h>   /* time */
#include "../include/utils.h"

//int threads = 1;

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
        /*calculateCentroid(observations, size, clusters);*/
    }
    else if (k < size)
    {
        clusters = malloc(sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));
        /* STEP 1 */
        size_t j = 0;
        size_t changed = 0;
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
                for (i = 0; i < k; i += 2)
                {
                    clusters[i].x /= clusters[i].count;
                    clusters[i + 1].x /= clusters[i + 1].count;
                    clusters[i].y /= clusters[i].count;
                    clusters[i + 1].y /= clusters[i + 1].count;
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
            if(it == 21)
                break;
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
    printf("N = %ld k = %d\n", s, k);
    int i;
    for (i = 0; i < k; i++)
    {
        printf("Center: (%.3f, %.3f) : Size: %ld\n", cl[i].x, cl[i].y, cl[i].count);
    }
    printf("\nIterations: %d\n", it - 1);
}

static void test(size_t size, int k)
{
    // size_t size = 10000000L;
    observation *observations = (observation *)malloc(sizeof(observation) * size);
    size_t i = 0;
    srand(10);
    for (; i < size; i++)
    {
        observations[i].x = (float)rand() / RAND_MAX;
        observations[i].y = (float)rand() / RAND_MAX;
    }
    // int k = 4; // No of clusters
    cluster *clusters = kMeans(observations, size, k);
    // printf("%d\n",threads);
    impri(clusters, size, k);
    free(observations);
    free(clusters);
}

/*!
 * This function calls the test
 * function
 */
int main(int argc, char *argv[])
{
    // argv[1] -> corresponde ao numero de amostras
    // argv[2] -> corresponde ao numero de clusters
    // argv[3] -> corresponde ao numero de threads
    // test();
    /*printf("%d\n", argc);
    size_t l[3];
    l[0] = atoi(argv[1]);
    l[1] = atoi(argv[2]);
    threads = atoi(argv[3]);
    l[2] = threads;
    for (size_t i = 1; i < 4; i++)
    {
        printf("%ld\n", l[i-1]);
    }
    printf("classificar amostras\n");//*/
    //threads = atoi(argv[3]);
    test(atoi(argv[1]), atoi(argv[2]));

    return 0;
}
