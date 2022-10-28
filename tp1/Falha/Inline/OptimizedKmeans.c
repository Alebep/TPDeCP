//#define _USE_MATH_DEFINES /* required for MS Visual C */
#include <float.h>  /* DBL_MAX, DBL_MIN */
#include <math.h>   /* PI, sin, cos */
#include <stdio.h>  /* printf */
#include <stdlib.h> /* rand */
#include <string.h> /* memset */
#include <time.h>   /* time */
#include "../include/utils.h"

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

/*!
 * This function calls the test
 * function
 */
int main()
{
    size_t changed = 0;
    size_t j = 0;
    size_t i = 0;
    size_t size = 10000000L;
    int k = 4;
    int t = 0;
    int index = -1; // candidato a ser apagado
    float minD = DBL_MAX;
    float dist = 0;

    observation *observations = (observation *)malloc(sizeof(observation) * size);
    cluster *clusters = malloc(sizeof(cluster) * k);
    memset(clusters, 0, k * sizeof(cluster));
    srand(10);
    for (; i < size; i++)
    {
        observations[i].x = (float)rand() / RAND_MAX;
        observations[i].y = (float)rand() / RAND_MAX;
    }

    /* STEP 1 */
    for (i = 0; i < k; i++)
    {
        clusters[i].x = observations[i].x;
        clusters[i].y = observations[i].y;
    }

    // com os centroids sendo as 4 primeiras amostras
    do
    {
        // STEP 3 and 4
        changed = 0;
        for (j = 0; j < size; j++)
        {
            minD = DBL_MAX;
            for (i = 0; i < k; i++)
            {
                // Calculate Squared Distance
                dist = (clusters[i].x - observations[j].x) * (clusters[i].x - observations[j].x) + (clusters[i].y - observations[j].y) * (clusters[i].y - observations[j].y);
                if (dist < minD)
                {
                    minD = dist;
                    observations[j].group = i;
               }
            }
        }
    } while (changed != 0); // Keep on grouping until we have*/
    // ate qui ok

    // kMeans segunda vez, zerando os centroids, para verificar se obtemos novos.
    do
    {
        // Initialize clusters
        size_t j = 0;

        for (i = 0; i < k; i++)
        {
            clusters[i].x = 0;
            clusters[i].y = 0;
            clusters[i].count = 0;
        }
        // STEP 2
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

        // STEP 3 and 4
        changed = 0; // this variable stores change in clustering
        for (j = 0; j < size; j++)
        {
            minD = DBL_MAX;
            for (i = 0; i < k; i++)
            {
                // Calculate Squared Distance
                dist = (clusters[i].x - observations[j].x) * (clusters[i].x - observations[j].x) + (clusters[i].y - observations[j].y) * (clusters[i].y - observations[j].y);
                if (dist < minD)
                {
                    minD = dist;
                    t = i;
                }
            }
            if (t != observations[j].group)
            {
                changed++;
                observations[j].group = t;
            }
        }
        it++;
    } while (changed != 0); // Keep on grouping until we have*/

    printf("N = %ld k = %d\n", size, k);
    for (i = 0; i < k; i++)
    {
        printf("Center: (%.3f, %.3f) : Size: %ld\n", clusters[i].x, clusters[i].y, clusters[i].count);
    }
    
    printf("\nIterations: %d\n", it);

    return 0;
}
