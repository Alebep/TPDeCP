//#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define OBSERVATIONS_COUNT 1000000
#define CLUSTERS_COUNT 32

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

void kernelMain(observation *observations, cluster *clusters, int *k, int *sizeO, int *changed)
{
    int i,pv = 0;
    for (i = 0; i < *sizeO; i++)
    {
        float minD = DBL_MAX;
        float dist = 0;
        int index = -1;
        int j = 0;
        for (j = 0; j < *k; j++)
        {
            /* Calculate Squared Distance*/
            dist = (clusters[j].x - observations[i].x) * (clusters[j].x - observations[i].x) 
            + (clusters[j].y - observations[i].y) * (clusters[j].y - observations[i].y);
            if (dist < minD)
            {
                minD = dist;
                index = j;
            }
        }
        pv = index;



        //pv = calculateNearst(observations + i, clusters, *k);
        if (pv != observations[i].group)
        {
            observations[i].group = pv;
            (*changed)++;
        }
    }
}

int main(int argc, char **argv)
{   
    observation *observations = malloc(OBSERVATIONS_COUNT * sizeof(observation));
    cluster *clusters = malloc(CLUSTERS_COUNT * sizeof(cluster));
    int i;

    srand(10);
    // Generate random observations
    for (i = 0; i < OBSERVATIONS_COUNT; i++)
    {
        observations[i].x = (float)rand() / RAND_MAX;
        observations[i].y = (float)rand() / RAND_MAX;
    }

    // Initialize clusters
    for (i = 0; i < CLUSTERS_COUNT; i++)
    {
        clusters[i].x = observations[i].x;
        clusters[i].y = observations[i].y;
        clusters[i].count = 0;
    }
    for (i = 0; i < OBSERVATIONS_COUNT; i++)
    {
        observations[i].group = calculateNearst(observations + i, clusters, CLUSTERS_COUNT);
    }
    

    int changed, t, it = 0;
    do
    {
        changed = 0;
        // Calculate new centroids for each cluster
        for (i = 0; i < CLUSTERS_COUNT; i++)
        {
            clusters[i].x = 0;
            clusters[i].y = 0;
            clusters[i].count = 0;
        }
        for (i = 0; i < OBSERVATIONS_COUNT; i++)
        {
            t = observations[i].group;
            clusters[t].x += observations[i].x;
            clusters[t].y += observations[i].y;
            clusters[t].count++;
        }
    
        for (i = 0; i < CLUSTERS_COUNT; i++)
        {
            clusters[i].x /= clusters[i].count;
            clusters[i].y /= clusters[i].count;
        }

        int gpu_countCL = CLUSTERS_COUNT;
        int gpu_countOB = OBSERVATIONS_COUNT;
        int gpu_changed = changed;



        
        // Calculate nearest cluster for each local observation
        //kernelMain(observation *observations, cluster *clusters, int *k, int *sizeO, int *changed) 
        kernelMain(observations, clusters, &gpu_countCL, &gpu_countOB, &gpu_changed);
        changed = gpu_changed;
        
        
        





       // numero de iteracoes da certa
       if (it == 20)
            break;
        it++;
        //printf(" %d\n",it);
    } while (changed > 0);

    
    printf("Final centroids:\n");
    for (i = 0; i < CLUSTERS_COUNT; i++)
    {
        printf("Cluster %d: (%.3f, %.3f) with %ld observations\n", i, clusters[i].x, clusters[i].y, clusters[i].count);
    }
    printf("%d",it);

    return 0;
}