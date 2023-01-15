#include <mpi.h>
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


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sizeCl = CLUSTERS_COUNT * sizeof(cluster);
    int sizeO = OBSERVATIONS_COUNT * sizeof(observation);

    observation *observations = (observation*) malloc(sizeO);
    cluster *clusters = (cluster*) malloc(sizeCl);
    observation *localObservations = (observation*) malloc((OBSERVATIONS_COUNT / size)* sizeof(observation));
    cluster *localClusters = (cluster*) malloc(CLUSTERS_COUNT * sizeof(cluster));
    int localObservationsCount = OBSERVATIONS_COUNT / size;
    int i;
    int changed, t, it = 0;

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

    int fchanged = 0;
    MPI_Scatter(observations, localObservationsCount, MPI_FLOAT, localObservations, localObservationsCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
    do
    {
        changed = 0;
        for (i = 0; i < localObservationsCount; i++)
        {
            //printf("%d\n \n",rank);
            //printf("%d %d",rank,i);
            t = calculateNearst(localObservations + i, clusters, CLUSTERS_COUNT);
            if (localObservations[i].group != t)
            {
                localObservations[i].group = t;
                changed++;
                //MPI_Barrier(MPI_COMM_WORLD);
            }
        }
        //MPI_Gather(localObservations, localObservationsCount, MPI_FLOAT, observations, localObservationsCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
        //MPI_Allgather(localObservations, localObservationsCount, MPI_FLOAT, observations, localObservationsCount, MPI_FLOAT, MPI_COMM_WORLD);
        //if(rank == 0){
            // Calculate new centroids for each cluster
            for (i = 0; i < CLUSTERS_COUNT; i++)
            {
                clusters[i].x = 0;
                clusters[i].y = 0;
                clusters[i].count = 0;
            }
            for (i = 0; i < localObservationsCount; i++)
            {
                t = observations[i].group;
                clusters[t].x += localObservationsCount[i].x;
                clusters[t].y += localObservationsCount[i].y;
                clusters[t].count++;
            }
            
            for (i = 0; i < CLUSTERS_COUNT; i++)
            {
                clusters[i].x /= clusters[i].count;
                clusters[i].y /= clusters[i].count;
            }
       // }
        /*else
        {
            ;
           // break;
        }*/

        /*for (i = 0; i < CLUSTERS_COUNT; i++)
        {
            
            printf("Cluster %d: (%.3f, %.3f) with %zu observations\n", i, clusters[i].x, clusters[i].y, clusters[i].count);
        }*/
       if (it == 19)
            break;
        it++;
        printf(" %d changed: %d fchanged: %d\n",it,changed,fchanged);
    } while (changed > -1);

    
    if (rank == 0)
    {
        int x = 0;
        printf("Final centroids:\n");
        for (i = 0; i < CLUSTERS_COUNT; i++)
        {
            x+=clusters[i].count;
            printf("Cluster %d: (%.3f, %.3f) with %zu observations\n", i, clusters[i].x, clusters[i].y, clusters[i].count);
            printf("somatorio: %d\n",x);
        }
        printf("%d",it);
    }

    MPI_Finalize();

    return 0;
}