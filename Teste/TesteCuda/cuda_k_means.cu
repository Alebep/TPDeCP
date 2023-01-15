//#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
/*
#define OBSERVATIONS_COUNT 1000000
#define CLUSTERS_COUNT 32
*/

//#define NUMThread 1000

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

/*__global__ Mostrar()
{
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    printf("%d",t);
}*/

/*
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
            //Calculate Squared Distance
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
//*/

__global__ void kernelMain(observation *observations, cluster *clusters, int *k, int *sizeO, int *changed)
{
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    int pv = 0;
    if(threadID < *sizeO)
    {
        float minD = DBL_MAX;
        float dist = 0;
        int index = -1;
        int j = 0;
        for (j = 0; j < *k; j++)
        {
            //Calculate Squared Distance
            dist = (clusters[j].x - observations[threadID].x) * (clusters[j].x - observations[threadID].x) 
            + (clusters[j].y - observations[threadID].y) * (clusters[j].y - observations[threadID].y);
            if (dist < minD)
            {
                minD = dist;
                index = j;
            }
        }
        pv = index;



        //pv = calculateNearst(observations + i, clusters, *k);
        if (pv != observations[threadID].group)
        {
            observations[threadID].group = pv;
            //(*changed)++;
            atomicAdd(changed, 1);
        }
    }
}



int main(int argc, char *argv[])
{   

    int OBSERVATIONS_COUNT = atoi(argv[1]);
    int CLUSTERS_COUNT = atoi(argv[2]);
    int numThreads = atoi(argv[3]);

    int sizeCl = CLUSTERS_COUNT * sizeof(cluster);
    int sizeO = OBSERVATIONS_COUNT * sizeof(observation);
    int sizeI = 1*sizeof(int);

    observation *observations = (observation*) malloc(sizeO);
    cluster *clusters = (cluster*) malloc(sizeCl);
    int i;

    //variaveis do device
    ///*
    observation *gpu_observations;
    cluster *gpu_clusters;
    int *gpu_countCL; //= CLUSTERS_COUNT;
    int *gpu_countOB; //= OBSERVATIONS_COUNT;
    int *gpu_changed; //= changed;
    cudaMalloc( &gpu_clusters, sizeCl);
    cudaMalloc( &gpu_observations, sizeO);
    cudaMalloc( &gpu_countCL, sizeI);
    cudaMalloc( &gpu_countOB, sizeI);
    cudaMalloc( &gpu_changed, sizeI);

    // copia dos parametros de inicializacao
    cudaMemcpy( gpu_countOB, &OBSERVATIONS_COUNT, sizeI, cudaMemcpyHostToDevice);
    cudaMemcpy( gpu_countCL, &CLUSTERS_COUNT, sizeI, cudaMemcpyHostToDevice);
    //*/

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

        /*
        int gpu_countCL= CLUSTERS_COUNT;
        int gpu_countOB= OBSERVATIONS_COUNT;
        int gpu_changed = changed;
        //sequencial
        kernelMain(observations, clusters, &gpu_countCL, &gpu_countOB, &gpu_changed);
        changed = gpu_changed;
        //*/

        ///*
        //copiando as observacoes, cluster e a variavel que controla as mudanca pro device
        cudaMemcpy( gpu_changed, &changed, sizeI, cudaMemcpyHostToDevice);
        cudaMemcpy( gpu_observations, observations, sizeO, cudaMemcpyHostToDevice);
        cudaMemcpy( gpu_clusters, clusters, sizeCl, cudaMemcpyHostToDevice);

        
        // Calculate nearest cluster for each local observation
        //kernelMain(observation *observations, cluster *clusters, int *k, int *sizeO, int *changed) 
        kernelMain<<<ceil(OBSERVATIONS_COUNT/numThreads/*NUMThread*/),numThreads/*NUMThread*/>>>(gpu_observations, gpu_clusters, gpu_countCL, gpu_countOB, gpu_changed);
        //changed = gpu_changed;
        
        
        cudaMemcpy( &changed, gpu_changed, sizeI, cudaMemcpyDeviceToHost);
        if(changed != 0)
        {
            cudaMemcpy( clusters, gpu_clusters, sizeCl, cudaMemcpyDeviceToHost);
            cudaMemcpy( observations, gpu_observations, sizeO, cudaMemcpyDeviceToHost);
        }
        //*/
        




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