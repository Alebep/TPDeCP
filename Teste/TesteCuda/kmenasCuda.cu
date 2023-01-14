//#define _USE_MATH_DEFINES /* required for MS Visual C */
#include <float.h>  /* DBL_MAX, DBL_MIN */
#include <math.h>   /* PI, sin, cos */
#include <stdio.h>  /* printf */
#include <stdlib.h> /* rand */
#include <string.h> /* memset */
#include <time.h>   /* time */
//#include "../include/utils.h"

#define numThreads 1000

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

__global__ void mainKernel(observation *observations, cluster *clusters, int *changed,int k,int N)
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    if(threadId < N)
    {
        int pv = 0;
        pv = (int) calculateNearst(observations + threadId, clusters, k);
        // SE HOUVE MUDANÇA DA POSICAO DAS AMOSTRAS
        if (pv != observations[threadId].group)
        {
            //aqui ha concorrencia mais e controlada, nao tm potencial para alterar nada
            (*changed)++;
            observations[threadId].group = pv;
        }
    }
}

__global__ void InitializeKernel(observation *observations, cluster *clusters,int k,int N)
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if(threadId < N)
    {
        observations[threadId].group =  (int) calculateNearst(observations + threadId, clusters, k);
    }
}

__device__ int calculateNearst(observation *o, cluster clusters[], int k)
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
    return (int) index;
}

void Initialize(cluster *clusters,observation *observations,size_t size, int k)
{
    observation *gpu_observations;
    cluster *gpu_clusters;
    int sizeClusters = sizeof(cluster) * k, sizeObservations = sizeof(observation) * size;
    cudaMalloc( &gpu_clusters, sizeClusters);
    cudaMalloc( &gpu_observations, sizeObservations);
    int i = 0;// j = 0;
    for (i = 0; i < k; i++)
    {
        clusters[i].x = observations[i].x;
        clusters[i].y = observations[i].y;
    }
    cudaMemcpy( gpu_observations, observations, sizeObservations, cudaMemcpyHostToDevice);
    cudaMemcpy( gpu_clusters, clusters, sizeClusters, cudaMemcpyHostToDevice);
    //InitializeKernel(observation *observations, cluster *clusters,int k,int N)
    InitializeKernel(observations, clusters,k,size);
    cudaMemcpy( observations, gpu_observations, sizeObservations, cudaMemcpyDeviceToHost);
    cudaMemcpy( clusters, gpu_clusters, sizeClusters, cudaMemcpyDeviceToHost);
    /*for (j = 0; j < size; j++)
    {
        observations[j].group =  calculateNearst(observations + j, clusters, k);
    }
    //printf("%f\n", ceil(size/1000));*/
}

cluster *kMeans(observation observations[], size_t size, int k)
{
    cudaDeviceReset();
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
        observation *gpu_observations;
        cluster *gpu_clusters;
        int *gpu_changed; /*gpu_size, *gpu_k;*/
        int sizeClusters = sizeof(cluster) * k, sizeObservations = sizeof(observation) * size;
        clusters = (cluster *) malloc(sizeof(cluster) * k);
        memset(clusters, 0, k * sizeof(cluster));
        cudaMalloc( &gpu_clusters, sizeClusters);
        cudaMalloc( &gpu_observations, sizeObservations);
        cudaMalloc( &gpu_changed, sizeof(size_t));
        /*cudaMalloc( &gpu_size, sizeof(int));
        cudaMalloc( &gpu_k, sizeof(int));
        /* STEP 1 */
        //size_t j = 0;
        size_t changed = 0;
        //int t = 0;
        int index = 0;
        int i = 0;
        Initialize(clusters,observations,size,k);
        it++;// a inicializacao conta como um iteracao
        do
        {
            /* Initialize clusters */
            size_t j = 0;
            for (i = 0; i < k; i++)
            {
                clusters[i].x = 0;
                clusters[i].y = 0;
                clusters[i].count = 0;
            }
            /* STEP 2*/
            for (j = 0; j < size; j++)
            {
                index = observations[j].group;
                clusters[index].x += observations[j].x;
                clusters[index].y += observations[j].y;
                clusters[index].count++;
            }
            for (i = 0; i < k; i += 2)
            {
                clusters[i].x /= clusters[i].count;
                clusters[i + 1].x /= clusters[i + 1].count;
                clusters[i].y /= clusters[i].count;
                clusters[i + 1].y /= clusters[i + 1].count;
            }
            //ISSO SERA PARALELO
            /* STEP 3 and 4 */ // int sizeClusters = sizeof(cluster) * k, sizeObservations = sizeof(observation) * size;
            changed = 0; // this variable stores change in clustering
            cudaMemcpy( gpu_observations, observations, sizeObservations, cudaMemcpyHostToDevice);
            cudaMemcpy( gpu_clusters, clusters, sizeClusters, cudaMemcpyHostToDevice);
            cudaMemcpy( gpu_changed, changed, sizeof(size_t), cudaMemcpyHostToDevice);
            /*cudaMemcpy( gpu_size, size, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy( gpu_k, k, sizeof(int), cudaMemcpyHostToDevice);*/
            //mainKernel(observation *observations, cluster *clusters, int changed,int k,int N)
            mainKernel<<<ceil(size/numThreads),numThreads>>>(gpu_observations,gpu_clusters,gpu_changed,k,size);

            cudaMemcpy( changed, gpu_changed, sizeof(int), cudaMemcpyDeviceToHost);
            if(changed != 0 )
            {
                cudaMemcpy( observations, gpu_observations, sizeObservations, cudaMemcpyDeviceToHost);
                cudaMemcpy( clusters, gpu_clusters, sizeClusters, cudaMemcpyDeviceToHost);
            }
            if (it == 21)
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

static void test(int size, int k)
{
    observation *observations = (observation *)malloc(sizeof(observation) * size);
    size_t i = 0;
    srand(10);
    for (; i < size; i++)
    {
        observations[i].x = (float)rand() / RAND_MAX;
        observations[i].y = (float)rand() / RAND_MAX;
    }
    cluster *clusters = kMeans(observations, size, k);
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
    test(atoi(argv[1]), atoi(argv[2]));

    return 0;
}
