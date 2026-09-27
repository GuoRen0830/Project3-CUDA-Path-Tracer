#include "streamCompaction.h"
#include "utilities.h"
#include <cuda_runtime.h>

namespace
{
    constexpr int BLOCK_SIZE = 128;

    PathSegment* dev_paths_cache = nullptr;
    int* dev_path_alive = nullptr;
    int* dev_path_scan = nullptr;

    __global__ void mapPathsToAlive(
        int numPaths,
        int* alive,
        const PathSegment* paths)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < numPaths)
        {
            alive[idx] = paths[idx].remainingBounces > 0 ? 1 : 0;
        }
    }

    __global__ void pathUpSweep(
        int numElements,
        int depth,
        int* data)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int stride = 1 << (depth + 1);
        int right = (idx + 1) * stride - 1;

        if (right < numElements)
        {
            int left = right - (stride >> 1);
            data[right] += data[left];
        }
    }

    __global__ void pathDownSweep(
        int numElements,
        int depth,
        int* data)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int stride = 1 << (depth + 1);
        int right = (idx + 1) * stride - 1;

        if (right < numElements)
        {
            int left = right - (stride >> 1);

            int temp = data[left];
            data[left] = data[right];
            data[right] += temp;
        }
    }

    __global__ void scatterAlivePaths(
        int numPaths,
        PathSegment* outputPaths,
        const PathSegment* inputPaths,
        const int* alive,
        const int* indices)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < numPaths && alive[idx] == 1)
        {
            outputPaths[indices[idx]] = inputPaths[idx];
        }
    }
}

namespace streamCompaction
{
    void init(int maxPathCount)
    {
        if (maxPathCount <= 0)
        {
            return;
        }

        int scanCapacity = utilityCore::nexPowerOfTwo(maxPathCount);

        cudaMalloc(&dev_paths_cache, maxPathCount * sizeof(PathSegment));
        cudaMalloc(&dev_path_alive, maxPathCount * sizeof(int));
        cudaMalloc(&dev_path_scan, scanCapacity * sizeof(int));
    }

    void free()
    {
        cudaFree(dev_paths_cache);
        cudaFree(dev_path_alive);
        cudaFree(dev_path_scan);

        dev_paths_cache = nullptr;
        dev_path_alive = nullptr;
        dev_path_scan = nullptr;
    }

    int compact(PathSegment*& paths, int numPaths)
    {
        if (numPaths <= 0)
        {
            return 0;
        }

        int numBlocks = (numPaths + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int paddedSize = utilityCore::nexPowerOfTwo(numPaths);

        int scanDepth = 0;
        for (int size = paddedSize; size > 1; size >>= 1)
        {
            scanDepth++;
        }

        // Mark active paths
        mapPathsToAlive << <numBlocks, BLOCK_SIZE >> > (
            numPaths,
            dev_path_alive,
            paths);

        // Initialize the padded scan input
        cudaMemset(dev_path_scan, 0, paddedSize * sizeof(int));
        cudaMemcpy(
            dev_path_scan,
            dev_path_alive,
            numPaths * sizeof(int),
            cudaMemcpyDeviceToDevice);

        // Up-Sweep
        for (int depth = 0; depth < scanDepth; ++depth)
        {
            int activeThreads = paddedSize >> (depth + 1);
            int blocks = (activeThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;

            pathUpSweep << <blocks, BLOCK_SIZE >> > (
                paddedSize,
                depth,
                dev_path_scan);
        }

        // Down-Sweep
        cudaMemset(
            dev_path_scan + paddedSize - 1,
            0,
            sizeof(int));

        for (int depth = scanDepth - 1; depth >= 0; --depth)
        {
            int activeThreads = paddedSize >> (depth + 1);
            int blocks = (activeThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;

            pathDownSweep << <blocks, BLOCK_SIZE >> > (
                paddedSize,
                depth,
                dev_path_scan);
        }

        // Scatter
        scatterAlivePaths << <numBlocks, BLOCK_SIZE >> > (
            numPaths,
            dev_paths_cache,
            paths,
            dev_path_alive,
            dev_path_scan);

        int lastIndex = 0;
        int lastAlive = 0;

        cudaMemcpy(
            &lastIndex,
            dev_path_scan + numPaths - 1,
            sizeof(int),
            cudaMemcpyDeviceToHost);

        cudaMemcpy(
            &lastAlive,
            dev_path_alive + numPaths - 1,
            sizeof(int),
            cudaMemcpyDeviceToHost);

        int alivePathCount = lastIndex + lastAlive;

        // Ping-Pong
        PathSegment* temp = paths;
        paths = dev_paths_cache;
        dev_paths_cache = temp;

        return alivePathCount;
    }
}