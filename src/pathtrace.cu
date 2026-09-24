#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

constexpr bool ENABLE_STREAM_COMPACTION = true;
constexpr bool ENABLE_MATERIAL_SORTING = true;
constexpr bool ENABLE_ANTIALIASING = true;

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static PathSegment* dev_paths_cache = nullptr;  // Compacted paths
static int* dev_path_alive = nullptr;
static int* dev_path_scan = nullptr;
static int path_scan_capacity = 0;
static int* dev_material_keys = nullptr;

__global__ void mapPathsToAlive(int n, int* alive, const PathSegment* paths)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        alive[idx] = paths[idx].remainingBounces > 0 ? 1 : 0;
    }
}

__global__ void pathUpSweep(int n, int depth, int* data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int stride = 1 << (depth + 1);
    int right = (idx + 1) * stride - 1;

    if (right < n)
    {
        int left = right - (stride >> 1);
        data[right] += data[left];
    }
}

__global__ void pathDownSweep(int n, int depth, int* data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int stride = 1 << (depth + 1);
    int right = (idx + 1) * stride - 1;

    if (right < n)
    {
        int left = right - (stride >> 1);

        int temp = data[left];
        data[left] = data[right];
        data[right] += temp;
    }
}

__global__ void scatterAlivePaths(
    int n,
    PathSegment* outputPaths,
    const PathSegment* inputPaths,
    const int* alive,
    const int* indices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n&& alive[idx] == 1)
    {
        outputPaths[indices[idx]] = inputPaths[idx];
    }
}

int compactPaths(int numPaths)
{
    if (numPaths <= 0)
    {
        return 0;
    }

    const int blockSize = 128;
    const int pathBlocks = (numPaths + blockSize - 1) / blockSize;
    int paddedSize = utilityCore::nexPowerOfTwo(numPaths);

    int scanDepth = 0;
    for (int size = paddedSize; size > 1; size >>= 1)
    {
        scanDepth++;
    }

    // Map paths to 0/1
    mapPathsToAlive << <pathBlocks, blockSize >> > (numPaths, dev_path_alive, dev_paths);

    // Initialize the padded scan buffer
    cudaMemset(dev_path_scan, 0, paddedSize * sizeof(int));

    cudaMemcpy(dev_path_scan, dev_path_alive, numPaths * sizeof(int), cudaMemcpyDeviceToDevice);

    // Up-sweep
    for (int depth = 0; depth < scanDepth; ++depth)
    {
        int activeThreads = paddedSize >> (depth + 1);
        int blocks = (activeThreads + blockSize - 1) / blockSize;
        pathUpSweep << <blocks, blockSize >> > (paddedSize, depth, dev_path_scan);
    }

    // Down-sweep
    cudaMemset(dev_path_scan + paddedSize - 1, 0, sizeof(int));

    for (int depth = scanDepth - 1; depth >= 0; --depth)
    {
        int activeThreads = paddedSize >> (depth + 1);
        int blocks = (activeThreads + blockSize - 1) / blockSize;
        pathDownSweep << <blocks, blockSize >> > (paddedSize, depth, dev_path_scan);
    }

    // Scatter
    scatterAlivePaths << <pathBlocks, blockSize >> > (
        numPaths,
        dev_paths_cache,
        dev_paths,
        dev_path_alive,
        dev_path_scan);

    // Number of active paths
    int lastIndex = 0;
    int lastAlive = 0;
    cudaMemcpy(&lastIndex, dev_path_scan + numPaths - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lastAlive, dev_path_alive + numPaths - 1, sizeof(int), cudaMemcpyDeviceToHost);

    int alivePathCount = lastIndex + lastAlive;

    // Ping-pong
    PathSegment* temp = dev_paths;
    dev_paths = dev_paths_cache;
    dev_paths_cache = temp;

    return alivePathCount;
}

__global__ void buildMaterialSortKeys(
    int numPaths,
    const ShadeableIntersection* intersections,
    const Material* materials,
    int* materialKeys)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPaths)
    {
        return;
    }

    ShadeableIntersection intersection = intersections[idx];
    if (intersection.t <= 0.0f)
    {
        materialKeys[idx] = MATERIAL_TYPE_COUNT;
    }
    else
    {
        materialKeys[idx] = static_cast<int>(materials[intersection.materialId].type);
    }
}

void sortPathsByMaterial(int numPaths)
{
    if (numPaths <= 1)
    {
        return;
    }

    thrust::device_ptr<int> keyBegin = thrust::device_pointer_cast(dev_material_keys);
    thrust::device_ptr<PathSegment> pathBegin = thrust::device_pointer_cast(dev_paths);
    thrust::device_ptr<ShadeableIntersection> intersectionBegin = thrust::device_pointer_cast(dev_intersections);
    
    auto valueBegin = thrust::make_zip_iterator(thrust::make_tuple(pathBegin, intersectionBegin));

    thrust::sort_by_key(thrust::device, keyBegin, keyBegin + numPaths, valueBegin);
}

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    path_scan_capacity = utilityCore::nexPowerOfTwo(pixelcount);
    cudaMalloc(&dev_paths_cache, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_path_alive, pixelcount * sizeof(int));
    cudaMalloc(&dev_path_scan, path_scan_capacity * sizeof(int));

    cudaMalloc(&dev_material_keys, pixelcount * sizeof(int));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_paths_cache);
    cudaFree(dev_path_alive);
    cudaFree(dev_path_scan);
    cudaFree(dev_material_keys);

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, bool enableAntialiasing)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;

        float sampleX = static_cast<float>(x) + 0.5f;
        float sampleY = static_cast<float>(y) + 0.5f;

        if (enableAntialiasing)
        {
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
            thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
            sampleX = static_cast<float>(x) + u01(rng);
            sampleY = static_cast<float>(y) + u01(rng);
        }

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (sampleX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (sampleY - (float)cam.resolution.y * 0.5f)
        );
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index >= num_paths)
    {
        return;
    }

    if (pathSegments[path_index].remainingBounces <= 0)
    {
        intersections[path_index].t = -1.0f;
        return;
    }

    PathSegment pathSegment = pathSegments[path_index];

    float t;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;

    // naive parse through global geoms

    for (int i = 0; i < geoms_size; i++)
    {
        Geom& geom = geoms[i];

        if (geom.type == CUBE)
        {
            t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == SPHERE)
        {
            t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        }
        // TODO: add more intersection tests here... triangle? metaball? CSG?

        // Compute the minimum t from the intersection tests to determine what
        // scene geometry object was hit first.
        if (t > 0.0f && t_min > t)
        {
            t_min = t;
            hit_geom_index = i;
            intersect_point = tmp_intersect;
            normal = tmp_normal;
        }
    }

    if (hit_geom_index == -1)
    {
        intersections[path_index].t = -1.0f;
    }
    else
    {
        // The ray hits something
        intersections[path_index].t = t_min;
        intersections[path_index].materialId = geoms[hit_geom_index].materialid;
        intersections[path_index].surfaceNormal = normal;
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

__global__ void shadeDiffuseMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths)
    {
        return;
    }

    PathSegment& path = pathSegments[idx];
    if (path.remainingBounces <= 0)
    {
        return;
    }

    ShadeableIntersection intersection = shadeableIntersections[idx];

    // Ray missed the scene
    if (intersection.t <= 0.0f)
    {
        path.color = BACKGROUND_COLOR;
        path.remainingBounces = 0;
        return;
    }

    Material material = materials[intersection.materialId];

    // Reach a light
    if (material.emittance > 0.0f)
    {
        path.color *= material.color * material.emittance;
        path.remainingBounces = 0;
        return;
    }

    // The final surface interaction but did not reach a light
    if (path.remainingBounces <= 1)
    {
        path.color = glm::vec3(0.0f);
        path.remainingBounces = 0;
        return;
    }

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, path.pixelIndex, path.remainingBounces);

    glm::vec3 intersectionPoint = path.ray.origin + intersection.t * glm::normalize(path.ray.direction);

    // Scatter
    switch (material.type)
    {
    case MATERIAL_DIFFUSE:
        scatterRay(path, intersectionPoint, intersection.surfaceNormal, material, rng);
        break;

    case MATERIAL_MIRROR:
        scatterMirror(path, intersectionPoint, intersection.surfaceNormal, material);
        break;

    default:
        path.color = glm::vec3(0.0f);
        path.remainingBounces = 0;
        break;
    }
}

__global__ void gatherTerminatedPaths(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= nPaths)
    {
        return;
    }

    // Gather remainingBounces == 0
    PathSegment& iterationPath = iterationPaths[index];
    if (iterationPath.remainingBounces == 0)
    {
        image[iterationPath.pixelIndex] += iterationPath.color;

        // Mark the contribution as consumed
        iterationPath.remainingBounces = -1;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths, ENABLE_ANTIALIASING);
    checkCUDAError("generate camera ray");

    int numPaths = pixelcount;

    for (int depth = 0; depth < traceDepth && numPaths > 0; ++depth)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, numPaths * sizeof(ShadeableIntersection));

        // Tracing
        int numBlocksPaths = (numPaths + blockSize1d - 1) / blockSize1d;
        computeIntersections << <numBlocksPaths, blockSize1d >> > (
            depth,
            numPaths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
            );
        checkCUDAError("trace one bounce");

        // Sorting
        if (ENABLE_MATERIAL_SORTING)
        {
            buildMaterialSortKeys << <numBlocksPaths, blockSize1d >> > (
                numPaths,
                dev_intersections,
                dev_materials,
                dev_material_keys);
            checkCUDAError("build material sort keys");

            sortPathsByMaterial(numPaths);
            checkCUDAError("sort paths by material");
        }

        // Shading
        shadeDiffuseMaterial << <numBlocksPaths, blockSize1d >> > (
            iter,
            numPaths,
            dev_intersections,
            dev_paths,
            dev_materials);
        checkCUDAError("shade one bounce");

        // Gather
        gatherTerminatedPaths << <numBlocksPaths, blockSize1d >> > (
            numPaths,
            dev_image,
            dev_paths);
        checkCUDAError("gather terminated paths");

        // Compact
        if (ENABLE_STREAM_COMPACTION)
        {
            numPaths = compactPaths(numPaths);
            checkCUDAError("compact active paths");
        }

        if (guiData != nullptr)
        {
            guiData->TracedDepth = depth + 1;
        }
    }

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
