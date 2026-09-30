#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <vector>
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
#include "streamCompaction.h"
#include "lighting.h"

constexpr bool ENABLE_STREAM_COMPACTION = true;
constexpr bool ENABLE_MATERIAL_SORTING = true;
constexpr bool ENABLE_ANTIALIASING = true;
constexpr bool ENABLE_MIS = true;

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
        
        glm::vec3 pix = image[index] / static_cast<float>(iter);
        pix = glm::max(pix, glm::vec3(0.0f));

        // Reinhard
        pix = pix / (glm::vec3(1.0f) + pix);

        // Gamma
        pix = glm::pow(pix, glm::vec3(1.0f / 2.2f));

        glm::ivec3 color(glm::clamp(pix, glm::vec3(0.0f), glm::vec3(1.0f)) * 255.0f + glm::vec3(0.5f));

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
static int* dev_material_keys = nullptr;
static int* dev_light_geom_indices = nullptr;
static int num_lights = 0;

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

    cudaMalloc(&dev_material_keys, pixelcount * sizeof(int));

    if (ENABLE_STREAM_COMPACTION)
    {
        streamCompaction::init(pixelcount);
    }

    std::vector<int> lightGeomIndices;

    for (int i = 0; i < scene->geoms.size(); ++i)
    {
        int materialId = scene->geoms[i].materialid;
        const Material& material = scene->materials[materialId];
        if (material.emittance > 0.0f)
        {
            lightGeomIndices.push_back(i);
        }
    }

    num_lights = lightGeomIndices.size();
    if (num_lights > 0)
    {
        cudaMalloc(&dev_light_geom_indices, num_lights * sizeof(int));
        cudaMemcpy(dev_light_geom_indices, lightGeomIndices.data(), num_lights * sizeof(int), cudaMemcpyHostToDevice);
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_material_keys);
    cudaFree(dev_light_geom_indices);

    streamCompaction::free();

    dev_light_geom_indices = nullptr;
    num_lights = 0;

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
        segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.radiance = glm::vec3(0.0f);
        segment.previousPosition = cam.position;
        segment.previousBsdfPdf = 0.0f;
        segment.previousBounceWasDelta = true;
        segment.previousLightSamplingEnabled = false;
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

    glm::vec3 intersectionPoint;
    glm::vec3 normal;
    int hitGeomId = -1;
    bool hitOutside = true;

    float t = sceneIntersectionTest(
        geoms,
        geoms_size,
        pathSegment.ray,
        intersectionPoint,
        normal,
        hitGeomId,
        hitOutside);

    ShadeableIntersection& intersection = intersections[path_index];

    if (t <= 0.0f)
    {
        intersection.t = -1.0f;
        intersection.geomId = -1;
        return;
    }

    intersection.t = t;
    intersection.surfaceNormal = normal;
    intersection.materialId = geoms[hitGeomId].materialid;
    intersection.geomId = hitGeomId;
    intersection.outside = hitOutside;
}

__global__ void shadeMaterial(
    int iter,
    int numPaths,
    const ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    const Geom* geoms,
    int numGeoms,
    const Material* materials,
    const int* lightGeomIndices,
    int numLights,
    bool enableMIS)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPaths)
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
        path.radiance += path.throughput * BACKGROUND_COLOR;
        path.remainingBounces = 0;
        return;
    }

    const Material& material = materials[intersection.materialId];

    glm::vec3 rayDirection = glm::normalize(path.ray.direction);
    glm::vec3 intersectionPoint = path.ray.origin + intersection.t * rayDirection;

    glm::vec3 wo = -rayDirection;

    // Reach a light
    if (material.emittance > 0.0f)
    {
        float weight = 1.0f;

        if (enableMIS
            && path.previousLightSamplingEnabled
            && !path.previousBounceWasDelta)
        {
            float pLight = lightPdfForHit(
                path.previousPosition,
                intersectionPoint,
                intersection.surfaceNormal,
                geoms[intersection.geomId],
                numLights);

            weight = powerHeuristic(path.previousBsdfPdf, pLight);
        }

        glm::vec3 emittedRadiance = material.color * material.emittance;

        path.radiance += path.throughput * emittedRadiance * weight;

        path.remainingBounces = 0;
        return;
    }

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, path.pixelIndex, path.remainingBounces);

    bool supportsDirectLighting = material.type == MATERIAL_DIFFUSE || material.type == MATERIAL_MICROFACET;

    bool useNEE = enableMIS && supportsDirectLighting && numLights > 0;

    bool canContinue = path.remainingBounces > 1;

    if (useNEE)
    {
        glm::vec3 directLighting = estimateDirectLightingNEE(
            intersectionPoint,
            intersection.surfaceNormal,
            wo,
            material,
            geoms,
            numGeoms,
            materials,
            lightGeomIndices,
            numLights,
            canContinue,
            rng);

        path.radiance += path.throughput * directLighting;
    }

    if (!canContinue)
    {
        path.remainingBounces = 0;
        return;
    }

    path.previousLightSamplingEnabled = useNEE;

    scatterRay(
        path,
        intersectionPoint,
        intersection.surfaceNormal,
        intersection.outside,
        material,
        rng);
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
        image[iterationPath.pixelIndex] += iterationPath.radiance;

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
        shadeMaterial << <numBlocksPaths, blockSize1d >> > (
            iter,
            numPaths,
            dev_intersections,
            dev_paths,
            dev_geoms,
            static_cast<int>(hst_scene->geoms.size()),
            dev_materials,
            dev_light_geom_indices,
            num_lights,
            ENABLE_MIS);
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
            numPaths = streamCompaction::compact(dev_paths, numPaths);
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
