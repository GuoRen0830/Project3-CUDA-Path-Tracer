#include "lighting.h"
#include "interactions.h"
#include "intersections.h"
#include "utilities.h"

#include <cmath>
#include <thrust/random.h>

__host__ __device__ LightSample invalidLightSample()
{
    LightSample sample{};

    sample.directionToLight = glm::vec3(0.0f);
    sample.emittedRadiance = glm::vec3(0.0f);

    sample.pdf = 0.0f;
    sample.geomId = -1;
    sample.valid = false;

    return sample;
}

__host__ __device__ float geometrySurfaceArea(const Geom& geom)
{
    float scaleX = fabsf(geom.scale.x);
    float scaleY = fabsf(geom.scale.y);
    float scaleZ = fabsf(geom.scale.z);

    switch (geom.type)
    {
    case CUBE:
        return 2.0f * (scaleX * scaleY + scaleX * scaleZ + scaleY * scaleZ);

    case SPHERE:
    {
        float radius = 0.5f * scaleX;
        return 4.0f * PI * radius * radius;
    }

    case RECTANGLE:
        return scaleX * scaleY;

    default:
        return 0.0f;
    }
}

__host__ __device__ bool sampleCubeSurface(
    const Geom& geom,
    thrust::default_random_engine& rng,
    glm::vec3& position,
    glm::vec3& normal,
    float& areaPdf)
{
    float scaleX = fabsf(geom.scale.x);
    float scaleY = fabsf(geom.scale.y);
    float scaleZ = fabsf(geom.scale.z);

    float areaYZ = scaleY * scaleZ;
    float areaXZ = scaleX * scaleZ;
    float areaXY = scaleX * scaleY;

    float totalArea = 2.0f * (areaYZ + areaXZ + areaXY);
    if (totalArea <= EPSILON)
    {
        return false;
    }

    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    float selector = u01(rng) * totalArea;
    float u = u01(rng) - 0.5f;
    float v = u01(rng) - 0.5f;

    glm::vec3 localPosition(0.0f);
    glm::vec3 localNormal(0.0f);

    if (selector < 2.0f * areaYZ)
    {
        // Sample one of the two X faces
        bool positiveFace = selector >= areaYZ;
        localPosition = glm::vec3(positiveFace ? 0.5f : -0.5f, u, v);
        localNormal = glm::vec3(positiveFace ? 1.0f : -1.0f, 0.0f, 0.0f);
    }
    else if (selector < 2.0f * areaYZ + 2.0f * areaXZ)
    {
        // Sample one of the two Y faces
        selector -= 2.0f * areaYZ;
        bool positiveFace = selector >= areaXZ;
        localPosition = glm::vec3(u, positiveFace ? 0.5f : -0.5f, v);
        localNormal = glm::vec3(0.0f, positiveFace ? 1.0f : -1.0f, 0.0f);
    }
    else
    {
        // Sample one of the two Z faces
        selector -= 2.0f * areaYZ + 2.0f * areaXZ;
        bool positiveFace = selector >= areaXY;
        localPosition = glm::vec3(u, v, positiveFace ? 0.5f : -0.5f);
        localNormal = glm::vec3(0.0f, 0.0f, positiveFace ? 1.0f : -1.0f);
    }

    position = multiplyMV(
        geom.transform,
        glm::vec4(localPosition, 1.0f));

    normal = glm::normalize(multiplyMV(
        geom.invTranspose,
        glm::vec4(localNormal, 0.0f)));

    areaPdf = 1.0f / totalArea;

    return true;
}

__host__ __device__ float sphereLightPdf(
    glm::vec3 referencePoint,
    const Geom& sphere)
{
    float scaleX = fabsf(sphere.scale.x);
    float scaleY = fabsf(sphere.scale.y);
    float scaleZ = fabsf(sphere.scale.z);

    bool uniformScale = fabsf(scaleX - scaleY) <= EPSILON && fabsf(scaleX - scaleZ) <= EPSILON;
    if (!uniformScale)
    {
        return 0.0f;
    }

    glm::vec3 center = multiplyMV(
        sphere.transform,
        glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

    float radius = 0.5f * scaleX;

    glm::vec3 toCenter = center - referencePoint;
    float distanceSquared = glm::dot(toCenter, toCenter);
    float radiusSquared = radius * radius;

    if (distanceSquared <= radiusSquared)
    {
        return 0.0f;
    }

    float sinThetaMaxSquared = radiusSquared / distanceSquared;

    float cosThetaMax = sqrtf(glm::max(0.0f, 1.0f - sinThetaMaxSquared));

    float solidAngle = TWO_PI * (1.0f - cosThetaMax);
    if (solidAngle <= EPSILON)
    {
        return 0.0f;
    }

    return 1.0f / solidAngle;
}

__host__ __device__ LightSample sampleSphereLight(
    glm::vec3 referencePoint,
    const Geom& sphere,
    const Material& lightMaterial,
    int geomId,
    float selectionPdf,
    thrust::default_random_engine& rng)
{
    LightSample sample = invalidLightSample();

    float scaleX = fabsf(sphere.scale.x);
    float scaleY = fabsf(sphere.scale.y);
    float scaleZ = fabsf(sphere.scale.z);

    bool uniformScale = fabsf(scaleX - scaleY) <= EPSILON && fabsf(scaleX - scaleZ) <= EPSILON;
    if (!uniformScale)
    {
        return sample;
    }

    glm::vec3 center = multiplyMV(
        sphere.transform,
        glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

    float radius = 0.5f * scaleX;
    float radiusSquared = radius * radius;

    glm::vec3 toCenter = center - referencePoint;
    float distanceSquared = glm::dot(toCenter, toCenter);
    if (distanceSquared <= radiusSquared)
    {
        return sample;
    }

    float distanceToCenter = sqrtf(distanceSquared);

    glm::vec3 centerDirection = toCenter / distanceToCenter;

    float sinThetaMaxSquared = radiusSquared / distanceSquared;

    float cosThetaMax = sqrtf(glm::max(
        0.0f,
        1.0f - sinThetaMaxSquared));

    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    float u1 = u01(rng);
    float u2 = u01(rng);

    float cosTheta = (1.0f - u1) + u1 * cosThetaMax;
    float sinTheta = sqrtf(glm::max(0.0f, 1.0f - cosTheta * cosTheta));

    float phi = TWO_PI * u2;

    glm::vec3 localDirection(
        sinTheta * cosf(phi),
        sinTheta * sinf(phi),
        cosTheta);

    glm::vec3 direction = glm::normalize(
        localToWorld(
            localDirection,
            centerDirection));

    glm::vec3 originToCenter = referencePoint - center;

    float b = glm::dot(originToCenter, direction);
    float c = glm::dot(originToCenter, originToCenter) - radiusSquared;

    float discriminant = b * b - c;
    if (discriminant <= 0.0f)
    {
        return sample;
    }

    float distance = -b - sqrtf(discriminant);
    if (distance <= EPSILON)
    {
        return sample;
    }

    float conditionalPdf = sphereLightPdf(referencePoint, sphere);
    if (conditionalPdf <= 0.0f)
    {
        return sample;
    }

    sample.directionToLight = direction;
    sample.emittedRadiance = lightMaterial.color * lightMaterial.emittance;

    sample.pdf = selectionPdf * conditionalPdf;
    sample.geomId = geomId;
    sample.valid = sample.pdf > 0.0f;

    return sample;
}

__host__ __device__ bool sampleRectangleSurface(
    const Geom& geom,
    thrust::default_random_engine& rng,
    glm::vec3& position,
    glm::vec3& normal,
    float& areaPdf)
{
    float area = fabsf(geom.scale.x * geom.scale.y);
    if (area <= EPSILON)
    {
        return false;
    }

    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    glm::vec3 localPosition(
        u01(rng) - 0.5f,
        u01(rng) - 0.5f,
        0.0f);

    position = multiplyMV(
        geom.transform,
        glm::vec4(localPosition, 1.0f));

    normal = glm::normalize(multiplyMV(
        geom.invTranspose,
        glm::vec4(0.0f, 0.0f, 1.0f, 0.0f)));

    areaPdf = 1.0f / area;

    return true;
}

__host__ __device__ LightSample sampleLight(
    glm::vec3 referencePoint,
    const Geom* geoms,
    const Material* materials,
    const int* lightGeomIndices,
    int numLights,
    thrust::default_random_engine& rng)
{
    LightSample sample = invalidLightSample();

    if (numLights <= 0
        || geoms == nullptr
        || materials == nullptr
        || lightGeomIndices == nullptr)
    {
        return sample;
    }

    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    int lightIndex = static_cast<int>(u01(rng) * static_cast<float>(numLights));
    lightIndex = glm::min(lightIndex, numLights - 1);

    int geomId = lightGeomIndices[lightIndex];
    Geom lightGeom = geoms[geomId];

    Material lightMaterial = materials[lightGeom.materialid];

    float selectionPdf = 1.0f / static_cast<float>(numLights);

    if (lightGeom.type == SPHERE)
    {
        return sampleSphereLight(
            referencePoint,
            lightGeom,
            lightMaterial,
            geomId,
            selectionPdf,
            rng);
    }

    glm::vec3 lightPosition;
    glm::vec3 lightNormal;
    float areaPdf = 0.0f;

    bool sampled = false;

    switch (lightGeom.type)
    {
    case CUBE:
        sampled = sampleCubeSurface(
            lightGeom,
            rng,
            lightPosition,
            lightNormal,
            areaPdf);
        break;

    case RECTANGLE:
        sampled = sampleRectangleSurface(
            lightGeom,
            rng,
            lightPosition,
            lightNormal,
            areaPdf);
        break;

    default:
        return sample;
    }

    if (!sampled || areaPdf <= 0.0f)
    {
        return sample;
    }

    glm::vec3 toLight = lightPosition - referencePoint;

    float distanceSquared = glm::dot(toLight, toLight);
    if (distanceSquared <= EPSILON)
    {
        return sample;
    }

    float distance = sqrtf(distanceSquared);
    glm::vec3 direction = toLight / distance;

    float cosThetaLight = glm::dot(lightNormal, -direction);
    if (cosThetaLight <= EPSILON)
    {
        return sample;
    }

    // Convert the area PDF to a solid-angle PDF
    float solidAnglePdf =
        selectionPdf
        * areaPdf
        * distanceSquared
        / cosThetaLight;

    sample.directionToLight = direction;
    sample.emittedRadiance = lightMaterial.color * lightMaterial.emittance;

    sample.pdf = solidAnglePdf;
    sample.geomId = geomId;
    sample.valid = solidAnglePdf > 0.0f;

    return sample;
}

__host__ __device__ glm::vec3 estimateDirectLightingNEE(
    glm::vec3 referencePoint,
    glm::vec3 normal,
    glm::vec3 wo,
    const Material& material,
    const Geom* geoms,
    int numGeoms,
    const Material* materials,
    const int* lightGeomIndices,
    int numLights,
    bool bsdfStrategyAvailable,
    thrust::default_random_engine& rng)
{
    if (numLights <= 0)
    {
        return glm::vec3(0.0f);
    }

    wo = glm::normalize(wo);
    glm::vec3 faceNormal = faceForwardNormal(normal, wo);

    LightSample lightSample = sampleLight(
        referencePoint,
        geoms,
        materials,
        lightGeomIndices,
        numLights,
        rng);

    if (!lightSample.valid || lightSample.pdf <= 0.0f)
    {
        return glm::vec3(0.0f);
    }

    float cosTheta = glm::max(glm::dot(faceNormal, lightSample.directionToLight), 0.0f);
    if (cosTheta <= 0.0f)
    {
        return glm::vec3(0.0f);
    }

    glm::vec3 f = evaluateBSDF(
        material,
        faceNormal,
        wo,
        lightSample.directionToLight);

    if (!isLightVisible(
        referencePoint,
        lightSample,
        geoms,
        numGeoms))
    {
        return glm::vec3(0.0f);
    }

    float weight = 1.0f;

    if (bsdfStrategyAvailable)
    {
        float pBsdf = bsdfPdf(
            material,
            faceNormal,
            wo,
            lightSample.directionToLight);

        weight = powerHeuristic(lightSample.pdf, pBsdf);
    }

    return weight
        * f
        * lightSample.emittedRadiance
        * cosTheta
        / lightSample.pdf;
}

__host__ __device__ bool isLightVisible(
    glm::vec3 referencePoint,
    const LightSample& lightSample,
    const Geom* geoms,
    int numGeoms)
{
    if (!lightSample.valid)
    {
        return false;
    }

    Ray shadowRay = spawnRay(referencePoint, lightSample.directionToLight);

    glm::vec3 hitPoint;
    glm::vec3 hitNormal;
    int hitGeomId = -1;
    bool hitOutside = true;

    float hitDistance = sceneIntersectionTest(
        geoms,
        numGeoms,
        shadowRay,
        hitPoint,
        hitNormal,
        hitGeomId,
        hitOutside);

    return hitDistance > 0.0f && hitGeomId == lightSample.geomId;
}

__host__ __device__ float lightPdfForHit(
    glm::vec3 referencePoint,
    glm::vec3 lightPoint,
    glm::vec3 lightNormal,
    const Geom& lightGeom,
    int numLights)
{
    if (numLights <= 0)
    {
        return 0.0f;
    }

    float selectionPdf = 1.0f / static_cast<float>(numLights);

    if (lightGeom.type == SPHERE)
    {
        float conditionalPdf = sphereLightPdf(referencePoint, lightGeom);
        return selectionPdf * conditionalPdf;
    }

    float area = geometrySurfaceArea(lightGeom);
    if (area <= EPSILON)
    {
        return 0.0f;
    }

    glm::vec3 toLight = lightPoint - referencePoint;

    float distanceSquared = glm::dot(toLight, toLight);
    if (distanceSquared <= EPSILON)
    {
        return 0.0f;
    }

    glm::vec3 direction = toLight / sqrtf(distanceSquared);

    float cosThetaLight = glm::dot(glm::normalize(lightNormal), -direction);
    if (cosThetaLight <= EPSILON)
    {
        return 0.0f;
    }

    float areaPdf = 1.0f / area;

    return selectionPdf
        * areaPdf
        * distanceSquared
        / cosThetaLight;
}

__host__ __device__ float powerHeuristic(
    float sampledPdf,
    float otherPdf)
{
    float scale = fmaxf(sampledPdf, otherPdf);
    if (scale <= 0.0f)
    {
        return 0.0f;
    }

    float a = sampledPdf / scale;
    float b = otherPdf / scale;

    return (a * a) / (a * a + b * b);
}