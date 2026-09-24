#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    glm::vec3 newDir = calculateRandomDirectionInHemisphere(normal, rng);

    // Offset along the normal
    pathSegment.ray.origin = intersect + normal * 0.0001f;
    pathSegment.ray.direction = glm::normalize(newDir);
    pathSegment.color *= m.color;
    pathSegment.remainingBounces--;
}

__host__ __device__ void scatterMirror(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m)
{
    glm::vec3 incident = glm::normalize(pathSegment.ray.direction);
    glm::vec3 reflectedDir = glm::normalize(glm::reflect(incident, normal));

    pathSegment.ray.origin = intersect + normal * 0.0001f;
    pathSegment.ray.direction = reflectedDir;
    pathSegment.color *= m.color;
    pathSegment.remainingBounces--;
}

__host__ __device__ float schlickFresnel(float cosTheta, float etaI, float etaT)
{
    float r0 = (etaI - etaT) / (etaI + etaT);
    r0 *= r0;

    float oneMinusCos = 1.0f - cosTheta;
    float oneMinusCos2 = oneMinusCos * oneMinusCos;
    float oneMinusCos5 = oneMinusCos2 * oneMinusCos2 * oneMinusCos;

    return r0 + (1.0f - r0) * oneMinusCos5;
}

__host__ __device__ void scatterDielectric(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool outside,
    const Material& m,
    thrust::default_random_engine& rng)
{
    glm::vec3 incident = glm::normalize(pathSegment.ray.direction);

    glm::vec3 faceNormal = glm::normalize(normal);
    if (glm::dot(incident, faceNormal) > 0.0f)
    {
        faceNormal = -faceNormal;
    }

    float ior = m.indexOfRefraction;
    float etaI = outside ? 1.0f : ior;
    float etaT = outside ? ior : 1.0f;
    float eta = etaI / etaT;

    float cosTheta = glm::clamp(glm::dot(-incident, faceNormal), 0.0f, 1.0f);
    float sinThetaSquared = glm::max(0.0f, 1.0f - cosTheta * cosTheta);

    bool totalInternalReflection = eta * eta * sinThetaSquared > 1.0f;
    float reflectProbability = schlickFresnel(cosTheta, etaI, etaT);
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    bool chooseReflection = totalInternalReflection || u01(rng) < reflectProbability;

    glm::vec3 outgoing;
    if (chooseReflection)
    {
        outgoing = glm::reflect(incident, faceNormal);
    }
    else
    {
        outgoing = glm::refract(incident, faceNormal, eta);
    }
    outgoing = glm::normalize(outgoing);

    float offsetSign = glm::dot(outgoing, faceNormal) > 0.0f ? 1.0f : -1.0f;

    pathSegment.ray.origin = intersect + offsetSign * 0.0001f * faceNormal;
    pathSegment.ray.direction = outgoing;
    pathSegment.color *= m.color;
    pathSegment.remainingBounces--;
}