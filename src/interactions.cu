#include "interactions.h"

#include "utilities.h"

#include <cmath>
#include <thrust/random.h>

__host__ __device__ glm::vec3 faceForwardNormal(glm::vec3 normal, glm::vec3 wo)
{
    normal = glm::normalize(normal);
    return glm::dot(normal, wo) >= 0.0f ? normal : -normal;
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

__host__ __device__ BSDFSample invalidBSDFSample()
{
    BSDFSample sample{};
    sample.direction = glm::vec3(0.0f);
    sample.f = glm::vec3(0.0f);
    sample.pdf = 0.0f;
    sample.isSpecular = false;
    sample.valid = false;
    return sample;
}

__host__ __device__ glm::vec3 localToWorld(
    glm::vec3 localDirection,
    glm::vec3 normal)
{
    normal = glm::normalize(normal);

    glm::vec3 directionNotNormal =
        fabsf(normal.z) < 0.999f
        ? glm::vec3(0.0f, 0.0f, 1.0f)
        : glm::vec3(0.0f, 1.0f, 0.0f);

    glm::vec3 tangent = glm::normalize(glm::cross(directionNotNormal, normal));
    glm::vec3 bitangent = glm::cross(normal, tangent);

    return localDirection.x * tangent
        + localDirection.y * bitangent
        + localDirection.z * normal;
}

__host__ __device__ glm::vec3 schlickFresnel(
    float cosTheta,
    glm::vec3 f0)
{
    cosTheta = glm::clamp(cosTheta, 0.0f, 1.0f);

    float oneMinusCos = 1.0f - cosTheta;
    float oneMinusCos2 = oneMinusCos * oneMinusCos;
    float oneMinusCos5 = oneMinusCos2 * oneMinusCos2 * oneMinusCos;

    return f0 + (glm::vec3(1.0f) - f0) * oneMinusCos5;
}

__host__ __device__ float trowbridgeReitzD(
    glm::vec3 normal,
    glm::vec3 halfVector,
    float roughness)
{
    float cosTheta = glm::dot(normal, halfVector);

    if (cosTheta <= 0.0f)
    {
        return 0.0f;
    }

    float alpha = glm::clamp(roughness, 0.001f, 1.0f);
    float alphaSquared = alpha * alpha;
    float cosThetaSquared = cosTheta * cosTheta;

    float denominator = cosThetaSquared * (alphaSquared - 1.0f) + 1.0f;

    return alphaSquared / (PI * denominator * denominator);
}

__host__ __device__ float trowbridgeReitzLambda(
    glm::vec3 normal,
    glm::vec3 direction,
    float roughness)
{
    float cosTheta = fabsf(glm::dot(normal, direction));
    if (cosTheta <= EPSILON)
    {
        return 1.0e20f;
    }

    float cosThetaSquared = cosTheta * cosTheta;
    float sinThetaSquared = glm::max(0.0f, 1.0f - cosThetaSquared);
    float tanThetaSquared = sinThetaSquared / cosThetaSquared;

    float alpha = glm::clamp(roughness, 0.001f, 1.0f);
    float alphaSquared = alpha * alpha;

    return 0.5f * (-1.0f + sqrtf(1.0f + alphaSquared * tanThetaSquared));
}

__host__ __device__ float trowbridgeReitzG(
    glm::vec3 normal,
    glm::vec3 wo,
    glm::vec3 wi,
    float roughness)
{
    return 1.0f / (1.0f + trowbridgeReitzLambda(normal, wo, roughness) + trowbridgeReitzLambda(normal, wi, roughness));
}

__host__ __device__ glm::vec3 sampleGGXHalfVector(
    glm::vec3 normal,
    float roughness,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    float u1 = glm::min(u01(rng), 0.999999f);
    float u2 = u01(rng);

    float alpha = glm::clamp(roughness, 0.001f, 1.0f);
    float alphaSquared = alpha * alpha;

    float tanThetaSquared = alphaSquared * u1 / (1.0f - u1);

    float cosTheta = 1.0f / sqrtf(1.0f + tanThetaSquared);
    float sinTheta = sqrtf(glm::max(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = TWO_PI * u2;

    glm::vec3 localHalfVector(
        sinTheta * cosf(phi),
        sinTheta * sinf(phi),
        cosTheta);

    return glm::normalize(localToWorld(localHalfVector, normal));
}

__host__ __device__ glm::vec3 evaluateMicrofacetReflection(
    const Material& m,
    glm::vec3 normal,
    glm::vec3 wo,
    glm::vec3 wi)
{
    float cosThetaO = glm::dot(normal, wo);
    float cosThetaI = glm::dot(normal, wi);
    if (cosThetaO <= 0.0f || cosThetaI <= 0.0f)
    {
        return glm::vec3(0.0f);
    }

    glm::vec3 halfVector = wo + wi;
    float halfVectorLengthSquared = glm::dot(halfVector, halfVector);
    if (halfVectorLengthSquared <= EPSILON)
    {
        return glm::vec3(0.0f);
    }

    halfVector = glm::normalize(halfVector);
    if (glm::dot(normal, halfVector) < 0.0f)
    {
        halfVector = -halfVector;
    }

    float D = trowbridgeReitzD(normal, halfVector, m.roughness);
    float G = trowbridgeReitzG(normal, wo, wi, m.roughness);

    glm::vec3 f0 = glm::clamp(m.color, glm::vec3(0.0f), glm::vec3(1.0f));

    glm::vec3 F = schlickFresnel(fabsf(glm::dot(wi, halfVector)), f0);

    return F * D * G / (4.0f * cosThetaI * cosThetaO);
}

__host__ __device__ float microfacetReflectionPdf(
    const Material& m,
    glm::vec3 normal,
    glm::vec3 wo,
    glm::vec3 wi)
{
    if (glm::dot(normal, wo) <= 0.0f || glm::dot(normal, wi) <= 0.0f)
    {
        return 0.0f;
    }

    glm::vec3 halfVector = wo + wi;
    if (glm::dot(halfVector, halfVector) <= EPSILON)
    {
        return 0.0f;
    }

    halfVector = glm::normalize(halfVector);
    if (glm::dot(normal, halfVector) < 0.0f)
    {
        halfVector = -halfVector;
    }

    float woDotHalf = fabsf(glm::dot(wo, halfVector));
    if (woDotHalf <= EPSILON)
    {
        return 0.0f;
    }

    float halfVectorPdf = trowbridgeReitzD( normal, halfVector, m.roughness)
        * glm::max(glm::dot(normal, halfVector), 0.0f);

    return halfVectorPdf / (4.0f * woDotHalf);
}

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

__host__ __device__ glm::vec3 evaluateBSDF(
    const Material& m,
    glm::vec3 normal,
    glm::vec3 wo,
    glm::vec3 wi)
{
    glm::vec3 faceNormal = faceForwardNormal(normal, wo);

    float cosThetaO = glm::dot(faceNormal, wo);
    float cosThetaI = glm::dot(faceNormal, wi);

    if (cosThetaO <= 0.0f || cosThetaI <= 0.0f)
    {
        return glm::vec3(0.0f);
    }

    switch (m.type)
    {
    case MATERIAL_DIFFUSE:
        return m.color / PI;

    case MATERIAL_MICROFACET:
        return evaluateMicrofacetReflection(m, faceNormal, wo, wi);

    case MATERIAL_MIRROR:
    case MATERIAL_DIELECTRIC:
    default:
        return glm::vec3(0.0f);
    }
}

__host__ __device__ float bsdfPdf(
    const Material& m,
    glm::vec3 normal,
    glm::vec3 wo,
    glm::vec3 wi)
{
    glm::vec3 faceNormal = faceForwardNormal(normal, wo);

    if (glm::dot(faceNormal, wo) <= 0.0f)
    {
        return 0.0f;
    }

    switch (m.type)
    {
    case MATERIAL_DIFFUSE:
    {
        float cosTheta = glm::dot(faceNormal, wi);
        return cosTheta > 0.0f ? cosTheta / PI : 0.0f;
    }

    case MATERIAL_MICROFACET:
        return microfacetReflectionPdf(m, faceNormal, wo, wi);

    case MATERIAL_MIRROR:
    case MATERIAL_DIELECTRIC:
    default:
        return 0.0f;
    }
}

__host__ __device__ BSDFSample sampleBSDF(
    const Material& m,
    glm::vec3 normal,
    bool outside,
    glm::vec3 wo,
    thrust::default_random_engine& rng)
{
    BSDFSample sample = invalidBSDFSample();

    wo = glm::normalize(wo);
    glm::vec3 faceNormal = faceForwardNormal(normal, wo);
    glm::vec3 incident = -wo;

    switch (m.type)
    {
    case MATERIAL_DIFFUSE:
    {
        sample.direction = calculateRandomDirectionInHemisphere(faceNormal, rng);
        sample.f = evaluateBSDF(m, faceNormal, wo, sample.direction);
        sample.pdf = bsdfPdf(m, faceNormal, wo, sample.direction);
        sample.isSpecular = false;
        break;
    }

    case MATERIAL_MIRROR:
    {
        sample.direction = glm::normalize(glm::reflect(incident, faceNormal));
        sample.f = m.color;
        sample.pdf = 1.0f;
        sample.isSpecular = true;
        break;
    }

    case MATERIAL_DIELECTRIC:
    {
        float etaI = outside ? 1.0f : m.indexOfRefraction;
        float etaT = outside ? m.indexOfRefraction : 1.0f;
        float eta = etaI / etaT;

        float cosTheta = glm::clamp(glm::dot(wo, faceNormal), 0.0f, 1.0f);
        float sinThetaSquared = glm::max(0.0f, 1.0f - cosTheta * cosTheta);

        bool totalInternalReflection = eta * eta * sinThetaSquared > 1.0f;

        float reflectProbability = schlickFresnel(cosTheta, etaI, etaT);

        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
        bool chooseReflection = totalInternalReflection || u01(rng) < reflectProbability;

        if (chooseReflection)
        {
            sample.direction = glm::normalize(glm::reflect(incident, faceNormal));
            sample.pdf = totalInternalReflection ? 1.0f : reflectProbability;
        }
        else
        {
            sample.direction = glm::normalize(glm::refract(incident, faceNormal, eta));
            sample.pdf = 1.0f - reflectProbability;
        }

        sample.f = m.color * sample.pdf;
        sample.isSpecular = true;
        break;
    }

    case MATERIAL_MICROFACET:
    {
        glm::vec3 halfVector = sampleGGXHalfVector(faceNormal, m.roughness, rng);

        float woDotHalf = glm::dot(wo, halfVector);
        if (woDotHalf <= EPSILON)
        {
            return sample;
        }

        sample.direction = glm::normalize(glm::reflect(incident, halfVector));
        if (glm::dot(faceNormal, sample.direction) <= 0.0f)
        {
            return sample;
        }

        sample.f = evaluateBSDF(m, faceNormal, wo, sample.direction);
        sample.pdf = bsdfPdf(m, faceNormal, wo, sample.direction);
        sample.isSpecular = false;
        break;
    }

    default:
        return sample;
    }

    sample.valid = sample.pdf > 0.0f && glm::dot(sample.direction, sample.direction) > 0.0f;

    return sample;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool outside,
    const Material &m,
    thrust::default_random_engine &rng)
{
    glm::vec3 wo = -glm::normalize(pathSegment.ray.direction);

    BSDFSample sample = sampleBSDF(m, normal, outside, wo, rng);
    if (!sample.valid)
    {
        pathSegment.color = glm::vec3(0.0f);
        pathSegment.remainingBounces = 0;
        return;
    }

    glm::vec3 faceNormal = faceForwardNormal(normal, wo);

    float cosTheta = sample.isSpecular ? 1.0f : fabsf(glm::dot(faceNormal, sample.direction));

    pathSegment.color *= sample.f * cosTheta / sample.pdf;

    float offsetSign = glm::dot(sample.direction, faceNormal) >= 0.0f ? 1.0f : -1.0f;

    pathSegment.ray.origin = intersect + offsetSign * 0.0001f * faceNormal;
    pathSegment.ray.direction = sample.direction;
    pathSegment.remainingBounces--;
}
