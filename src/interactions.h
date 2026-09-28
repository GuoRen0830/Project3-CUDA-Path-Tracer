#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>

#include <thrust/random.h>

struct BSDFSample
{
    glm::vec3 direction;
    glm::vec3 f;
    float pdf;
    bool isSpecular;
    bool valid;
};

__host__ __device__ glm::vec3 faceForwardNormal(glm::vec3 normal, glm::vec3 wo);
__host__ __device__ float schlickFresnel(float cosTheta, float etaI, float etaT);
__host__ __device__ BSDFSample invalidBSDFSample();
__host__ __device__ glm::vec3 localToWorld(glm::vec3 localDirection, glm::vec3 normal);

__host__ __device__ glm::vec3 schlickFresnel(float cosTheta, glm::vec3 f0);
__host__ __device__ float trowbridgeReitzD(glm::vec3 normal, glm::vec3 halfVector, float roughness);
__host__ __device__ float trowbridgeReitzLambda(glm::vec3 normal, glm::vec3 direction, float roughness);
__host__ __device__ float trowbridgeReitzG(glm::vec3 normal, glm::vec3 wo, glm::vec3 wi, float roughness);
__host__ __device__ glm::vec3 sampleGGXHalfVector(glm::vec3 normal, float roughness, thrust::default_random_engine& rng);
__host__ __device__ glm::vec3 evaluateMicrofacetReflection(const Material& m, glm::vec3 normal, glm::vec3 wo, glm::vec3 wi);
__host__ __device__ float microfacetReflectionPdf(const Material& m, glm::vec3 normal, glm::vec3 wo, glm::vec3 wi);

/**
 * Computes a cosine-weighted random direction in a hemisphere.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, 
    thrust::default_random_engine& rng);

/**
 * Evaluates the BSDF for a pair of world-space directions.
*/
__host__ __device__ glm::vec3 evaluateBSDF(
    const Material& m,
    glm::vec3 normal,
    glm::vec3 wo,
    glm::vec3 wi);

/**
 * Returns the BSDF sampling PDF for a world-space direction.
*/
__host__ __device__ float bsdfPdf(
    const Material& m,
    glm::vec3 normal,
    glm::vec3 wo,
    glm::vec3 wi);

/**
 * Samples the material BSDF
*/
__host__ __device__ BSDFSample sampleBSDF(
    const Material& m,
    glm::vec3 normal,
    bool outside,
    glm::vec3 wo,
    thrust::default_random_engine& rng);

/**
 * Samples the BSDF and updates the path in place
 */
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool outside,
    const Material& m,
    thrust::default_random_engine& rng);
