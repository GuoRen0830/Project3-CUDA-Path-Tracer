#pragma once
#include "sceneStructs.h"
#include <thrust/random.h>

__host__ __device__ LightSample invalidLightSample();

__host__ __device__ float geometrySurfaceArea(const Geom& geom);

__host__ __device__ bool sampleCubeSurface(
	const Geom& geom,
	thrust::default_random_engine& rng,
	glm::vec3& position,
	glm::vec3& normal,
	float& areaPdf);

__host__ __device__ float sphereLightPdf(
	glm::vec3 referencePoint,
	const Geom& sphere);

__host__ __device__ LightSample sampleSphereLight(
	glm::vec3 referencePoint,
	const Geom& sphere,
	const Material& lightMaterial,
	int geomId,
	float selectionPdf,
	thrust::default_random_engine& rng);

__host__ __device__ bool sampleRectangleSurface(
	const Geom& geom,
	thrust::default_random_engine& rng,
	glm::vec3& position,
	glm::vec3& normal,
	float& areaPdf);

/**
 * Samples one emissive geometry and one point on its surface
 */
__host__ __device__ LightSample sampleLight(
	glm::vec3 referencePoint,
	const Geom* geoms,
	const Material* materials,
	const int* lightGeomIndices,
	int numLights,
	thrust::default_random_engine& rng);

/**
 * Estimates direct lighting with light and BSDF sampling
 */
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
	thrust::default_random_engine& rng);

/**
 * Tests whether a sampled light point is visible
 */
__host__ __device__ bool isLightVisible(
	glm::vec3 referencePoint,
	const LightSample& lightSample,
	const Geom* geoms,
	int numGeoms);

/**
 * Computes the light-sampling PDF for an emissive hit
 */
__host__ __device__ float lightPdfForHit(
	glm::vec3 referencePoint,
	glm::vec3 lightPoint,
	glm::vec3 lightNormal,
	const Geom& lightGeom,
	int numLights);

/**
 * Computes the balance weight using the power heuristic
 */
__host__ __device__ float powerHeuristic(
	float sampledPdf,
	float otherPdf);