#include "intersections.h"
#include "utilities.h"
#include <cfloat>
#include <cmath>
#include <glm/geometric.hpp>

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n(0.0f);
    glm::vec3 tmax_n(0.0f);
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n(0.0f);
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float rectangleIntersectionTest(
    const Geom& rectangle,
    Ray ray,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    glm::vec3 localOrigin = multiplyMV(
        rectangle.inverseTransform,
        glm::vec4(ray.origin, 1.0f));

    glm::vec3 localDirection = multiplyMV(
        rectangle.inverseTransform,
        glm::vec4(ray.direction, 0.0f));

    const glm::vec3 localNormal(0.0f, 0.0f, 1.0f);

    float denominator = glm::dot(localNormal, localDirection);
    if (denominator >= -EPSILON)
    {
        return -1.0f;
    }

    float t = -localOrigin.z / localDirection.z;
    if (t <= 0.0f)
    {
        return -1.0f;
    }

    glm::vec3 localPoint = localOrigin + t * localDirection;

    if (fabsf(localPoint.x) > 0.5f || fabsf(localPoint.y) > 0.5f)
    {
        return -1.0f;
    }

    intersectionPoint = multiplyMV(
        rectangle.transform,
        glm::vec4(localPoint, 1.0f));

    normal = glm::normalize(multiplyMV(
        rectangle.invTranspose,
        glm::vec4(localNormal, 0.0f)));

    outside = true;

    return glm::length(intersectionPoint - ray.origin);
}

__host__ __device__ bool triangleIntersectionTest(
    const Ray& ray,
    const glm::vec3& p0,
    const glm::vec3& p1,
    const glm::vec3& p2,
    float tMin,
    float tMax,
    float& t,
    float& u,
    float& v)
{
    const glm::vec3 edge1 = p1 - p0;
    const glm::vec3 edge2 = p2 - p0;

    const glm::vec3 pvec = glm::cross(ray.direction, edge2);
    const float det = glm::dot(edge1, pvec);

    constexpr float determinantEpsilon = 1e-8f;

    if (fabsf(det) <= determinantEpsilon)
    {
        return false;
    }

    const float invDet = 1.0f / det;

    // Compute the weight of p1
    const glm::vec3 tvec = ray.origin - p0;
    const float candidateU = glm::dot(tvec, pvec) * invDet;
    if (candidateU < 0.0f || candidateU > 1.0f)
    {
        return false;
    }

    // Compute the weight of p2
    const glm::vec3 qvec = glm::cross(tvec, edge1);
    const float candidateV = glm::dot(ray.direction, qvec) * invDet;
    if (candidateV < 0.0f || candidateU + candidateV > 1.0f)
    {
        return false;
    }

    // Compute the ray parameter
    const float candidateT = glm::dot(edge2, qvec) * invDet;
    if (!(candidateT > tMin && candidateT < tMax))
    {
        return false;
    }

    t = candidateT;
    u = candidateU;
    v = candidateV;

    return true;
}

__host__ __device__ float sceneIntersectionTest(
    const Geom* geoms,
    int geomsSize,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    int& geomId,
    bool& outside)
{
    float closestT = FLT_MAX;
    geomId = -1;
    outside = true;

    for (int i = 0; i < geomsSize; ++i)
    {
        Geom geom = geoms[i];

        float t = -1.0f;
        glm::vec3 testPoint;
        glm::vec3 testNormal;
        bool testOutside = true;

        switch (geom.type)
        {
        case CUBE:
            t = boxIntersectionTest(geom, r, testPoint, testNormal, testOutside);
            break;

        case SPHERE:
            t = sphereIntersectionTest(geom, r, testPoint, testNormal, testOutside);
            break;

        case RECTANGLE:
            t = rectangleIntersectionTest(geom, r, testPoint, testNormal, testOutside);
            break;

        default:
            break;
        }

        if (t > 0.0001f && t < closestT)
        {
            closestT = t;
            geomId = i;
            intersectionPoint = testPoint;
            normal = testNormal;
            outside = testOutside;
        }
    }

    return geomId >= 0 ? closestT : -1.0f;
}