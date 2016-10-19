
typedef struct TypeVec3
{
    float x,y,z;
} vec3;

typedef struct TypeTriangle
{
    vec3 p1;
    vec3 p2;
    vec3 p3;
} Triangle;

typedef struct TypeRay
{
    vec3 start;
    vec3 dir;
} Ray;

vec3 subVec(vec3 a, vec3 b)
{
    vec3 returnMe;
    returnMe.x = a.x - b.x;
    returnMe.y = a.y - b.y;
    returnMe.z = a.z - b.z;
    return returnMe;
}

float dotVec(vec3 a, vec3 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

vec3 crossVec(vec3 a, vec3 b)
{
    vec3 returnMe;
    returnMe.x = a.y*b.z - a.z*b.y;
    returnMe.y = a.z*b.x - a.x*b.z;
    returnMe.z = a.x*b.y - a.y*b.x;
    return returnMe;
}

#define EPSILON 0.000001
float testCollide(Triangle inT, Ray inR, float *inU, float *inV)
{
    vec3 edge1, edge2;
    vec3 P, Q, T;
    float det, inv_det, u, v;
    float t;

    edge1 = subVec(inT.p2, inT.p1);
    edge2 = subVec(inT.p3, inT.p1);

    P = crossVec(inR.dir, edge2);
    det = dotVec(edge1, P);
    if(det > -EPSILON && det < EPSILON) return 0;
    inv_det = 1.0f / det;

    T = subVec(inR.start, inT.p1);
    u = dotVec(T,P) * inv_det;
    if(u < 0.0f || u > 1.0f) return 0;

    Q = crossVec(T, edge1);
    v = dotVec(inR.dir, Q) * inv_det;
    if(v < 0.0f || u + v > 1.0f) return 0;

    t = dotVec(edge2, Q) * inv_det;

    if(t > EPSILON)
    {
        *inU = u;
        *inV = v;
        return t;
    }
    return 0;
}

__kernel void sampleRays(__global const float* triangles, __global const float* rays, __global float* results)
{
    int tindex = get_local_size(0) * get_group_id(0);
    Triangle tri;
    tri.p1.x = triangles[tindex+0];
    tri.p1.y = triangles[tindex+1];
    tri.p1.z = triangles[tindex+2];

    tri.p2.x = triangles[tindex+3];
    tri.p2.y = triangles[tindex+4];
    tri.p2.z = triangles[tindex+5];

    tri.p3.x = triangles[tindex+6];
    tri.p3.y = triangles[tindex+7];
    tri.p3.z = triangles[tindex+8];
//    tri.p1.x = 0.0f;
//    tri.p1.y = 1.0f;
//    tri.p1.z = 1.0f;
//
//    tri.p2.x = 1.0f;
//    tri.p2.y = -1.0f;
//    tri.p2.z = 1.0f;
//
//    tri.p3.x = -1.0f;
//    tri.p3.y = -1.0f;
//    tri.p3.z = 1.0f;

    int rindex = get_local_size(1) * get_group_id(1);
    Ray ray;
    ray.start.x = rays[rindex+0];
    ray.start.y = rays[rindex+1];
    ray.start.z = rays[rindex+2];
    ray.dir.x = rays[rindex+3];
    ray.dir.y = rays[rindex+4];
    ray.dir.z = rays[rindex+5];
//    ray.start.x = 0.0f;
//    ray.start.y = 0.0f;
//    ray.start.z = 0.0f;
//    ray.dir.x = 0.0f;
//    ray.dir.y = 0.0f;
//    ray.dir.z = 1.0f;

    float u;
    float v;
    float coll = testCollide(tri,ray, &u, &v);
    int cindex = get_local_size(2) * get_group_id(1);
    if(coll != 0 && coll < results[cindex+0])
    {
        results[cindex+0] = coll;
        results[cindex+1] = u;
        results[cindex+2] = v;
    }
//    results[tindex+0] = triangles[tindex+0];
//    results[tindex+1] = triangles[tindex+1];
//    results[tindex+2] = triangles[tindex+2];
//    results[tindex+3] = triangles[tindex+3];
//    results[tindex+4] = triangles[tindex+4];
//    results[tindex+5] = triangles[tindex+5];
//    results[tindex+6] = triangles[tindex+6];
//    results[tindex+7] = triangles[tindex+7];
//    results[tindex+8] = triangles[tindex+8];

//    results[rindex+0] = rays[rindex+0];
//    results[rindex+1] = rays[rindex+1];
//    results[rindex+2] = rays[rindex+2];
//    results[rindex+3] = rays[rindex+3];
//    results[rindex+4] = rays[rindex+4];
//    results[rindex+5] = rays[rindex+5];
}
