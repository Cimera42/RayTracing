
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
    return a.x*b.x + a.y*b.y + a.y*b.y;
}

vec3 crossVec(vec3 a, vec3 b)
{
    vec3 returnMe;
    returnMe.x =   a.y*b.z - a.z*b.y;
    returnMe.y = -(a.x*b.z - a.z*b.x);
    returnMe.z =   a.x*b.y - a.y*b.x;
    return returnMe;
}

#define EPSILON 0.000001
float testCollide(Triangle inT, Ray inR)
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
        return t;
    }
    return 0;
}

__kernel void addTogether(__global const float* A, __global const int* B, __global int* C)
{
    C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];
}
