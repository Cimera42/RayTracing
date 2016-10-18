#include <iostream>
#include <fstream>
#include <string>
#include <glm/glm.hpp>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <utility>
#include <CL/opencl.h>

#include <sstream>
std::string Convert (float number){
    std::ostringstream buff;
    buff<<number;
    return buff.str();
}

struct Triangle
{
    glm::vec3 points[3];

    Triangle(glm::vec3 in1, glm::vec3 in2, glm::vec3 in3)
    {
        points[0] = in1;
        points[1] = in2;
        points[2] = in3;
    }

    std::string toString()
    {
        std::string s = "Triangle:\n";
        for(int i = 0; i < 3; i++)
        {
            s += "(" + Convert(points[i].x) + ", " + Convert(points[i].y) + ", " + Convert(points[i].z) + ")\n";
        }
        return s;
    }
};

struct Ray
{
    glm::vec3 start;
    glm::vec3 dir;

    Ray(glm::vec3 inStart, glm::vec3 inDir)
    {
        start = inStart;
        dir = inDir;
    }

    std::string toString()
    {
        std::string s = "Ray:\n";
        s += "(" + Convert(start.x) + ", " + Convert(start.y) + ", " + Convert(start.z) + ")\n";
        s += "<" + Convert(dir.x) + ", " + Convert(dir.y) + ", " + Convert(dir.z) + ">\n";
        return s;
    }
};

//Möller–Trumbore intersection algorithm (wikipedia)
#define EPSILON 0.000001
float testCollide(Triangle inT, Ray inR, float* outU, float* outV)
{
    glm::vec3 edge1, edge2;
    glm::vec3 P, Q, T;
    float det, inv_det, u, v;
    float t;

    edge1 = inT.points[1] - inT.points[0];
    edge2 = inT.points[2] - inT.points[0];

    P = glm::cross(inR.dir, edge2);
    det = glm::dot(edge1, P);
    if(det > -EPSILON && det < EPSILON) return 0;
    inv_det = 1.0f / det;

    T = inR.start - inT.points[0];
    u = glm::dot(T,P) * inv_det;
    if(u < 0.0f || u > 1.0f) return 0;

    Q = glm::cross(T, edge1);
    v = glm::dot(inR.dir, Q) * inv_det;
    if(v < 0.0f || u + v > 1.0f) return 0;

    t = glm::dot(edge2, Q) * inv_det;

    if(t > EPSILON)
    {
        *outU = u;
        *outV = v;
        return t;
    }
    return 0;
}

void genPng()
{
    std::vector<Triangle> t;
    t.push_back(Triangle(glm::vec3(0,1,2),glm::vec3(1,-1,1),glm::vec3(-1,-1,1)));
    t.push_back(Triangle(glm::vec3(-1,1,2),glm::vec3(1,1,2),glm::vec3(-1 ,0.5,2)));
    std::cout << t[0].toString() << std::endl;

    int imgSize = 1000;
    std::vector<char> d(imgSize*imgSize*3);
    for(int i = 0; i < imgSize; i++)
    {
        for(int j = 0; j < imgSize; j++)
        {
            float minColl = 1000;
            float minU;
            float minV;
            float pX = -(float)j/imgSize*2+1;
            float pY = (float)i/imgSize*2-1;
            glm::vec3 dir = -glm::normalize(glm::vec3(pX,pY,-1));
            Ray r = Ray(glm::vec3(0,0,0),dir);

            for(int k = 0; k < t.size(); k++)
            {
                float u,v;
                float coll = testCollide(t[k],r,&u,&v);

                if(coll > 0 && coll < minColl)
                {
                    minColl = coll;
                    minU = u;
                    minV = v;
                }
            }
            if(minColl < 1000)
            {
                d[(i*imgSize + j)*3+0] = 255 * (minColl>0)*minU;
                d[(i*imgSize + j)*3+1] = 255 * (minColl>0)*(1-minU-minV);
                d[(i*imgSize + j)*3+2] = 255 * (minColl>0)*minV;
            }
            else
            {
                d[(i*imgSize + j)*3+0] = 0;
                d[(i*imgSize + j)*3+1] = 0;
                d[(i*imgSize + j)*3+2] = 0;
            }
        }
    }

    stbi_write_png("img.png", imgSize, imgSize, 3, d.data(), imgSize*3);
}

void CheckError (cl_int error)
{
	if (error != CL_SUCCESS) {
		std::cerr << "OpenCL call failed with error " << error << std::endl;
	}
}

int main()
{
    std::ifstream in ("./arrayAdd.cl");
	std::string src (
		(std::istreambuf_iterator<char> (in)),
		std::istreambuf_iterator<char> ());

    cl_uint platformIdCount = 0;
    clGetPlatformIDs(0, NULL, &platformIdCount);

    std::vector<cl_platform_id> platformIds(platformIdCount);
    clGetPlatformIDs(platformIdCount, platformIds.data(), NULL);

    cl_uint deviceIdCount = 0;
    clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceIdCount);

    std::vector<cl_device_id> deviceIds(deviceIdCount);
    clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(), NULL);

    const cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(platformIds[0]),
        0, 0
    };

    cl_int error;
    cl_context context = clCreateContext(contextProperties, deviceIdCount,
                                         deviceIds.data(), NULL,
                                         NULL, &error);
    CheckError(error);

    size_t lengths [1] = { src.size () };
	const char* sources [1] = { src.data () };
	cl_program program = clCreateProgramWithSource(context, 1, sources, lengths, &error);
    CheckError(error);

    CheckError (clBuildProgram(program, deviceIdCount,
                deviceIds.data(), NULL, NULL, NULL));

    cl_kernel kernel = clCreateKernel(program, "addTogether", &error);
    CheckError(error);

    cl_command_queue command_queue = clCreateCommandQueue(context, deviceIds[0], 0, &error);

    cl_mem memA = clCreateBuffer(context, CL_MEM_READ_WRITE, 10 * sizeof(int), NULL, &error);
    CheckError(error);
    cl_mem memB = clCreateBuffer(context, CL_MEM_READ_WRITE, 10 * sizeof(int), NULL, &error);
    CheckError(error);
    cl_mem memC = clCreateBuffer(context, CL_MEM_READ_WRITE, 10 * sizeof(int), NULL, &error);
    CheckError(error);

    int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};

    error = clEnqueueWriteBuffer(command_queue, memA, CL_TRUE, 0, sizeof(int)*10, A, 0, NULL, NULL);
    CheckError(error);
    error = clEnqueueWriteBuffer(command_queue, memB, CL_TRUE, 0, sizeof(int)*10, B, 0, NULL, NULL);
    CheckError(error);

    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memA);
    CheckError(error);
    error = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memB);
    CheckError(error);
    error = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&memC);
    CheckError(error);

    error = clEnqueueTask(command_queue, kernel, 0, NULL,NULL);
    CheckError(error);
    std::size_t offsets[] = {0};
    std::size_t sizes[] = {10};
    error = clEnqueueNDRangeKernel(command_queue, kernel, 1, offsets, sizes, NULL,
                                   0, NULL, NULL);
    CheckError(error);

    int C[10];
    error = clEnqueueReadBuffer(command_queue, memC, CL_TRUE, 0,
                                10 * sizeof(int), C, 0, NULL, NULL);

    for(int i = 0; i < 10; i++)
    {
        std::cout << C[i] << std::endl;
    }

    genPng();

    return 0;
}
