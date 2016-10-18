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

    std::vector<float> toData()
    {
        std::vector<float> vals(9);
        vals[0] = points[0].x;
        vals[1] = points[0].y;
        vals[2] = points[0].z;

        vals[3] = points[1].x;
        vals[4] = points[1].y;
        vals[5] = points[1].z;

        vals[6] = points[2].x;
        vals[7] = points[2].y;
        vals[8] = points[2].z;
        return vals;
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

    std::vector<float> toData()
    {
        std::vector<float> vals(6);
        vals[0] = start.x;
        vals[1] = start.y;
        vals[2] = start.z;

        vals[3] = dir.x;
        vals[4] = dir.y;
        vals[5] = dir.z;
        return vals;
    }
};

//void genPng()
//{
//    std::vector<char> d(imgSize*imgSize*3);
//    for(int i = 0; i < imgSize; i++)
//    {
//        for(int j = 0; j < imgSize; j++)
//        {
//            float minColl = 1000;
//            float minU;
//            float minV;
//            float pX = -(float)j/imgSize*2+1;
//            float pY = (float)i/imgSize*2-1;
//            glm::vec3 dir = -glm::normalize(glm::vec3(pX,pY,-1));
//            Ray r = Ray(glm::vec3(0,0,0),dir);
//
//            for(int k = 0; k < t.size(); k++)
//            {
//                float u,v;
//                float coll = testCollide(t[k],r,&u,&v);
//
//                if(coll > 0 && coll < minColl)
//                {
//                    minColl = coll;
//                    minU = u;
//                    minV = v;
//                }
//            }
//            if(minColl < 1000)
//            {
//                d[(i*imgSize + j)*3+0] = 255 * (minColl>0)*minU;
//                d[(i*imgSize + j)*3+1] = 255 * (minColl>0)*(1-minU-minV);
//                d[(i*imgSize + j)*3+2] = 255 * (minColl>0)*minV;
//            }
//            else
//            {
//                d[(i*imgSize + j)*3+0] = 0;
//                d[(i*imgSize + j)*3+1] = 0;
//                d[(i*imgSize + j)*3+2] = 0;
//            }
//        }
//    }
//
//    stbi_write_png("img.png", imgSize, imgSize, 3, d.data(), imgSize*3);
//}

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
    in.close();

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

    cl_kernel kernel = clCreateKernel(program, "sampleRays", &error);
    CheckError(error);

    cl_command_queue command_queue = clCreateCommandQueue(context, deviceIds[0], 0, &error);

    std::vector<Triangle> tris;
    tris.push_back(Triangle(glm::vec3(0,1,1),glm::vec3(1,-1,1),glm::vec3(-1,-1,1)));
    //tris.push_back(Triangle(glm::vec3(-1,1,2),glm::vec3(1,1,2),glm::vec3(-1,0.5,2)));
    std::vector<float> triData;
    for(int i = 0; i < tris.size(); i++)
    {
        std::vector<float> data = tris[i].toData();
        triData.insert(triData.end(), data.begin(), data.end());
    }

    int imgSize = 10;
    std::vector<Ray> rays;
    std::vector<float> rayData;
    for(int i = 0; i < imgSize; i++)
    {
        for(int j = 0; j < imgSize; j++)
        {
            float pX = (float)j/imgSize*10-5;
            float pY = (float)i/imgSize*10-5;
            glm::vec3 p = /*-glm::normalize(*/glm::vec3(pX,pY,0)/*)*/;
            Ray ray = Ray(p,glm::vec3(0,0,1));
            rays.push_back(ray);

            std::vector<float> data = ray.toData();
            rayData.insert(rayData.end(), data.begin(), data.end());
        }
    }

    cl_mem memTriangles = clCreateBuffer(context, CL_MEM_READ_WRITE, triData.size() * sizeof(float), NULL, &error);
    CheckError(error);
    cl_mem memRays = clCreateBuffer(context, CL_MEM_READ_WRITE, rayData.size() * sizeof(float), NULL, &error);
    CheckError(error);
    cl_mem memResults = clCreateBuffer(context, CL_MEM_READ_WRITE, imgSize*imgSize * sizeof(float), NULL, &error);
    CheckError(error);

    error = clEnqueueWriteBuffer(command_queue, memTriangles, CL_TRUE, 0, triData.size() * sizeof(float), triData.data(), 0, NULL, NULL);
    CheckError(error);
    error = clEnqueueWriteBuffer(command_queue, memRays, CL_TRUE, 0, rayData.size() * sizeof(float), rayData.data(), 0, NULL, NULL);
    CheckError(error);

    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memTriangles);
    CheckError(error);
    error = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memRays);
    CheckError(error);
    error = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&memResults);
    CheckError(error);

    error = clEnqueueTask(command_queue, kernel, 0, NULL,NULL);
    CheckError(error);
    std::size_t offsets[] = {0,0};
    std::size_t sizes[] = {triData.size(), rayData.size()};
    std::size_t step[] = {9,6};
    error = clEnqueueNDRangeKernel(command_queue, kernel, 2, offsets, sizes, step,
                                   0, NULL, NULL);
    CheckError(error);

    float results[imgSize*imgSize];
    error = clEnqueueReadBuffer(command_queue, memResults, CL_TRUE, 0,
                                imgSize*imgSize * sizeof(float), results, 0, NULL, NULL);

    std::vector<char> d(imgSize*imgSize*3);
    for(int i = 0; i < imgSize*imgSize; i++)
    {
        if(results[i] > 0)
        {
            d[i*3+0] = 255*results[i];
            d[i*3+1] = 255*results[i];
            d[i*3+2] = 255*results[i];
        }
        else
        {
            d[i*3+0] = 0;
            d[i*3+1] = 0;
            d[i*3+2] = 0;
        }
    }
    stbi_write_png("img.png", imgSize, imgSize, 3, d.data(), imgSize*3);

	clReleaseContext (context);

    return 0;
}
