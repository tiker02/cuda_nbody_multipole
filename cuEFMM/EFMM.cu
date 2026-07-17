#include "EFMM.cuh"
#include <cuda_runtime.h>
#include <cstdio>

#define CHECK(call)                                                                 \
{                                                                                 \
    const cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                       \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
}

#define CHECK_KERNELCALL()                                                          \
{                                                                                 \
    const cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess) {                                                       \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
}

namespace cufmm{

    void bodies_H2D(exafmm::Bodies& h_b, cufmm::Bodies& d_b)
    {
        unsigned int N = h_b.size();
        size_t bytes = N * sizeof(exafmm::real_t);
        size_t bool_bytes = N * sizeof(bool);

        cudaStream_t stream0, stream1;
        CHECK(cudaStreamCreate(&stream0));
        CHECK(cudaStreamCreate(&stream1));

        //pinned memory staging buffers
        exafmm::real_t *stage0, *stage1;
        CHECK(cudaMallocHost(&stage0, bytes));
        CHECK(cudaMallocHost(&stage1, bytes));

        // Helper pointers for casting the buffers when we pack booleans
        bool* b_stage0 = (bool*)stage0;
        bool* b_stage1 = (bool*)stage1;

        // ==========================================
        // ASYNCHRONOUS PIPELINE
        // ==========================================
        for (unsigned int i = 0; i < N; i++) stage0[i] = h_b[i].X[0];
        CHECK(cudaMemcpyAsync(d_b.x, stage0, bytes, cudaMemcpyHostToDevice, stream0));

        for (unsigned int i = 0; i < N; i++) stage1[i] = h_b[i].X[1];
        CHECK(cudaMemcpyAsync(d_b.y, stage1, bytes, cudaMemcpyHostToDevice, stream1));


        CHECK(cudaStreamSynchronize(stream0));
        for (unsigned int i = 0; i < N; i++) stage0[i] = h_b[i].X[2];
        CHECK(cudaMemcpyAsync(d_b.z, stage0, bytes, cudaMemcpyHostToDevice, stream0));

        CHECK(cudaStreamSynchronize(stream1));
        for (unsigned int i = 0; i < N; i++) stage1[i] = h_b[i].q;
        CHECK(cudaMemcpyAsync(d_b.q, stage1, bytes, cudaMemcpyHostToDevice, stream1));


        CHECK(cudaStreamSynchronize(stream0));
        for (unsigned int i = 0; i < N; i++) b_stage0[i] = h_b[i].issource;
        CHECK(cudaMemcpyAsync(d_b.issource, b_stage0, bool_bytes, cudaMemcpyHostToDevice, stream0));

        CHECK(cudaStreamSynchronize(stream1));
        for (unsigned int i = 0; i < N; i++) b_stage1[i] = h_b[i].issink;
        CHECK(cudaMemcpyAsync(d_b.issink, b_stage1, bool_bytes, cudaMemcpyHostToDevice, stream1));

        // Zero out the forces and potentials asynchronously on the device
        CHECK(cudaMemsetAsync(d_b.Fx, 0, bytes, stream0));
        CHECK(cudaMemsetAsync(d_b.Fy, 0, bytes, stream0));
        CHECK(cudaMemsetAsync(d_b.Fz, 0, bytes, stream0));
        CHECK(cudaMemsetAsync(d_b.p,  0, bytes, stream0));
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaFreeHost(stage0));
        CHECK(cudaFreeHost(stage1));
        CHECK(cudaStreamDestroy(stream0));
        CHECK(cudaStreamDestroy(stream1));
    }

    void bodies_D2H(const Bodies& d_b, std::vector<exafmm::Body>& aos) {
        unsigned int N = aos.size();
        if (N == 0) return;

        size_t bytes = N * sizeof(exafmm::real_t);

        cudaStream_t stream0, stream1;
        CHECK(cudaStreamCreate(&stream0));
        CHECK(cudaStreamCreate(&stream1));

        exafmm::real_t *stage0 = nullptr, *stage1 = nullptr;
        CHECK(cudaMallocHost(&stage0, bytes));
        CHECK(cudaMallocHost(&stage1, bytes));


        CHECK(cudaMemcpyAsync(stage0, d_b.Fx, bytes, cudaMemcpyDeviceToHost, stream0));
        CHECK(cudaMemcpyAsync(stage1, d_b.Fy, bytes, cudaMemcpyDeviceToHost, stream1));

        CHECK(cudaStreamSynchronize(stream0));
        for (unsigned int i = 0; i < N; i++) aos[i].F[0] = stage0[i];

        CHECK(cudaMemcpyAsync(stage0, d_b.Fz, bytes, cudaMemcpyDeviceToHost, stream0));

        CHECK(cudaStreamSynchronize(stream1));
        for (unsigned int i = 0; i < N; i++) aos[i].F[1] = stage1[i];

        CHECK(cudaMemcpyAsync(stage1, d_b.p, bytes, cudaMemcpyDeviceToHost, stream1));

        CHECK(cudaStreamSynchronize(stream0));
        for (unsigned int i = 0; i < N; i++) aos[i].F[2] = stage0[i];

        CHECK(cudaStreamSynchronize(stream1));
        for (unsigned int i = 0; i < N; i++) aos[i].p = stage1[i];

        CHECK(cudaFreeHost(stage0));
        CHECK(cudaFreeHost(stage1));
        CHECK(cudaStreamDestroy(stream0));
        CHECK(cudaStreamDestroy(stream1));
    }

    cufmm::Bodies device_bodies_alloc(unsigned int N)
    {
        cufmm::Bodies b;
        b.N = N;
        size_t bytes = N * sizeof(exafmm::real_t);
        size_t bool_bytes = N * sizeof(bool);

        CHECK(cudaMalloc(&b.x, bytes));
        CHECK(cudaMalloc(&b.y, bytes));
        CHECK(cudaMalloc(&b.z, bytes));
        CHECK(cudaMalloc(&b.q, bytes));
        CHECK(cudaMalloc(&b.Fx, bytes));
        CHECK(cudaMalloc(&b.Fy, bytes));
        CHECK(cudaMalloc(&b.Fz, bytes));
        CHECK(cudaMalloc(&b.p, bytes));
        CHECK(cudaMalloc(&b.Vx, bytes));
        CHECK(cudaMalloc(&b.Vy, bytes));
        CHECK(cudaMalloc(&b.Vz, bytes));
        CHECK(cudaMalloc(&b.acc_old, bytes));
        CHECK(cudaMalloc(&b.timestamp, bytes));
        CHECK(cudaMalloc(&b.issink, bool_bytes));
        CHECK(cudaMalloc(&b.issource, bool_bytes));

        return b;
    }

    void device_bodies_free(Bodies& d_b) {
        if (d_b.N == 0) return;

        CHECK(cudaFree(d_b.x));
        CHECK(cudaFree(d_b.y));
        CHECK(cudaFree(d_b.z));
        CHECK(cudaFree(d_b.q));
        CHECK(cudaFree(d_b.Fx));
        CHECK(cudaFree(d_b.Fy));
        CHECK(cudaFree(d_b.Fz));
        CHECK(cudaFree(d_b.p));
        CHECK(cudaFree(d_b.Vx));
        CHECK(cudaFree(d_b.Vy));
        CHECK(cudaFree(d_b.Vz));
        CHECK(cudaFree(d_b.acc_old));
        CHECK(cudaFree(d_b.timestamp));
        CHECK(cudaFree(d_b.issink));
        CHECK(cudaFree(d_b.issource));
        
        d_b.N = 0; 
    }
}