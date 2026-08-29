#include "EFMM.cuh"
#include "interactions.cuh"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
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
        nvtxRangePushA("Data transfers to device");
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
        nvtxRangePushA("Asynchronous Transfer");
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
        for (unsigned int i = 0; i < N; i++) stage0[i] = h_b[i].V[0];
        CHECK(cudaMemcpyAsync(d_b.Vx, stage0, bytes, cudaMemcpyHostToDevice, stream0));

        CHECK(cudaStreamSynchronize(stream1));
        for (unsigned int i = 0; i < N; i++) stage1[i] = h_b[i].V[1];
        CHECK(cudaMemcpyAsync(d_b.Vy, stage1, bytes, cudaMemcpyHostToDevice, stream1));

        CHECK(cudaStreamSynchronize(stream0));
        for (unsigned int i = 0; i < N; i++) stage0[i] = h_b[i].V[2];
        CHECK(cudaMemcpyAsync(d_b.Vz, stage0, bytes, cudaMemcpyHostToDevice, stream0));

        
        CHECK(cudaStreamSynchronize(stream1));
        for (unsigned int i = 0; i < N; i++) b_stage1[i] = h_b[i].issink;
        CHECK(cudaMemcpyAsync(d_b.issink, b_stage1, bool_bytes, cudaMemcpyHostToDevice, stream1));

        CHECK(cudaStreamSynchronize(stream0));
        for (unsigned int i = 0; i < N; i++) b_stage0[i] = h_b[i].issource;
        CHECK(cudaMemcpyAsync(d_b.issource, b_stage0, bool_bytes, cudaMemcpyHostToDevice, stream0));



        nvtxRangePop();

        // Zero out the forces and potentials asynchronously on the device
        nvtxRangePushA("MemSet 0");
        CHECK(cudaMemsetAsync(d_b.Fx, 0, bytes, stream0));
        CHECK(cudaMemsetAsync(d_b.Fy, 0, bytes, stream0));
        CHECK(cudaMemsetAsync(d_b.Fz, 0, bytes, stream0));
        CHECK(cudaMemsetAsync(d_b.p,  0, bytes, stream0));
        CHECK(cudaMemsetAsync(d_b.acc_old,  0, bytes, stream0));
        CHECK(cudaMemsetAsync(d_b.timestep,  0, bytes, stream0));
        CHECK(cudaDeviceSynchronize());
        nvtxRangePop();
        CHECK(cudaFreeHost(stage0));
        CHECK(cudaFreeHost(stage1));
        CHECK(cudaStreamDestroy(stream0));
        CHECK(cudaStreamDestroy(stream1));
        nvtxRangePop();
    }

    void bodies_D2H(const Bodies& d_b, std::vector<exafmm::Body>& aos) {
        nvtxRangePushA("Data transfers to host");
        unsigned int N = aos.size();
        if (N == 0) return;

        size_t bytes = N * sizeof(exafmm::real_t);

        cudaStream_t stream0, stream1;
        CHECK(cudaStreamCreate(&stream0));
        CHECK(cudaStreamCreate(&stream1));

        exafmm::real_t *stage0 = nullptr, *stage1 = nullptr;
        CHECK(cudaMallocHost(&stage0, bytes));
        CHECK(cudaMallocHost(&stage1, bytes));

        nvtxRangePushA("Asynchronous Transfer");
        CHECK(cudaMemcpyAsync(stage0, d_b.Fx, bytes, cudaMemcpyDeviceToHost, stream0));
        CHECK(cudaMemcpyAsync(stage1, d_b.Fy, bytes, cudaMemcpyDeviceToHost, stream1));

        CHECK(cudaStreamSynchronize(stream0));
        for (unsigned int i = 0; i < N; i++) aos[i].F[0] += stage0[i];

        CHECK(cudaMemcpyAsync(stage0, d_b.Fz, bytes, cudaMemcpyDeviceToHost, stream0));

        CHECK(cudaStreamSynchronize(stream1));
        for (unsigned int i = 0; i < N; i++) aos[i].F[1] += stage1[i];

        CHECK(cudaMemcpyAsync(stage1, d_b.p, bytes, cudaMemcpyDeviceToHost, stream1));

        CHECK(cudaStreamSynchronize(stream0));
        for (unsigned int i = 0; i < N; i++) aos[i].F[2] += stage0[i];

        CHECK(cudaMemcpyAsync(stage0, d_b.acc_old, bytes, cudaMemcpyDeviceToHost, stream0));

        CHECK(cudaStreamSynchronize(stream1));
        for (unsigned int i = 0; i < N; i++) aos[i].p += stage1[i];

        CHECK(cudaMemcpyAsync(stage1, d_b.timestep, bytes, cudaMemcpyDeviceToHost, stream1));

        CHECK(cudaStreamSynchronize(stream0));
        for (unsigned int i = 0; i < N; i++) aos[i].acc_old += stage0[i];

        CHECK(cudaStreamSynchronize(stream1));
        for (unsigned int i = 0; i < N; i++) aos[i].timestep += stage1[i];
        nvtxRangePop();

        CHECK(cudaFreeHost(stage0));
        CHECK(cudaFreeHost(stage1));
        CHECK(cudaStreamDestroy(stream0));
        CHECK(cudaStreamDestroy(stream1));
        nvtxRangePop();
    }

    cufmm::Bodies device_bodies_alloc(unsigned int N)
    {
        cufmm::Bodies b;
        b.N = N;
        size_t bytes = N * sizeof(exafmm::real_t);
        size_t bool_bytes = N * sizeof(bool);

        CHECK(cudaMalloc((void**)&b.x, bytes));
        CHECK(cudaMalloc((void**)&b.y, bytes));
        CHECK(cudaMalloc((void**)&b.z, bytes));
        CHECK(cudaMalloc((void**)&b.q, bytes));
        CHECK(cudaMalloc((void**)&b.Fx, bytes));
        CHECK(cudaMalloc((void**)&b.Fy, bytes));
        CHECK(cudaMalloc((void**)&b.Fz, bytes));
        CHECK(cudaMalloc((void**)&b.p, bytes));
        CHECK(cudaMalloc((void**)&b.Vx, bytes));
        CHECK(cudaMalloc((void**)&b.Vy, bytes));
        CHECK(cudaMalloc((void**)&b.Vz, bytes));
        CHECK(cudaMalloc((void**)&b.acc_old, bytes));
        CHECK(cudaMalloc((void**)&b.timestep, bytes));
        CHECK(cudaMalloc((void**)&b.issink, bool_bytes));
        CHECK(cudaMalloc((void**)&b.issource, bool_bytes));

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
        CHECK(cudaFree(d_b.timestep));
        CHECK(cudaFree(d_b.issink));
        CHECK(cudaFree(d_b.issource));
        
        d_b.N = 0; 
    }

    template<Implementation impl>
    __global__ void cuP2P(
        cufmm::Bodies bodies,
        cufmm::DeviceInteractionView interactions,
        double dt_param
    )
    {
        int target_cell = blockIdx.x;
        int target_body_offset = interactions.target_body_offset[target_cell];
        int target_size = interactions.target_size[target_cell];

        int i = threadIdx.x;

        if(i < target_size)
        {
            exafmm::real_t ax = 0;
            exafmm::real_t ay = 0;
            exafmm::real_t az = 0;
            exafmm::real_t pot = 0;
            exafmm::real_t acc_old_i = 0;
            exafmm::real_t ts_accum = 0;
            exafmm::real_t dt_scale = dt_param * M_SQRT1_2;

            exafmm::real_t dX, dY, dZ;
            exafmm::real_t dVx, dVy, dVz;

            exafmm::real_t Xi = bodies.x[target_body_offset + i];
            exafmm::real_t Yi = bodies.y[target_body_offset + i];
            exafmm::real_t Zi = bodies.z[target_body_offset + i];

            exafmm::real_t Vxi, Vyi, Vzi, qi;
   
            if constexpr(impl == Implementation::standard)
            {
                Vxi = bodies.Vx[target_body_offset + i];
                Vyi = bodies.Vy[target_body_offset + i];
                Vzi = bodies.Vz[target_body_offset + i];
                qi = bodies.q[target_body_offset + i];
            }

            for(int inter = 0; inter < interactions.n_int_p2p[target_cell]; inter++)
            {
                exafmm::real_t timestep = 1e38;
                unsigned int base = interactions.offset[target_cell];
                unsigned int source_body_base = interactions.source_body_offset[base + inter];
                for(int j = source_body_base; j < source_body_base + interactions.source_size[base + inter] && j < bodies.N; j++)
                {
                    dX = bodies.x[j] - Xi;
                    dY = bodies.y[j] - Yi;
                    dZ = bodies.z[j] - Zi;                    

                    
                    exafmm::real_t R2 = dX*dX + dY*dY + dZ*dZ;

                    if (R2 > 0)
                    {
                        //math operations in the following blocks might look odd:
                        //optimizations were performed to reduce as much as possible MUFU instructions 
                        exafmm::real_t invR = rsqrt(R2);
                        
                        if constexpr(impl == Implementation::standard)
                        {
                            dVx = bodies.Vx[j] - Vxi;
                            dVy = bodies.Vy[j] - Vyi;
                            dVz = bodies.Vz[j] - Vzi;						
                        
                            exafmm::real_t v2 = dVx*dVx + dVy*dVy + dVz*dVz;                    
                            exafmm::real_t vdotdr2 = (dX * dVx + dY * dVy + dZ * dVz) * invR;

                            exafmm::real_t invR3 = invR*invR*invR;
                            exafmm::real_t tau = dt_scale * rsqrt( invR3 * (qi + bodies.q[j]));
                            exafmm::real_t half_dtau = 0.75 * tau * vdotdr2;
                            if (half_dtau > 0.5) half_dtau = 0.5;
                            exafmm::real_t t = 1.0 / (1.0 - half_dtau);
                            tau *= t;
                            if (tau < timestep) timestep = tau;

                            if (v2 > 0)
                            {
                                exafmm::real_t R = R2 * invR;
                                exafmm::real_t inv_v = rsqrt(v2);
                                tau = dt_param * R * inv_v;
                                half_dtau = 0.5 * tau * vdotdr2 * (1.0 + (qi + bodies.q[j]) * inv_v * inv_v * invR);
                                
                                if (half_dtau > 0.5) half_dtau = 0.5;
                                t = 1.0 / (1.0 - half_dtau);
                                tau *= t;
                                if (tau < timestep) timestep = tau;
                            }
                        }

                        exafmm::real_t invR2 = invR * invR;

                        if constexpr(impl == Implementation::low)
                        {
                            acc_old_i += bodies.q[j] * invR2;
                        }else
                        {
                            exafmm::real_t d_pot = bodies.q[j] * invR * bodies.issource[j];
                            pot += d_pot;
                            
                            exafmm::real_t mult = invR2 * d_pot;
                            dX *= mult;  dY *= mult;  dZ *= mult;
                            ax += dX;    ay += dY;    az += dZ; 
                        }                                                 
                    }
                }

                if constexpr(impl == Implementation::standard)
                {
                    timestep *= timestep;
                    timestep *= timestep;
                    timestep = (exafmm::real_t)1 / timestep;
                    ts_accum += timestep;
                }
            }

            if constexpr(impl == Implementation::low)
            {
                bodies.acc_old[target_body_offset + i] += acc_old_i;
            }else if(bodies.issink[target_body_offset + i])
            {
                bodies.p[target_body_offset + i] += pot;
                bodies.Fx[target_body_offset + i] += ax;
                bodies.Fy[target_body_offset + i] += ay;
                bodies.Fz[target_body_offset + i] += az;
                bodies.timestep[target_body_offset + i] += ts_accum;
            }
        }
    }

    template<Implementation impl>
    void cuP2P_launch(cufmm::Bodies d_bodies)
    {
        nvtxRangePushA("P2P launch");
        cufmm::DeviceInteractionView d_inter = cufmm::interaction_mgr.upload_to_device();
        CHECK(cudaDeviceSynchronize());
        nvtxRangePushA("Kernel and async");

        // 1 Block per active target cell
        int num_blocks = interaction_mgr.get_num_targets();
        int threads_per_block = exafmm::ncrit; 

        cuP2P<impl><<<num_blocks, threads_per_block>>>(
            d_bodies,
            d_inter,
            exafmm::dt_param
        );
        CHECK_KERNELCALL();
        CHECK(cudaDeviceSynchronize());
        nvtxRangePop();
        interaction_mgr.reset();
        nvtxRangePop();
    }

    //explicit template instantiation for g++ linker
    template void cuP2P_launch<Implementation::standard>(cufmm::Bodies bodies);
    template void cuP2P_launch<Implementation::simple>(cufmm::Bodies bodies);
    template void cuP2P_launch<Implementation::low>(cufmm::Bodies bodies);
}