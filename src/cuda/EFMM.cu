#include "EFMM.cuh"
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

    Interactions interactions;

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

    // If we are doing horizontal_traversing for the first time, we add to the count
    // of interactions. We will then allocate the memory to save these interactions and
    // traverse the tree a second time, and we will save the actual data needed for the interactions
    void add_interaction(exafmm::Cell& Ci, exafmm::Cell& Cj, interaction type, bool exploring)
    {
        if(!exploring) 
        {
            if(type == P2P)
            {
// __sync_fetch_and_add atomically increments the value and returns the OLD value
                int current_saved = __sync_fetch_and_add(&interactions.saved_interactions[Ci.index], 1);
                
                int source_idx = interactions.offset[Ci.index] + current_saved;
                interactions.source_body_offset[source_idx] = Cj.BODY - &exafmm::bodies[0];
                interactions.source_size[source_idx] = Cj.NBODY;

                if (interactions.source_body_offset[source_idx] + interactions.source_size[source_idx] > exafmm::bodies.size()) {
                    printf("FATAL: Target cell %d pointer is outside main array! Offset: %ld, size: %ld\n", Ci.index, interactions.source_body_offset[source_idx], interactions.source_size[source_idx]);
                    exit(1);
                }
            }
        }
        else 
        {            
            if(type == P2P)
            {
                // Atomically increment the total P2P interaction count
                __sync_fetch_and_add(&interactions.n_int_p2p[Ci.index], 1);
                
                // Benign data race: Multiple threads might write this, but they write the exact same integer
                interactions.target_body_offset[Ci.index] =  Ci.BODY - &exafmm::bodies[0];
                interactions.target_size[Ci.index] = Ci.NBODY;

                if (interactions.target_body_offset[Ci.index] + interactions.target_size[Ci.index] > exafmm::bodies.size()) {
                    printf("FATAL: Source cell %d pointer is outside main array! Offset: %ld, size: %ld\n", Cj.index, interactions.target_body_offset[Ci.index], interactions.target_size[Ci.index]);
                    exit(1);
                }
            }
        }
    }

    void interactions_manage(bool explored, int ncell)
    {
        if(!explored && ncell > 0)
        {
            interactions.n_cells = ncell;
#ifdef DEBUG
            printf("Total number of cells: %ld\n", ncell);
#endif
            interactions.n_int_p2p = (unsigned int *) calloc(ncell, sizeof(unsigned int));
            interactions.target_body_offset = (unsigned int *) calloc(ncell, sizeof(unsigned int));
            interactions.target_size = (unsigned int *) calloc(ncell, sizeof(unsigned int));
            interactions.offset = (unsigned int *) calloc(ncell, sizeof(unsigned int));
        }
        else if(explored)
        {
            int total_int;
            int p2p_offset = 0;
            interactions.n_p2p_targets = 0;
            for(int c = 0; c < interactions.n_cells; c++)
            {
                if(interactions.n_int_p2p[c] > 0)
                {
                    interactions.offset[c] = p2p_offset;
                    p2p_offset += interactions.n_int_p2p[c];
                    interactions.n_p2p_targets++;
                } 
            }
            total_int = p2p_offset;
            interactions.total_p2p = total_int;
            interactions.source_body_offset = (unsigned int *) calloc(total_int, sizeof(unsigned int));
            interactions.source_size = (unsigned int *) calloc(total_int, sizeof(unsigned int));
            interactions.saved_interactions = (unsigned int *) calloc(interactions.n_cells, sizeof(unsigned int));

#ifdef DEBUG
            printf("Exploration completed: %d target cells and %d total interactions\n", interactions.n_p2p_targets, interactions.total_p2p );
#endif
        }
    }

    // remove the unnecessary memory from arrays that will be passed to GPU
    unsigned int* p2p_interactions_compress_to_device(unsigned int ** target_array)
    {
        unsigned int* new_target_array;
        CHECK(cudaMallocHost(&new_target_array, interactions.n_p2p_targets * sizeof(unsigned int)));
        int t = 0;
        for(int c = 0; c < interactions.n_cells; c++)
        {
            if(interactions.n_int_p2p[c] > 0)
            {
                new_target_array[t] = (*target_array)[c];
                t++;
            } 
        }
        free(*target_array);
        //we substitute the pointer to later free the pinned memory
        *target_array = new_target_array;
        unsigned int* d_target;
        CHECK(cudaMalloc(&d_target, interactions.n_p2p_targets * sizeof(unsigned int)));
        CHECK(cudaMemcpyAsync(d_target, new_target_array, interactions.n_p2p_targets * sizeof(unsigned int), cudaMemcpyHostToDevice));
        return d_target;
    }

    template<Implementation impl>
    __global__ void cuP2P(
        const unsigned int* __restrict__ offsets, 
        const unsigned int* __restrict__ n_int, 
        const unsigned int* __restrict__ target_body_offsets, 
        const unsigned int* __restrict__ target_sizes, 
        const unsigned int* __restrict__ source_body_offsets, 
        const unsigned int* __restrict__ source_sizes,
        cufmm::Bodies bodies,
        double dt_param
    )
    {
        int target_cell = blockIdx.x;
        int target_body_offset = target_body_offsets[target_cell];
        int target_size = target_sizes[target_cell];

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

            for(int inter = 0; inter < n_int[target_cell]; inter++)
            {
                exafmm::real_t timestep = 1e38;
                unsigned int base = offsets[target_cell];
                unsigned int source_body_base = source_body_offsets[base + inter];
                for(int j = source_body_base; j < source_body_base + source_sizes[base + inter] && j < bodies.N; j++)
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
        nvtxRangePushA("Interaction compress");
        unsigned int* d_target_body_offset = p2p_interactions_compress_to_device(&interactions.target_body_offset);
        unsigned int* d_offset             = p2p_interactions_compress_to_device(&interactions.offset);
        unsigned int* d_target_size        = p2p_interactions_compress_to_device(&interactions.target_size);
        unsigned int* d_n_int              = p2p_interactions_compress_to_device(&interactions.n_int_p2p);
        nvtxRangePop();
        unsigned int* d_source_body_offset;
        unsigned int* d_source_size;

        size_t source_bytes = interactions.total_p2p * sizeof(unsigned int);

        CHECK(cudaMalloc((void**)&d_source_body_offset, source_bytes));
        CHECK(cudaMalloc((void**)&d_source_size, source_bytes));

        CHECK(cudaMemcpy(d_source_body_offset, interactions.source_body_offset, source_bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_source_size, interactions.source_size, source_bytes, cudaMemcpyHostToDevice));

        CHECK(cudaDeviceSynchronize());
        nvtxRangePushA("Kernel and async");

        // 1 Block per active target cell
        int num_blocks = interactions.n_p2p_targets;
        int threads_per_block = exafmm::ncrit; 

#ifdef DEBUG
        printf("Kerenl launch: %d target cells\n", interactions.n_p2p_targets);
#endif 

        cuP2P<impl><<<num_blocks, threads_per_block>>>(
            d_offset, 
            d_n_int, 
            d_target_body_offset, 
            d_target_size, 
            d_source_body_offset, 
            d_source_size,
            d_bodies,
            exafmm::dt_param
        );
        CHECK_KERNELCALL();

        //we can free host memory while the kernel is running
        CHECK(cudaFreeHost(interactions.target_body_offset));
        CHECK(cudaFreeHost(interactions.target_size));
        CHECK(cudaFreeHost(interactions.offset));
        CHECK(cudaFreeHost(interactions.n_int_p2p));

        free(interactions.source_body_offset);
        free(interactions.source_size);
        free(interactions.saved_interactions);


        CHECK(cudaDeviceSynchronize());
        nvtxRangePop();

        CHECK(cudaFree(d_target_body_offset));
        CHECK(cudaFree(d_target_size));
        CHECK(cudaFree(d_offset));
        CHECK(cudaFree(d_n_int));

        CHECK(cudaFree(d_source_body_offset));
        CHECK(cudaFree(d_source_size));
        nvtxRangePop();
    }

    //explicit template instantiation for g++ linker
    template void cuP2P_launch<Implementation::standard>(cufmm::Bodies bodies);
    template void cuP2P_launch<Implementation::simple>(cufmm::Bodies bodies);
    template void cuP2P_launch<Implementation::low>(cufmm::Bodies bodies);
}