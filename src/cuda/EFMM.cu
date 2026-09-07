#include "EFMM.cuh"
#include "interactions.cuh"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <cuda/std/cmath>
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
        int N = h_b.size();
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
        for (int i = 0; i < N; i++) stage0[i] = h_b[i].X[0];
        CHECK(cudaMemcpyAsync(d_b.x, stage0, bytes, cudaMemcpyHostToDevice, stream0));

        for (int i = 0; i < N; i++) stage1[i] = h_b[i].X[1];
        CHECK(cudaMemcpyAsync(d_b.y, stage1, bytes, cudaMemcpyHostToDevice, stream1));


        CHECK(cudaStreamSynchronize(stream0));
        for (int i = 0; i < N; i++) stage0[i] = h_b[i].X[2];
        CHECK(cudaMemcpyAsync(d_b.z, stage0, bytes, cudaMemcpyHostToDevice, stream0));

        CHECK(cudaStreamSynchronize(stream1));
        for (int i = 0; i < N; i++) stage1[i] = h_b[i].q;
        CHECK(cudaMemcpyAsync(d_b.q, stage1, bytes, cudaMemcpyHostToDevice, stream1));

        CHECK(cudaStreamSynchronize(stream0));
        for (int i = 0; i < N; i++) stage0[i] = h_b[i].V[0];
        CHECK(cudaMemcpyAsync(d_b.Vx, stage0, bytes, cudaMemcpyHostToDevice, stream0));

        CHECK(cudaStreamSynchronize(stream1));
        for (int i = 0; i < N; i++) stage1[i] = h_b[i].V[1];
        CHECK(cudaMemcpyAsync(d_b.Vy, stage1, bytes, cudaMemcpyHostToDevice, stream1));

        CHECK(cudaStreamSynchronize(stream0));
        for (int i = 0; i < N; i++) stage0[i] = h_b[i].V[2];
        CHECK(cudaMemcpyAsync(d_b.Vz, stage0, bytes, cudaMemcpyHostToDevice, stream0));

        
        CHECK(cudaStreamSynchronize(stream1));
        for (int i = 0; i < N; i++) b_stage1[i] = h_b[i].issink;
        CHECK(cudaMemcpyAsync(d_b.issink, b_stage1, bool_bytes, cudaMemcpyHostToDevice, stream1));

        CHECK(cudaStreamSynchronize(stream0));
        for (int i = 0; i < N; i++) b_stage0[i] = h_b[i].issource;
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
        int N = aos.size();
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
        for (int i = 0; i < N; i++) aos[i].F[0] += stage0[i];

        CHECK(cudaMemcpyAsync(stage0, d_b.Fz, bytes, cudaMemcpyDeviceToHost, stream0));

        CHECK(cudaStreamSynchronize(stream1));
        for (int i = 0; i < N; i++) aos[i].F[1] += stage1[i];

        CHECK(cudaMemcpyAsync(stage1, d_b.p, bytes, cudaMemcpyDeviceToHost, stream1));

        CHECK(cudaStreamSynchronize(stream0));
        for (int i = 0; i < N; i++) aos[i].F[2] += stage0[i];

        CHECK(cudaMemcpyAsync(stage0, d_b.acc_old, bytes, cudaMemcpyDeviceToHost, stream0));

        CHECK(cudaStreamSynchronize(stream1));
        for (int i = 0; i < N; i++) aos[i].p += stage1[i];

        CHECK(cudaMemcpyAsync(stage1, d_b.timestep, bytes, cudaMemcpyDeviceToHost, stream1));

        CHECK(cudaStreamSynchronize(stream0));
        for (int i = 0; i < N; i++) aos[i].acc_old += stage0[i];

        CHECK(cudaStreamSynchronize(stream1));
        for (int i = 0; i < N; i++) aos[i].timestep += stage1[i];
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
        exafmm::real_t dt_param
    )
    {
        asm volatile (".pragma \"enable_smem_spilling\";");

        const int task_id = blockIdx.x;
        if (task_id >= interactions.num_tasks) return;

        exafmm::real_t dt_scale = dt_param * M_SQRT1_2;

        const P2PTask task = interactions.tasks[task_id];

        const int tid      = threadIdx.y * blockDim.x + threadIdx.x;
        const int lane_id = threadIdx.x; 
        const int warp_id = threadIdx.y;
        const int num_warps = blockDim.y;
        const int block_sz = blockDim.x * blockDim.y;
        const int total_targets = task.target_chunk_size;
        //WARNING: OPTIMIZED TO REQUIRE <= warpSize
        const int warp_targets = (total_targets + num_warps - 1) / num_warps;


        constexpr int TILE_SIZE = 256;
        constexpr int STANDARD_TILE = (impl == Implementation::standard)? TILE_SIZE : 1; // unfortunately the compiler does not allow 0
        __shared__ exafmm::real_t s_x[TILE_SIZE];
        __shared__ exafmm::real_t s_y[TILE_SIZE];
        __shared__ exafmm::real_t s_z[TILE_SIZE];
        __shared__ exafmm::real_t s_q[TILE_SIZE];
        __shared__ bool s_issrc[TILE_SIZE];
        __shared__ exafmm::real_t s_vx[STANDARD_TILE];
        __shared__ exafmm::real_t s_vy[STANDARD_TILE];
        __shared__ exafmm::real_t s_vz[STANDARD_TILE];

        // this implementation assigns a warp to each target body. Because of block size limits,
        // we are required to do "warp coarsening". With the introduction of warp shuffling optimization,
        // the first warp_targets threads of each warp will store a specific target's data (t_) for the whole execution...
        int local_target = warp_id + lane_id * num_warps;
        bool is_valid_target = (local_target < total_targets);
        int target_idx = is_valid_target ? (task.target_body_offset + local_target) : 0;

        exafmm::real_t t_ax = 0;
        exafmm::real_t t_ay = 0;
        exafmm::real_t t_az = 0;
        exafmm::real_t t_pot = 0;
        exafmm::real_t t_acc_old_i = 0;
        exafmm::real_t t_timestep = 1e38;
        exafmm::real_t t_ts_accum = 0;

        exafmm::real_t t_Xi, t_Yi, t_Zi;
        exafmm::real_t t_Vxi, t_Vyi, t_Vzi, t_qi;

        if(is_valid_target){
            t_Xi = bodies.x[target_idx];
            t_Yi = bodies.y[target_idx];
            t_Zi = bodies.z[target_idx];

            if constexpr(impl == Implementation::standard)
            {
                t_Vxi = bodies.Vx[target_idx];
                t_Vyi = bodies.Vy[target_idx];
                t_Vzi = bodies.Vz[target_idx];
                t_qi = bodies.q[target_idx];
            }
        }
        
        for(int cj = 0; cj < task.num_source_cells; cj++)
        {
            exafmm::real_t timestep;
            int base = task.source_list_offset;
            int source_body_base = interactions.source_body_offset[base + cj];
            int src_count = interactions.source_size[base + cj];
            
            //shmem tiling of source bodies
            //int warps_per_tile = TILE_SIZE / warpSize;
            for (int tile_base = 0; tile_base < src_count; tile_base += TILE_SIZE) {
                int cur_tile = min(TILE_SIZE, src_count - tile_base);
                __syncthreads();
                for (int l = tid; l < cur_tile; l += block_sz) {
                    int g_idx = source_body_base + tile_base + l;
                    s_x[l] = bodies.x[g_idx];
                    s_y[l] = bodies.y[g_idx];
                    s_z[l] = bodies.z[g_idx];
                    s_q[l] = bodies.q[g_idx];
                    s_issrc[l] = bodies.issource[g_idx];
                    if constexpr (impl == Implementation::standard) {
                        s_vx[l] = bodies.Vx[g_idx];
                        s_vy[l] = bodies.Vy[g_idx];
                        s_vz[l] = bodies.Vz[g_idx];
                    }
                }
                __syncthreads();

        
                //... each of these data will be broadcasted to the whole warp
                // when is that target's turn. This allows us (wrt previous impl.) 
                // to have a more persistent sharing of the source tile, 
                // inverting the loop hierarchy from targets -> source_cell (-> TILE) -> cell_bodies 
                // to source_cell (-> TILE) -> targets -> cell_bodies, but without the need to reload 
                // target data for each source_cell/tile ...
                #pragma unroll 16
                for (int wt = 0; wt < warp_targets; wt++)
                {

                    int this_target = warp_id + wt * num_warps;
                    if(this_target >= total_targets) continue;

                    exafmm::real_t w_ax = 0;
                    exafmm::real_t w_ay = 0;
                    exafmm::real_t w_az = 0;
                    exafmm::real_t w_pot = 0;
                    exafmm::real_t w_acc_old_i = 0;

                    exafmm::real_t w_Xi, w_Yi, w_Zi;
                    exafmm::real_t w_Vxi, w_Vyi, w_Vzi, w_qi;
                    exafmm::real_t dX, dY, dZ;
                    exafmm::real_t dVx, dVy, dVz;                
                    
                    w_Xi = __shfl_sync(0xFFFFFFFF, t_Xi, wt);
                    w_Yi = __shfl_sync(0xFFFFFFFF, t_Yi, wt);
                    w_Zi = __shfl_sync(0xFFFFFFFF, t_Zi, wt);


                    if constexpr(impl == Implementation::standard)
                    {
                        w_Vxi = __shfl_sync(0xFFFFFFFF, t_Vxi, wt);
                        w_Vyi = __shfl_sync(0xFFFFFFFF, t_Vyi, wt);
                        w_Vzi = __shfl_sync(0xFFFFFFFF, t_Vzi, wt);
                        w_qi = __shfl_sync(0xFFFFFFFF, t_qi, wt);
                        timestep = __shfl_sync(0xFFFFFFFF, t_timestep, wt);
                    }                       

                    #pragma unroll 8
                    for(int j = lane_id; j < cur_tile; j += blockDim.x)  //warp size
                    {
 

                        dX = s_x[j] - w_Xi;
                        dY = s_y[j] - w_Yi;
                        dZ = s_z[j] - w_Zi;

                        if constexpr(impl == Implementation::standard)
                        {
                            dVx = s_vx[j] - w_Vxi;
                            dVy = s_vy[j] - w_Vyi;
                            dVz = s_vz[j] - w_Vzi;						
                        }

                        exafmm::real_t R2 = dX*dX + dY*dY + dZ*dZ;

                      
                        //math operations in the following blocks might look odd:
                        //optimizations were performed to reduce as much as possible MUFU instructions 

                        //removed branch dependent from R2
                        exafmm::real_t safe_R2 = (R2 > (exafmm::real_t)0.0) ? R2 : (exafmm::real_t)1.0;
                        exafmm::real_t invR = rsqrt(safe_R2);
                        


                        exafmm::real_t invR2 = invR * invR;

                        exafmm::real_t mask = (R2 > (exafmm::real_t)0.0) ? (exafmm::real_t)1.0 : (exafmm::real_t)0.0;
                        if constexpr(impl == Implementation::low)
                        {
                            w_acc_old_i += s_q[j] * invR2 * mask;
                        }else
                        {
                            exafmm::real_t d_pot = s_q[j] * invR * s_issrc[j];
                            w_pot += d_pot * mask;
                            
                            exafmm::real_t mult = invR2 * d_pot * mask;
                            w_ax += dX * mult;    
                            w_ay += dY * mult;    
                            w_az += dZ * mult; 
                        }           
                        if constexpr(impl == Implementation::standard)
                        {                            
                            exafmm::real_t q_sum = w_qi + s_q[j];
                                                
                            exafmm::real_t vdotdr2 = (dX * dVx + dY * dVy + dZ * dVz) * invR;

                            exafmm::real_t invR3 = invR2*invR;
                            exafmm::real_t tau = dt_scale * rsqrt( invR3 * q_sum);
                            
                            exafmm::real_t v2 = dVx*dVx + dVy*dVy + dVz*dVz;  
                            
                            exafmm::real_t half_dtau = ((exafmm::real_t) 0.75) * tau * vdotdr2;
                            half_dtau = cuda::std::fmin(half_dtau, (exafmm::real_t) 0.5);
                            exafmm::real_t t = ((exafmm::real_t)1.0) / (((exafmm::real_t)1.0) - half_dtau);
                            tau *= t;
                            if (tau < timestep) timestep = tau;

                            if (v2 > 0)
                            {
                                exafmm::real_t R = R2 * invR;
                                exafmm::real_t inv_v = rsqrt(v2);
                                tau = dt_param * R * inv_v;
                                half_dtau = ((exafmm::real_t)0.5) * tau * vdotdr2 * (((exafmm::real_t)1.0) + q_sum * inv_v * inv_v * invR);
                                half_dtau = cuda::std::fmin(half_dtau, (exafmm::real_t) 0.5);
                                exafmm::real_t t = ((exafmm::real_t)1.0) / (((exafmm::real_t)1.0) - half_dtau);
                                tau *= t;
                                if (tau < timestep) timestep = tau;
                            }
                        }
                    }
                    //... of course with the inverted loop we need to save the results for each
                    // target at the end of it's loop. To avoid an exploding number of atomic writes, 
                    // we of course use once again warp shuffling, in this case implementing a sum reduction 
                    // (and min for timestep)

                    if constexpr (impl == Implementation::low) 
                    {
                        #pragma unroll
                        for (int offset = 16; offset > 0; offset /= 2) {
                            w_acc_old_i += __shfl_down_sync(0xFFFFFFFF, w_acc_old_i, offset);
                        }
                        w_acc_old_i = __shfl_sync(0xFFFFFFFF, w_acc_old_i, 0);
                        if(lane_id == wt) t_acc_old_i += w_acc_old_i;
                    }else
                    {
                        #pragma unroll
                        for (int offset = 16; offset > 0; offset /= 2) {
                            w_pot += __shfl_down_sync(0xFFFFFFFF, w_pot, offset);
                            w_ax += __shfl_down_sync(0xFFFFFFFF, w_ax, offset);
                            w_ay += __shfl_down_sync(0xFFFFFFFF, w_ay, offset);
                            w_az += __shfl_down_sync(0xFFFFFFFF, w_az, offset);
                        }
                        w_pot = __shfl_sync(0xFFFFFFFF, w_pot, 0);
                        w_ax = __shfl_sync(0xFFFFFFFF, w_ax, 0);
                        w_ay = __shfl_sync(0xFFFFFFFF, w_ay, 0);
                        w_az = __shfl_sync(0xFFFFFFFF, w_az, 0);

                        if(lane_id == wt){
                            t_pot += w_pot;
                            t_ax += w_ax;
                            t_ay += w_ay;
                            t_az += w_az;
                        }
                    }


                    //the timestep tracking is the most tricky one: it is source cell specific:
                    // the reduction is required at the end of each target loop , but the accumulation is 
                    // after the end of the loop on tiles
                    if constexpr(impl == Implementation::standard)
                    {
                        //reducing on timestep: in sequential execution, it should be the min across the
                        //interactions with the source
                        #pragma unroll
                        for (int offset = 16; offset > 0; offset /= 2) {
                            exafmm::real_t other_ts = __shfl_down_sync(0xFFFFFFFF, timestep, offset);
                            if (other_ts < timestep) timestep = other_ts;
                        }
                        exafmm::real_t reduced_ts = __shfl_sync(0xFFFFFFFF, timestep, 0);
                        if(lane_id == wt) t_timestep = reduced_ts;
                    }

                }
            }
            if constexpr(impl == Implementation::standard)
            {
                t_timestep *= t_timestep;
                t_timestep *= t_timestep;
                t_timestep = (exafmm::real_t)1 / t_timestep;
                t_ts_accum += t_timestep;
                // we need to reset for the next target cell;
                t_timestep = 1e38;
            
            }
        }

        if(is_valid_target)
        {
            if(task.requires_atomic){
                if constexpr (impl == Implementation::low) 
                {
                    atomicAdd(&bodies.acc_old[target_idx], t_acc_old_i);
                } 
                else if (bodies.issink[target_idx]) 
                {
                    atomicAdd(&bodies.p[target_idx], t_pot);
                    atomicAdd(&bodies.Fx[target_idx], t_ax);
                    atomicAdd(&bodies.Fy[target_idx], t_ay);
                    atomicAdd(&bodies.Fz[target_idx], t_az);
                    atomicAdd(&bodies.timestep[target_idx], t_ts_accum);
                }
            }else {
                if constexpr (impl == Implementation::low) {
                    bodies.acc_old[target_idx] += t_acc_old_i;
                } else if (bodies.issink[target_idx]) {
                    bodies.p[target_idx] += t_pot;
                    bodies.Fx[target_idx] += t_ax;
                    bodies.Fy[target_idx] += t_ay;
                    bodies.Fz[target_idx] += t_az;
                    bodies.timestep[target_idx] += t_ts_accum;
                }
            }
        }
    }

    template<Implementation impl>
    void cuP2P_launch(cufmm::Bodies d_bodies)
    {
        cudaStream_t stream_heavy, stream_light;
        CHECK(cudaStreamCreateWithFlags(&stream_heavy, cudaStreamNonBlocking));
        CHECK(cudaStreamCreateWithFlags(&stream_light, cudaStreamNonBlocking));

        // Event to synchronize shared source tables between streams (see upload_to_device function)
        cudaEvent_t event_sources_uploaded;
        CHECK(cudaEventCreateWithFlags(&event_sources_uploaded, cudaEventDisableTiming));

        // (asynchronous)
        nvtxRangePushA("H2D Task Uploads");
        cufmm::DualDeviceInteractionView d_inter = cufmm::interaction_mgr.upload_to_device(stream_heavy, stream_light);
        
        CHECK(cudaEventRecord(event_sources_uploaded, stream_heavy));
        CHECK(cudaStreamWaitEvent(stream_light, event_sources_uploaded, 0));
        nvtxRangePop();



        nvtxRangePushA("Concurrent Kernels");
        
        //warp granularity executions
        dim3 threads_heavy(32, 8);
        dim3 threads_light(32, 2);                  

        if (d_inter.heavy.num_tasks > 0) {
            cuP2P<impl><<<d_inter.heavy.num_tasks, threads_heavy, 0, stream_heavy>>>(
                d_bodies,
                d_inter.heavy,
                exafmm::dt_param
            );
        }

        if (d_inter.light.num_tasks > 0) {
            cuP2P<impl><<<d_inter.light.num_tasks, threads_light, 0, stream_light>>>(
                d_bodies,
                d_inter.light,
                exafmm::dt_param
            );
        }

        CHECK_KERNELCALL();

        CHECK(cudaStreamSynchronize(stream_heavy));
        CHECK(cudaStreamSynchronize(stream_light));
        nvtxRangePop();

        CHECK(cudaEventDestroy(event_sources_uploaded));
        CHECK(cudaStreamDestroy(stream_heavy));
        CHECK(cudaStreamDestroy(stream_light));

        interaction_mgr.reset();
        nvtxRangePop();
    }

    //explicit template instantiation for g++ linker
    template void cuP2P_launch<Implementation::standard>(cufmm::Bodies bodies);
    template void cuP2P_launch<Implementation::simple>(cufmm::Bodies bodies);
    template void cuP2P_launch<Implementation::low>(cufmm::Bodies bodies);
}