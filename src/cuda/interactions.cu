#include "interactions.cuh"
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <cstring>

#define CUDA_CHECK(call)                                                              \
{                                                                                     \
    const cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                         \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                                           \
    }                                                                                 \
}

#define ALPHA 1.35

namespace cufmm {

    InteractionManager interaction_mgr;

InteractionManager::InteractionManager()
        : n_cells(0), n_p2p_targets(0), total_p2p(0),
          h_n_int_p2p(nullptr), h_target_body_offset(nullptr),
          h_target_size(nullptr), h_offset(nullptr), h_saved_interactions(nullptr),
          h_source_body_offset(nullptr), h_source_size(nullptr),
          d_tasks_heavy(nullptr), d_tasks_light(nullptr), d_source_body_offset(nullptr), d_source_size(nullptr) {}

    InteractionManager::~InteractionManager() {
        reset();
    }

    void InteractionManager::free_host() {
        if (h_n_int_p2p)          { free(h_n_int_p2p);          h_n_int_p2p = nullptr; }
        if (h_target_body_offset) { free(h_target_body_offset); h_target_body_offset = nullptr; }
        if (h_target_size)        { free(h_target_size);        h_target_size = nullptr; }
        if (h_offset)             { free(h_offset);             h_offset = nullptr; }
        if (h_saved_interactions) { free(h_saved_interactions); h_saved_interactions = nullptr; }
        if (h_source_body_offset) { free(h_source_body_offset); h_source_body_offset = nullptr; }
        if (h_source_size)        { free(h_source_size);        h_source_size = nullptr; }
        h_tasks_heavy.clear();
        h_tasks_light.clear();
    }

    void InteractionManager::free_device() {
        if (d_tasks_heavy)        { CUDA_CHECK(cudaFree(d_tasks_heavy));        d_tasks_heavy = nullptr; }
        if (d_tasks_light)        { CUDA_CHECK(cudaFree(d_tasks_light));        d_tasks_light = nullptr; }
        if (d_source_body_offset) { CUDA_CHECK(cudaFree(d_source_body_offset)); d_source_body_offset = nullptr; }
        if (d_source_size)        { CUDA_CHECK(cudaFree(d_source_size));        d_source_size = nullptr; }
    }

    void InteractionManager::reset() {
        free_host();
        free_device();
        n_cells = 0;
        n_p2p_targets = 0;
        total_p2p = 0;
    }

    void InteractionManager::init(int ncell) {
        reset();
        n_cells = ncell;
        if (n_cells <= 0) return;

        int device_id = 0;
        cudaGetDevice(&device_id);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device_id);

        num_sm =  props.multiProcessorCount;
        h_n_int_p2p          = (int*) calloc(n_cells, sizeof(int));
        h_target_body_offset = (int*) calloc(n_cells, sizeof(int));
        h_target_size        = (int*) calloc(n_cells, sizeof(int));
        h_offset             = (int*) calloc(n_cells, sizeof(int));
        h_saved_interactions = (int*) calloc(n_cells, sizeof(int));
    }

    
    // If we are doing horizontal_traversing for the first time, we add to the count
    // of interactions. We will then allocate the memory to save these interactions and
    // traverse the tree a second time, and we will save the actual data needed for the interactions
    void InteractionManager::add_interaction(const exafmm::Cell& Ci, const exafmm::Cell& Cj, interaction type, bool exploring) {
        if (type != P2P) return;

        if (exploring) {
            // __sync_fetch_and_add atomically increments the value and returns the OLD value
            //(remember that this might get called during OpenMP multithreading)
            __sync_fetch_and_add(&h_n_int_p2p[Ci.index], 1);

            h_target_body_offset[Ci.index] = Ci.BODY - &exafmm::bodies[0];
            h_target_size[Ci.index]        = Ci.NBODY;
        } else {
            int current_saved = __sync_fetch_and_add(&h_saved_interactions[Ci.index], 1);
            int source_idx    = h_offset[Ci.index] + current_saved;

            h_source_body_offset[source_idx] = Cj.BODY - &exafmm::bodies[0];
            h_source_size[source_idx]        = Cj.NBODY;
        }
    }

    void InteractionManager::finalize_exploration() {
         int p2p_offset = 0;
        n_p2p_targets = 0;

        // remove the unnecessary memory from arrays that will be passed to GPU
        for (int c = 0; c < n_cells; c++) {
            if (h_n_int_p2p[c] > 0) {
                h_offset[c] = p2p_offset;
                p2p_offset += h_n_int_p2p[c];
                n_p2p_targets++;
            }
        }
        total_p2p = p2p_offset;

        h_source_body_offset = ( int*) calloc(total_p2p, sizeof( int));
        h_source_size        = ( int*) calloc(total_p2p, sizeof( int));
    }

    int InteractionManager::target_workload(int total_pairs) 
    {
        // Target 4 full waves of execution across all SMs with 2 concurrent blocks per SM
        int target_blocks = num_sm * 2 * 40; 
        
        int derived_target = total_pairs / target_blocks;

        int W_MIN = 32768;   // 128 loop iterations per thread
        int W_MAX = 1048576;  // 512 loop iterations per thread

        if (derived_target < W_MIN) return W_MIN;
        if (derived_target > W_MAX) return W_MAX;
        return derived_target;
    }

    void InteractionManager::load_balance() {
        h_tasks_heavy.clear();
        h_tasks_light.clear();

        int total_pairs = 0;
        for (int c = 0; c < n_cells; c++) {
            if (h_n_int_p2p[c] == 0) continue;

            int tsb = 0;
            int base = h_offset[c];
            int count = h_n_int_p2p[c];
            for (int s = 0; s < count; s++) {
                tsb += h_source_size[base + s];
            }
            total_pairs += (int)h_target_size[c] * tsb;
        }

        int target_workload = this->target_workload(total_pairs);

        constexpr int S_THRESHOLD = 256; // Minimum source bodies for source-first binning
        //split if current workload is bigger
        int max_single_work = ALPHA * target_workload;
        //if workload way smaller than target, would ruin load balancing
        int micro_threshold = target_workload / 4;


        for (int c = 0; c < n_cells; c++) {
            if (h_n_int_p2p[c] == 0) continue;

            const int cell_size       = h_target_size[c];
            const int base_src_idx    = h_offset[c];
            const int total_src_cells = h_n_int_p2p[c];

            //total source bodies)
            int tsb = 0;
            for (int s = 0; s < total_src_cells; s++) {
                tsb += h_source_size[base_src_idx + s];
            }

            if (cell_size == 0 || tsb == 0) continue;

            int total_pairs = (int)cell_size * tsb;

            //workload is too small to armonize
            if (total_pairs < micro_threshold) {
                P2PTask task;
                task.target_body_offset = h_target_body_offset[c];
                task.target_chunk_size  = cell_size;
                task.source_list_offset = base_src_idx;
                task.num_source_cells   = total_src_cells;
                task.requires_atomic    = 0;
                h_tasks_light.push_back(task);
                continue;
            }

            // Workload is small enough for a single block
            if (total_pairs <= max_single_work) {
                P2PTask task;
                task.target_body_offset = h_target_body_offset[c];
                task.target_chunk_size  = cell_size;
                task.source_list_offset = base_src_idx;
                task.num_source_cells   = total_src_cells;
                task.requires_atomic    = 0;
                h_tasks_heavy.push_back(task);
                continue;
            }

            // Source Partitioning 
            // At the moment we prioritise this, expecting atomic writes, but avoiding excessive source loading
            // caused by target partitioning. We set a specific threshold to switch to that
            if (tsb >= S_THRESHOLD) {
                int num_src_bins = (total_pairs + target_workload / 2) / target_workload;
                if (num_src_bins < 2) num_src_bins = 2;

                int bin_target = total_pairs / num_src_bins;

                int current_src_start = base_src_idx;
                int current_src_cells = 0;
                int accumulated_src_bodies = 0;
                int bins_created = 0;

                for (int s = 0; s < total_src_cells; s++) {
                    int src_idx = base_src_idx + s;
                    accumulated_src_bodies += h_source_size[src_idx];
                    current_src_cells++;

                    int current_bin_pairs = (int)cell_size * accumulated_src_bodies;
                    bool is_last_src = (s == total_src_cells - 1);
                    bool reached_target = (current_bin_pairs >= bin_target);
                    bool remaining_bins_left = (bins_created < num_src_bins - 1);

                    if ((reached_target && remaining_bins_left) || is_last_src) {
                        P2PTask task;
                        task.target_body_offset = h_target_body_offset[c];
                        task.target_chunk_size  = cell_size;
                        task.source_list_offset = current_src_start;
                        task.num_source_cells   = current_src_cells;
                        task.requires_atomic    = 1; // Multiple blocks write to the same target body range

                        h_tasks_heavy.push_back(task);

                        bins_created++;
                        current_src_start += current_src_cells;
                        current_src_cells = 0;
                        accumulated_src_bodies = 0;
                    }
                }
            }
            // Target partitioning
            else {
                int chunk_size = 32;
                while (chunk_size <= exafmm::ncrit && (chunk_size * 2 * tsb) <= max_single_work) {
                    chunk_size *= 2;
                }

                int num_target_chunks = (cell_size + chunk_size - 1) / chunk_size;
                int current_target_offset = h_target_body_offset[c];
                int remaining_targets = cell_size;

                for (int tc = 0; tc < num_target_chunks; tc++) {
                    int current_target_size = (remaining_targets + (num_target_chunks - tc) - 1) / (num_target_chunks - tc);
                    remaining_targets -= current_target_size;

                    P2PTask task;
                    task.target_body_offset = current_target_offset;
                    task.target_chunk_size  = current_target_size;
                    task.source_list_offset = base_src_idx;
                    task.num_source_cells   = total_src_cells;
                    task.requires_atomic    = 0; // Each chunk owns unique target indices

                    h_tasks_heavy.push_back(task);
                    current_target_offset += current_target_size;
                }
            }
        }

#ifdef DEBUG
        std::cout << "Heavy tasks:" << std::endl;
        load_balance_stats(target_workload, h_tasks_heavy);
        std::cout << "Micro tasks:" << std::endl;
        load_balance_stats(target_workload / 4, h_tasks_light);
#endif
    }    

    DualDeviceInteractionView InteractionManager::upload_to_device(cudaStream_t stream_heavy, cudaStream_t stream_light) 
    {
        free_device();

        int num_heavy = h_tasks_heavy.size();
        int num_light = h_tasks_light.size();

        size_t heavy_bytes  = num_heavy * sizeof(P2PTask);
        size_t light_bytes  = num_light * sizeof(P2PTask);
        size_t source_bytes = total_p2p * sizeof(int);

        if (num_heavy > 0) CUDA_CHECK(cudaMalloc((void**)&d_tasks_heavy, heavy_bytes));
        if (num_light > 0) CUDA_CHECK(cudaMalloc((void**)&d_tasks_light, light_bytes));
        if (total_p2p > 0) {
            CUDA_CHECK(cudaMalloc((void**)&d_source_body_offset, source_bytes));
            CUDA_CHECK(cudaMalloc((void**)&d_source_size, source_bytes));
        }

        if (num_heavy > 0) {
            CUDA_CHECK(cudaMemcpyAsync(d_tasks_heavy, h_tasks_heavy.data(), heavy_bytes, cudaMemcpyHostToDevice, stream_heavy));
        }
        if (num_light > 0) {
            CUDA_CHECK(cudaMemcpyAsync(d_tasks_light, h_tasks_light.data(), light_bytes, cudaMemcpyHostToDevice, stream_light));
        }
        if (total_p2p > 0) {
            CUDA_CHECK(cudaMemcpyAsync(d_source_body_offset, h_source_body_offset, source_bytes, cudaMemcpyHostToDevice, stream_heavy));
            CUDA_CHECK(cudaMemcpyAsync(d_source_size, h_source_size, source_bytes, cudaMemcpyHostToDevice, stream_heavy));
        }

        DualDeviceInteractionView dual_view;
        
        dual_view.heavy = {num_heavy, d_tasks_heavy, d_source_body_offset, d_source_size};
        dual_view.light = {num_light, d_tasks_light, d_source_body_offset, d_source_size};

        return dual_view;
    }

    void InteractionManager::load_balance_stats(int target_workload, const std::vector<P2PTask>& h_tasks) const {
        if (h_tasks.empty()) {
            std::cout << "\n[Load Balancer] No tasks generated to profile.\n";
            return;
        }

        // -------------------------------------------------------------
        // 1. Gather Per-Task Metrics
        // -------------------------------------------------------------
        size_t total_tasks = h_tasks.size();
        std::vector<int> task_work(total_tasks);
        std::vector<int> task_sources(total_tasks);
        
        int atomic_tasks = 0;
        int total_pairs_computed = 0;

        for (size_t i = 0; i < total_tasks; i++) {
            const P2PTask& t = h_tasks[i];
            
            int src_bodies_in_task = 0;
            for (int s = 0; s < t.num_source_cells; s++) {
                src_bodies_in_task += h_source_size[t.source_list_offset + s];
            }

            int work = (int)t.target_chunk_size * src_bodies_in_task;
            task_work[i] = work;
            task_sources[i] = src_bodies_in_task;
            total_pairs_computed += work;

            if (t.requires_atomic) {
                atomic_tasks++;
            }
        }

        // -------------------------------------------------------------
        // 2. Statistical Analysis (Min, Max, Mean, StdDev, Percentiles)
        // -------------------------------------------------------------
        std::vector<int> sorted_work = task_work;
        std::sort(sorted_work.begin(), sorted_work.end());

        int min_work = sorted_work.front();
        int max_work = sorted_work.back();
        int p25_work = sorted_work[total_tasks * 25 / 100];
        int p50_work = sorted_work[total_tasks * 50 / 100];
        int p75_work = sorted_work[total_tasks * 75 / 100];
        int p95_work = sorted_work[total_tasks * 95 / 100];
        int p99_work = sorted_work[total_tasks * 99 / 100];

        double mean_work = (double)total_pairs_computed / total_tasks;
        double variance = 0.0;
        for (auto w : task_work) {
            variance += ((double)w - mean_work) * ((double)w - mean_work);
        }
        double std_dev = std::sqrt(variance / total_tasks);

        // Hardware waves calculation (assuming 2 blocks concurrently per SM)
        double waves = (num_sm > 0) ? (double)total_tasks / (num_sm * 2.0) : 0.0;

        // -------------------------------------------------------------
        // 3. Workload Histogram (5 Bins relative to W_target)
        // -------------------------------------------------------------
        int bin_micro  = 0; // < 0.25 W_target
        int bin_light  = 0; // [0.25, 0.75) W_target
        int bin_target = 0; // [0.75, 1.35) W_target (Ideal Target Zone)
        int bin_heavy  = 0; // [1.35, 2.00) W_target
        int bin_outlier= 0; // >= 2.00 W_target

        for (auto w : task_work) {
            double ratio = (double)w / target_workload;
            if (ratio < 0.25)       bin_micro++;
            else if (ratio < 0.75)  bin_light++;
            else if (ratio <= 1.35) bin_target++;
            else if (ratio < 2.00)  bin_heavy++;
            else                    bin_outlier++;
        }

        // -------------------------------------------------------------
        // 4. Formatted Output
        // -------------------------------------------------------------
        std::cout << "\n===============================================================\n";
        std::cout << "                 P2P LOAD BALANCER PROFILE REPORT              \n";
        std::cout << "===============================================================\n";
        std::cout << std::left << std::setw(30) << "Target Workload (W_target):" 
                  << std::right << std::setw(15) << target_workload << " pairs/block\n";
        std::cout << std::left << std::setw(30) << "Total Pairwise Evaluations:" 
                  << std::right << std::setw(15) << total_pairs_computed << " pairs\n";
        std::cout << std::left << std::setw(30) << "Generated GPU Tasks (Blocks):" 
                  << std::right << std::setw(15) << total_tasks << "\n";
        std::cout << std::left << std::setw(30) << "Target Cells Processed:" 
                  << std::right << std::setw(15) << n_p2p_targets << " / " << n_cells << "\n";
        std::cout << std::left << std::setw(30) << "Hardware SM Count:" 
                  << std::right << std::setw(15) << num_sm << " SMs\n";
        std::cout << std::left << std::setw(30) << "GPU Waves (at 2 blocks/SM):" 
                  << std::right << std::setw(15) << std::fixed << std::setprecision(2) << waves << " waves\n";
        
        std::cout << "---------------------------------------------------------------\n";
        std::cout << " TASK ARITHMETIC WORK DISTRIBUTION (Pairs/Task)\n";
        std::cout << "---------------------------------------------------------------\n";
        std::cout << std::left << std::setw(15) << "Min Work:"   << std::right << std::setw(15) << min_work 
                  << "  (" << std::fixed << std::setprecision(2) << (double)min_work / target_workload << "x W_target)\n";
        std::cout << std::left << std::setw(15) << "25th Pct (P25):" << std::right << std::setw(15) << p25_work << "\n";
        std::cout << std::left << std::setw(15) << "Median   (P50):" << std::right << std::setw(15) << p50_work << "\n";
        std::cout << std::left << std::setw(15) << "75th Pct (P75):" << std::right << std::setw(15) << p75_work << "\n";
        std::cout << std::left << std::setw(15) << "95th Pct (P95):" << std::right << std::setw(15) << p95_work << "\n";
        std::cout << std::left << std::setw(15) << "99th Pct (P99):" << std::right << std::setw(15) << p99_work << "\n";
        std::cout << std::left << std::setw(15) << "Max Work:"   << std::right << std::setw(15) << max_work 
                  << "  (" << std::fixed << std::setprecision(2) << (double)max_work / target_workload << "x W_target)\n";
        std::cout << std::left << std::setw(15) << "Mean Work:"  << std::right << std::setw(15) << (int)mean_work << "\n";
        std::cout << std::left << std::setw(15) << "Std Dev:"    << std::right << std::setw(15) << (int)std_dev 
                  << "  (" << std::fixed << std::setprecision(1) << (std_dev / mean_work * 100.0) << "% CV)\n";

        std::cout << "---------------------------------------------------------------\n";
        std::cout << " GLOBAL ATOMIC WRITE-BACK OVERHEAD\n";
        std::cout << "---------------------------------------------------------------\n";
        double atomic_pct = (double)atomic_tasks / total_tasks * 100.0;
        std::cout << std::left << std::setw(30) << "Direct Write-Back Tasks:" 
                  << std::right << std::setw(10) << (total_tasks - atomic_tasks) 
                  << " (" << std::fixed << std::setprecision(1) << (100.0 - atomic_pct) << "% - Zero atomics)\n";
        std::cout << std::left << std::setw(30) << "Atomic Write-Back Tasks:" 
                  << std::right << std::setw(10) << atomic_tasks 
                  << " (" << std::fixed << std::setprecision(1) << atomic_pct << "%)\n";

        std::cout << "---------------------------------------------------------------\n";
        std::cout << " WORKLOAD HISTOGRAM\n";
        std::cout << "---------------------------------------------------------------\n";
        auto print_bar = [&](const char* label, int count) {
            double pct = (double)count / total_tasks * 100.0;
            int bar_len = (int)(pct / 2.0); // 50 chars = 100%
            std::string bar(bar_len, '#');
            std::cout << std::left << std::setw(20) << label << " | "
                      << std::right << std::setw(6) << count << " (" << std::setw(5) << std::fixed << std::setprecision(1) << pct << "%) | "
                      << bar << "\n";
        };

        print_bar("< 0.25x (Micro)",    bin_micro);
        print_bar("0.25-0.75x (Light)", bin_light);
        print_bar("0.75-1.35x (Target)",bin_target);
        print_bar("1.35-2.00x (Heavy)", bin_heavy);
        print_bar(">= 2.00x (Outlier)", bin_outlier);
        std::cout << "===============================================================\n\n";
    }
}