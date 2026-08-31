#ifndef INTERACTIONS_CUH
#define INTERACTIONS_CUH

#include "exafmm.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace cufmm
{
    enum interaction { P2P, M2L };

    constexpr int pw_inter_threshold = 32;

    // Single work package assigned to 1 thread block
    struct P2PTask {
        int target_body_offset;  // Global index of first target body in this chunk
        int target_chunk_size;   // Number of target bodies
        int source_list_offset;  // Starting index in flat source arrays
        int num_source_cells;    // Number of Cj cells in this source bin
        int requires_atomic;     // 1 if multiple blocks process this target slice, 0 otherwise
    };

    // Lightweight POD struct passed to CUDA kernels by value
    struct DeviceInteractionView {
        int num_tasks;
        const P2PTask* __restrict__ tasks;
        const int* __restrict__ source_body_offset;
        const int* __restrict__ source_size;
    };

    struct DualDeviceInteractionView {
        DeviceInteractionView heavy;
        DeviceInteractionView light;
    };

    class InteractionManager {
    public:
        InteractionManager();
        ~InteractionManager();

        InteractionManager(const InteractionManager&) = delete;
        InteractionManager& operator=(const InteractionManager&) = delete;

        // Lifecycle methods
        void init(int ncell);
        void finalize_exploration();
        void add_interaction(const exafmm::Cell& Ci, const exafmm::Cell& Cj, interaction type, bool exploring);
        
        // GPU Data Management
        DualDeviceInteractionView upload_to_device(cudaStream_t stream_heavy = 0, cudaStream_t stream_light = 0);
        void free_device();
        void reset();

        int get_num_targets() const { return n_p2p_targets; }
        int get_total_p2p() const   { return total_p2p; }
        int get_num_cells() const            { return n_cells; }
        void load_balance();


    private:
        int num_sm;
        int n_cells;                                         // Total number of cells in the tree
        int n_p2p_targets;                          // Number of target cells involved in P2P interactions             
        int total_p2p;                              // Total number of P2P interactions
        int total_p2p_pw;

        int* h_n_int_p2p;                          // Number of P2P interactions per cell
        int* h_target_body_offset;                 // Starting index of Ci's bodies in the global array
        int* h_target_size;                        // Number of bodies in Ci
        int* h_offset;                             // Where Ci's list starts in the flat source arrays
        int* h_saved_interactions;                 // Running tracker of how many Cj's Ci has written

        int* h_source_body_offset;                 // Flat array of Cj body offsets
        int* h_source_size;                        // Flat array of Cj body counts

        std::vector<P2PTask> h_tasks_heavy;
        std::vector<P2PTask> h_tasks_light;
        // Device Memory Pointers
        P2PTask* d_tasks_heavy;
        P2PTask* d_tasks_light;
        int* d_source_body_offset;
        int* d_source_size;

        void load_balance_stats(int target_workload, const std::vector<P2PTask>& h_tasks) const;
        int target_workload(int total_pairs);
        void free_host();
    };

    extern InteractionManager interaction_mgr;
}

#endif 