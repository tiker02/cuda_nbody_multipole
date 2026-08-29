#ifndef INTERACTIONS_CUH
#define INTERACTIONS_CUH

#include "exafmm.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace cufmm
{
    enum interaction { P2P, M2L };

    // Lightweight POD struct passed to CUDA kernels by value
    struct DeviceInteractionView {
        unsigned int n_p2p_targets;
        const unsigned int* __restrict__ target_body_offset;
        const unsigned int* __restrict__ target_size;
        const unsigned int* __restrict__ offset;
        const unsigned int* __restrict__ n_int_p2p;
        const unsigned int* __restrict__ source_body_offset;
        const unsigned int* __restrict__ source_size;
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
        DeviceInteractionView upload_to_device(cudaStream_t stream = 0);
        void free_device();
        void reset();

        unsigned int get_num_targets() const { return n_p2p_targets; }
        unsigned int get_total_p2p() const   { return total_p2p; }
        int get_num_cells() const            { return n_cells; }

    private:
        int n_cells;                                         // Total number of cells in the tree
        unsigned int n_p2p_targets;                          // Number of target cells involved in P2P interactions             
        unsigned int total_p2p;                              // Total number of P2P interactions

        // Host Exploration arrays (Indexed by cell ID)
        unsigned int* h_n_int_p2p;                          // Number of P2P interactions per cell
        unsigned int* h_target_body_offset;                 // Starting index of Ci's bodies in the global array
        unsigned int* h_target_size;                        // Number of bodies in Ci
        unsigned int* h_offset;                             // Where Ci's list starts in the flat source arrays
        unsigned int* h_saved_interactions;                 // Running tracker of how many Cj's Ci has written

        // Host Flat Source arrays (Size: total_p2p)
        unsigned int* h_source_body_offset;                 // Flat array of Cj body offsets
        unsigned int* h_source_size;                        // Flat array of Cj body counts

        // Device Memory Pointers
        unsigned int* d_target_body_offset;
        unsigned int* d_target_size;
        unsigned int* d_offset;
        unsigned int* d_n_int_p2p;
        unsigned int* d_source_body_offset;
        unsigned int* d_source_size;

        void free_host();
    };

    extern InteractionManager interaction_mgr;
}

#endif 