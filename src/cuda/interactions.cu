#include "interactions.cuh"
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call)                                                              \
{                                                                                     \
    const cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                         \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                                           \
    }                                                                                 \
}

namespace cufmm {

    InteractionManager interaction_mgr;

    InteractionManager::InteractionManager()
        : n_cells(0), n_p2p_targets(0), total_p2p(0),
          h_n_int_p2p(nullptr), h_target_body_offset(nullptr),
          h_target_size(nullptr), h_offset(nullptr), h_saved_interactions(nullptr),
          h_source_body_offset(nullptr), h_source_size(nullptr),
          d_target_body_offset(nullptr), d_target_size(nullptr),
          d_offset(nullptr), d_n_int_p2p(nullptr),
          d_source_body_offset(nullptr), d_source_size(nullptr) {}

    InteractionManager::~InteractionManager() {
        reset();
    }

    void InteractionManager::free_host() {
        free(h_n_int_p2p);           h_n_int_p2p = nullptr;
        free(h_target_body_offset);  h_target_body_offset = nullptr;
        free(h_target_size);         h_target_size = nullptr;
        free(h_offset);              h_offset = nullptr;
        free(h_saved_interactions);   h_saved_interactions = nullptr;
        free(h_source_body_offset);  h_source_body_offset = nullptr;
        free(h_source_size);         h_source_size = nullptr;
    }

    void InteractionManager::free_device() {
        if (d_target_body_offset) { CUDA_CHECK(cudaFree(d_target_body_offset)); d_target_body_offset = nullptr; }
        if (d_target_size)        { CUDA_CHECK(cudaFree(d_target_size));        d_target_size = nullptr; }
        if (d_offset)             { CUDA_CHECK(cudaFree(d_offset));             d_offset = nullptr; }
        if (d_n_int_p2p)          { CUDA_CHECK(cudaFree(d_n_int_p2p));          d_n_int_p2p = nullptr; }
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

        h_n_int_p2p          = (unsigned int*) calloc(n_cells, sizeof(unsigned int));
        h_target_body_offset = (unsigned int*) calloc(n_cells, sizeof(unsigned int));
        h_target_size        = (unsigned int*) calloc(n_cells, sizeof(unsigned int));
        h_offset             = (unsigned int*) calloc(n_cells, sizeof(unsigned int));
        h_saved_interactions = (unsigned int*) calloc(n_cells, sizeof(unsigned int));
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
        unsigned int p2p_offset = 0;
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

        h_source_body_offset = (unsigned int*) calloc(total_p2p, sizeof(unsigned int));
        h_source_size        = (unsigned int*) calloc(total_p2p, sizeof(unsigned int));
    }

    DeviceInteractionView InteractionManager::upload_to_device(cudaStream_t stream) {
        free_device();

        unsigned int *pinned_target_offset, *pinned_target_size, *pinned_offset, *pinned_n_int;
        size_t target_bytes = n_p2p_targets * sizeof(unsigned int);

        CUDA_CHECK(cudaMallocHost(&pinned_target_offset, target_bytes));
        CUDA_CHECK(cudaMallocHost(&pinned_target_size,   target_bytes));
        CUDA_CHECK(cudaMallocHost(&pinned_offset,        target_bytes));
        CUDA_CHECK(cudaMallocHost(&pinned_n_int,         target_bytes));

        
        int t = 0;
        for (int c = 0; c < n_cells; c++) {
            if (h_n_int_p2p[c] > 0) {
                pinned_target_offset[t] = h_target_body_offset[c];
                pinned_target_size[t]   = h_target_size[c];
                pinned_offset[t]        = h_offset[c];
                pinned_n_int[t]         = h_n_int_p2p[c];
                t++;
            }
        }

        // 3. Allocate device memory
        CUDA_CHECK(cudaMalloc((void**)&d_target_body_offset, target_bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_target_size,        target_bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_offset,             target_bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_n_int_p2p,          target_bytes));

        size_t source_bytes = total_p2p * sizeof(unsigned int);
        CUDA_CHECK(cudaMalloc((void**)&d_source_body_offset, source_bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_source_size,        source_bytes));

        // 4. Asynchronous data transfers
        CUDA_CHECK(cudaMemcpyAsync(d_target_body_offset, pinned_target_offset, target_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_target_size,        pinned_target_size,   target_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_offset,             pinned_offset,        target_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_n_int_p2p,          pinned_n_int,         target_bytes, cudaMemcpyHostToDevice, stream));

        CUDA_CHECK(cudaMemcpyAsync(d_source_body_offset, h_source_body_offset, source_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_source_size,        h_source_size,        source_bytes, cudaMemcpyHostToDevice, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 5. Clean up temporary pinned buffers
        CUDA_CHECK(cudaFreeHost(pinned_target_offset));
        CUDA_CHECK(cudaFreeHost(pinned_target_size));
        CUDA_CHECK(cudaFreeHost(pinned_offset));
        CUDA_CHECK(cudaFreeHost(pinned_n_int));

        // 6. Return POD view to pass directly to kernel
        DeviceInteractionView view;
        view.n_p2p_targets       = n_p2p_targets;
        view.target_body_offset  = d_target_body_offset;
        view.target_size         = d_target_size;
        view.offset              = d_offset;
        view.n_int_p2p           = d_n_int_p2p;
        view.source_body_offset  = d_source_body_offset;
        view.source_size         = d_source_size;

        return view;
    }
}