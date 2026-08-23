#ifndef EXAFMM_CUH
#define GPU_KERNELS_CUH

#include "exafmm.h"
#pragma once

namespace cufmm
{    
    enum interaction { P2P, M2L };
    enum Implementation { standard, simple, low };

    typedef struct 
    {
        unsigned int N;
        exafmm::real_t* __restrict__ x;
        exafmm::real_t* __restrict__ y;
        exafmm::real_t* __restrict__ z;
        exafmm::real_t* __restrict__ q;
        exafmm::real_t* __restrict__ Fx;
        exafmm::real_t* __restrict__ Fy;
        exafmm::real_t* __restrict__ Fz;
        exafmm::real_t* __restrict__ p;
        exafmm::real_t* __restrict__ Vx;
        exafmm::real_t* __restrict__ Vy;
        exafmm::real_t* __restrict__ Vz;
        exafmm::real_t* __restrict__ acc_old;
        exafmm::real_t* __restrict__ timestep;
        bool*  __restrict__ issink;
        bool*  __restrict__ issource;
    } Bodies;

    struct Interactions {
        int n_cells;                            // Total number of cells in the tree
        
        // Arrays populated during Exploration pass
        unsigned int n_p2p_targets;             // Number of target cells involved in P2P interactions             
        unsigned int* n_int_p2p;                // Number of P2P interactions per cell
        unsigned int total_p2p;                 // Total number of P2P interactions
        unsigned int* target_body_offset;       // Starting index of Ci's bodies in the global array
        unsigned int* target_size;              // Number of bodies in Ci
        unsigned int* offset;                   // Where Ci's list starts in the flat source arrays
        
        // Arrays populated during Data Saving pass
        unsigned int* source_body_offset;       // Flat array of Cj body offsets
        unsigned int* source_size;              // Flat array of Cj body counts
        unsigned int* saved_interactions;       // Running tracker of how many Cj's Ci has written
    };

    extern Interactions interactions;

    void bodies_H2D(exafmm::Bodies& h_b, cufmm::Bodies& d_b);
    void bodies_D2H(const Bodies& d_b, std::vector<exafmm::Body>& aos);
    cufmm::Bodies device_bodies_alloc(unsigned int N);
    void device_bodies_free(cufmm::Bodies& d_b);

    void add_interaction(exafmm::Cell& Ci, exafmm::Cell& Cj, interaction type, bool exploring);
    void interactions_manage(bool explored, int ncell = 0);

    template<Implementation impl> void cuP2P_launch(cufmm::Bodies d_bodies);

}

#endif