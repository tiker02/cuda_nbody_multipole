#ifndef EXAFMM_CUH
#define GPU_KERNELS_CUH

#include "exafmm.h"
#pragma once

namespace cufmm
{
    typedef struct 
    {
        unsigned int N;
        exafmm::real_t* x;
        exafmm::real_t* y;
        exafmm::real_t* z;
        exafmm::real_t* q;
        exafmm::real_t* Fx;
        exafmm::real_t* Fy;
        exafmm::real_t* Fz;
        exafmm::real_t* p;
        exafmm::real_t* Vx;
        exafmm::real_t* Vy;
        exafmm::real_t* Vz;
        exafmm::real_t* acc_old;
        exafmm::real_t* timestamp;
        bool* issink;
        bool* issource;
    } Bodies;

    void bodies_H2D(exafmm::Bodies& h_b, cufmm::Bodies& d_b);
    void bodies_D2H(const Bodies& d_b, std::vector<exafmm::Body>& aos);
    cufmm::Bodies device_bodies_alloc(unsigned int N);
    void device_bodies_free(cufmm::Bodies& d_b);
}

#endif