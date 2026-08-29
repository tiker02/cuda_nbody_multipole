#ifndef EXAFMM_CUH
#define EXAFMM_CUH

#include "exafmm.h"
#pragma once

namespace cufmm
{    
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


    void bodies_H2D(exafmm::Bodies& h_b, cufmm::Bodies& d_b);
    void bodies_D2H(const Bodies& d_b, std::vector<exafmm::Body>& aos);
    cufmm::Bodies device_bodies_alloc(unsigned int N);
    void device_bodies_free(cufmm::Bodies& d_b);


    template<Implementation impl> void cuP2P_launch(cufmm::Bodies d_bodies);

}

#endif