#include <cmath>

struct hh_Instance  {               // address
    double* __restrict__ minf;      //  0
    double* __restrict__ mtau;      //  8
    double* __restrict__ m;         // 16
    double* __restrict__ Dm;        // 24
    double* __restrict__ v_unused;  // 32
    double* __restrict__ g_unused;  // 40
    double* __restrict__ voltage;   // 48
    int* __restrict__ node_index;   // 56
    double t;                       // 64
    double dt;                      // 72
    double celsius;                 // 80
    int secondorder;                // 88
    int node_count;                 // 92
};

void nrn_state_hh_ext(void* __restrict__ mech){
    auto inst = static_cast<hh_Instance*>(mech);
    int id;
    int node_id;
    double v;
    for(int id = 0; id<inst->node_count; ++id) {
        node_id = inst->node_index[id];
        v = inst->voltage[node_id];
        inst->m[id] = exp(inst->m[id])+exp(inst->minf[id])+(inst->minf[id]-inst->m[id])/inst->mtau[id]+inst->m[id]+inst->minf[id]*inst->mtau[id];
    }
}
