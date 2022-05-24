#include <cmath>

struct hh_Instance {
    const double* __restrict__ gl;
    const double* __restrict__ el;
    double* __restrict__ minf;
    double* __restrict__ mtau;
    double* __restrict__ il;
    double* __restrict__ m;
    double* __restrict__ Dm;
    double* __restrict__ v_unused;
    double* __restrict__ g_unused;
    double* __restrict__ voltage;
    int* __restrict__ node_index;
    double* __restrict__ vec_rhs;
    double* __restrict__ vec_d;
    double* __restrict__ _shadow_rhs;
    double* __restrict__ _shadow_d;
    double t;
    double dt;
    double celsius;
    int secondorder;
    int node_count;
};

void nrn_state_hh_ext(void* __restrict__ mech) {
    auto inst = static_cast<hh_Instance*>(mech);
    int id;
    int node_id;
    double v;
    for (int id = 0; id < inst->node_count; ++id) {
        node_id = inst->node_index[id];
        v = inst->voltage[node_id];
        inst->m[id] = exp(inst->m[id]) + exp(inst->minf[id]) +
                      (inst->minf[id] - inst->m[id]) / inst->mtau[id] + inst->m[id] +
                      inst->minf[id] * inst->mtau[id];
    }
}
