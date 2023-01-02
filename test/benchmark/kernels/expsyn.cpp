#include <cmath>

struct ExpSyn_Instance {
    const double* __restrict__ tau;
    const double* __restrict__ e;
    double* __restrict__ i;
    double* __restrict__ g_state;
    double* __restrict__ Dg_state;
    double* __restrict__ v_unused;
    double* __restrict__ g_unused;
    double* __restrict__ tsave;
    const double* __restrict__ node_area;
    const double* __restrict__ point_process;
    int* __restrict__ node_area_index;
    int* __restrict__ point_process_index;
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

void nrn_cur_ext(void* __restrict__ mech) {
    auto inst = static_cast<ExpSyn_Instance*>(mech);
    int id;
    int node_id, node_area_id;
    double v, g, rhs, v_org, current, mfactor;

    #pragma ivdep
    for (id = 0; id < inst->node_count; id++) {
        node_id = inst->node_index[id];
        node_area_id = inst->node_area_index[id];
        v = inst->voltage[node_id];
        v_org = v;
        v = v + 0.001;
        {
            current = 0.0;
            inst->i[id] = inst->g_state[id] * (v - inst->e[id]);
            current += inst->i[id];
            g = current;
        }
        v = v_org;
        {
            current = 0.0;
            inst->i[id] = inst->g_state[id] * (v - inst->e[id]);
            current += inst->i[id];
            rhs = current;
        }
        g = (g-rhs)/0.001;
        mfactor = 1.e2/inst->node_area[node_area_id];
        g = g*mfactor;
        rhs = rhs*mfactor;
        inst->_shadow_rhs[id] = rhs;
        inst->_shadow_d[id] = g;
    }
    for (id = 0; id < inst->node_count; id++) {
        node_id = inst->node_index[id];
        inst->vec_rhs[node_id] -= inst->_shadow_rhs[id];
        inst->vec_d[node_id] += inst->_shadow_d[id];
    }
}

void nrn_state_ext(void* __restrict__ mech) {
    auto inst = static_cast<ExpSyn_Instance*>(mech);
    int id;
    int node_id;
    double v;

    #pragma ivdep
    for (id = 0; id < inst->node_count; ++id) {
        node_id = inst->node_index[id];
        v = inst->voltage[node_id];
        inst->g_state[id] = inst->g_state[id] + (1.0 - exp(inst->dt * (( -1.0) / inst->tau[id]))) * ( -(0.0) / (( -1.0) / inst->tau[id]) - inst->g_state[id]);
    }
}
