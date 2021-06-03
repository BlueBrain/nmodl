

struct hh_Instance  {
    double* __restrict__ minf;
    double* __restrict__ mtau;
    double* __restrict__ m;
    double* __restrict__ nai;
    double* __restrict__ Dm;
    double* __restrict__ v_unused;
    double* __restrict__ g_unused;
    double* __restrict__ ion_nai;
    double* __restrict__ style_na;
    int* __restrict__ ion_nai_index;
    int* __restrict__ style_na_index;
    double* __restrict__ voltage;
    int* __restrict__ node_index;
    double t;
    double dt;
    double celsius;
    int secondorder;
    int node_count;
};

void nrn_state_hh_ext(void* __restrict__ mech){
    auto inst = static_cast<hh_Instance*>(mech);
    int id;
    int node_id, nai_id, ion_nai_id;
    double v;
    for(int id = 0; id<inst->node_count; ++id) {
        node_id = inst->node_index[id];
        nai_id = inst->ion_nai_index[id];
        ion_nai_id = inst->ion_nai_index[id];
        v = inst->voltage[node_id];
        inst->nai[id] = inst->ion_nai[nai_id];
        inst->m[id] = (inst->minf[id]-inst->m[id])/inst->mtau[id];
        inst->ion_nai[ion_nai_id] = inst->nai[id];
    }
}
