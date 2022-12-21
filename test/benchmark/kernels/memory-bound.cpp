

struct hh_Instance  {
    const double* __restrict__ gl;
    const double* __restrict__ el;
    double* __restrict__ minf;
    double* __restrict__ mtau;
    double* __restrict__ il;
    double* __restrict__ m;
    double* __restrict__ nai;
    double* __restrict__ Dm;
    double* __restrict__ v_unused;
    double* __restrict__ g_unused;
    double* __restrict__ ion_nai;
    const double* __restrict__ style_na;
    int* __restrict__ ion_nai_index;
    int* __restrict__ style_na_index;
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
    auto inst = static_cast<hh_Instance*>(mech);
    int id;
    int node_id, ion_nai_index;
    double v, g, rhs, v_org, current;

    #pragma ivdep
    for (id = 0; id < inst->node_count; id++) {
        node_id = inst->node_index[id];
        ion_nai_index = inst->ion_nai_index[id];
        v = inst->voltage[node_id];
        inst->nai[id] = inst->ion_nai[ion_nai_index];
        v_org = v;
        v = v + 0.001;
        {
            current = 0.0;
            inst->il[id] = inst->gl[id] * (v - inst->el[id]);
            current += inst->il[id];
            g = current;
        }
        v = v_org;
        {
            current = 0.0;
            inst->il[id] = inst->gl[id] * (v - inst->el[id]);
            current += inst->il[id];
            rhs = current;
        }
        g = (g-rhs)/0.001;
        inst->ion_nai[ion_nai_index] = inst->nai[id];
        inst->vec_rhs[node_id] -= rhs;
        inst->vec_d[node_id] += g;
    }
}

void nrn_state_ext(void* __restrict__ mech) {
    auto inst = static_cast<hh_Instance*>(mech);
    int id;
    int node_id, nai_id, ion_nai_id;
    double v;

    #pragma ivdep
    for (int id = 0; id < inst->node_count; ++id) {
        node_id = inst->node_index[id];
        nai_id = inst->ion_nai_index[id];
        ion_nai_id = inst->ion_nai_index[id];
        v = inst->voltage[node_id];
        inst->nai[id] = inst->ion_nai[nai_id];
        inst->m[id] = (inst->minf[id] - inst->m[id]) / inst->mtau[id];
        inst->ion_nai[ion_nai_id] = inst->nai[id];
    }
}