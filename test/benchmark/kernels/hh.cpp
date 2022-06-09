#include <cmath>

struct hh_Instance {
    const double* __restrict__ gnabar;
    const double* __restrict__ gkbar;
    const double* __restrict__ gl;
    const double* __restrict__ el;
    double* __restrict__ gna;
    double* __restrict__ gk;
    double* __restrict__ il;
    double* __restrict__ minf;
    double* __restrict__ hinf;
    double* __restrict__ ninf;
    double* __restrict__ mtau;
    double* __restrict__ htau;
    double* __restrict__ ntau;
    double* __restrict__ m;
    double* __restrict__ h;
    double* __restrict__ n;
    double* __restrict__ Dm;
    double* __restrict__ Dh;
    double* __restrict__ Dn;
    double* __restrict__ ena;
    double* __restrict__ ek;
    double* __restrict__ ina;
    double* __restrict__ ik;
    double* __restrict__ v_unused;
    double* __restrict__ g_unused;
    const double* __restrict__ ion_ena;
    double* __restrict__ ion_ina;
    double* __restrict__ ion_dinadv;
    const double* __restrict__ ion_ek;
    double* __restrict__ ion_ik;
    double* __restrict__ ion_dikdv;
    int* __restrict__ ion_ena_index;
    int* __restrict__ ion_ina_index;
    int* __restrict__ ion_dinadv_index;
    int* __restrict__ ion_ek_index;
    int* __restrict__ ion_ik_index;
    int* __restrict__ ion_dikdv_index;
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
    int node_id, ena_id, ek_id, ion_dinadv_id, ion_dikdv_id, ion_ina_id, ion_ik_id;
    double v, g, rhs, v_org, current, dina, dik;

    #pragma ivdep
    for (id = 0; id < inst->node_count; id++) {
        node_id = inst->node_index[id];
        ena_id = inst->ion_ena_index[id];
        ek_id = inst->ion_ek_index[id];
        ion_dinadv_id = inst->ion_dinadv_index[id];
        ion_dikdv_id = inst->ion_dikdv_index[id];
        ion_ina_id = inst->ion_ina_index[id];
        ion_ik_id = inst->ion_ik_index[id];
        v = inst->voltage[node_id];
        inst->ena[id] = inst->ion_ena[ena_id];
        inst->ek[id] = inst->ion_ek[ek_id];
        v_org = v;
        v = v + 0.001;
        {
            double current = 0.0;
            inst->gna[id] = inst->gnabar[id] * inst->m[id] * inst->m[id] * inst->m[id] * inst->h[id];
            inst->ina[id] = inst->gna[id] * (v - inst->ena[id]);
            inst->gk[id] = inst->gkbar[id] * inst->n[id] * inst->n[id] * inst->n[id] * inst->n[id];
            inst->ik[id] = inst->gk[id] * (v - inst->ek[id]);
            inst->il[id] = inst->gl[id] * (v - inst->el[id]);
            current += inst->il[id];
            current += inst->ina[id];
            current += inst->ik[id];
            g = current;
        }
        dina = inst->ina[id];
        dik = inst->ik[id];
        v = v_org;
        {
            double current = 0.0;
            inst->gna[id] = inst->gnabar[id] * inst->m[id] * inst->m[id] * inst->m[id] * inst->h[id];
            inst->ina[id] = inst->gna[id] * (v - inst->ena[id]);
            inst->gk[id] = inst->gkbar[id] * inst->n[id] * inst->n[id] * inst->n[id] * inst->n[id];
            inst->ik[id] = inst->gk[id] * (v - inst->ek[id]);
            inst->il[id] = inst->gl[id] * (v - inst->el[id]);
            current += inst->il[id];
            current += inst->ina[id];
            current += inst->ik[id];
            rhs = current;
        }
        g = (g-rhs)/0.001;
        inst->ion_dinadv[ion_dinadv_id] += (dina-inst->ina[id])/0.001;
        inst->ion_dikdv[ion_dikdv_id] += (dik-inst->ik[id])/0.001;
        inst->ion_ina[ion_ina_id] += inst->ina[id];
        inst->ion_ik[ion_ik_id] += inst->ik[id];
        inst->vec_rhs[node_id] -= rhs;
        inst->vec_d[node_id] += g;
    }
}

void nrn_state_ext(void* __restrict__ mech) {
    auto inst = static_cast<hh_Instance*>(mech);
    int id;
    int node_id, ena_id, ek_id;
    double v;

    #pragma ivdep
    for (id = 0; id < inst->node_count; ++id) {
        node_id = inst->node_index[id];
        ena_id = inst->ion_ena_index[id];
        ek_id = inst->ion_ek_index[id];
        v = inst->voltage[node_id];
        inst->ena[id] = inst->ion_ena[ena_id];
        inst->ek[id] = inst->ion_ek[ek_id];
        {
            double alpha, beta, sum, q10, vtrap_in_0, vtrap_in_1, v_in_1;
            v_in_1 = v;
            q10 = pow(3.0, ((inst->celsius - 6.3) / 10.0));
            {
                double x_in_0, y_in_0;
                x_in_0 =  -(v_in_1 + 40.0);
                y_in_0 = 10.0;
                if (fabs(x_in_0 / y_in_0) < 1e-6) {
                    vtrap_in_0 = y_in_0 * (1.0 - x_in_0 / y_in_0 / 2.0);
                } else {
                    vtrap_in_0 = x_in_0 / (exp(x_in_0 / y_in_0) - 1.0);
                }
            }
            alpha = .1 * vtrap_in_0;
            beta = 4.0 * exp( -(v_in_1 + 65.0) / 18.0);
            sum = alpha + beta;
            inst->mtau[id] = 1.0 / (q10 * sum);
            inst->minf[id] = alpha / sum;
            alpha = .07 * exp( -(v_in_1 + 65.0) / 20.0);
            beta = 1.0 / (exp( -(v_in_1 + 35.0) / 10.0) + 1.0);
            sum = alpha + beta;
            inst->htau[id] = 1.0 / (q10 * sum);
            inst->hinf[id] = alpha / sum;
            {
                double x_in_1, y_in_1;
                x_in_1 =  -(v_in_1 + 55.0);
                y_in_1 = 10.0;
                if (fabs(x_in_1 / y_in_1) < 1e-6) {
                    vtrap_in_1 = y_in_1 * (1.0 - x_in_1 / y_in_1 / 2.0);
                } else {
                    vtrap_in_1 = x_in_1 / (exp(x_in_1 / y_in_1) - 1.0);
                }
            }
            alpha = .01 * vtrap_in_1;
            beta = .125 * exp( -(v_in_1 + 65.0) / 80.0);
            sum = alpha + beta;
            inst->ntau[id] = 1.0 / (q10 * sum);
            inst->ninf[id] = alpha / sum;
        }
        inst->m[id] = inst->m[id] + (1.0 - exp(inst->dt * (((( -1.0))) / inst->mtau[id]))) * ( -(((inst->minf[id])) / inst->mtau[id]) / (((( -1.0))) / inst->mtau[id]) - inst->m[id]);
        inst->h[id] = inst->h[id] + (1.0 - exp(inst->dt * (((( -1.0))) / inst->htau[id]))) * ( -(((inst->hinf[id])) / inst->htau[id]) / (((( -1.0))) / inst->htau[id]) - inst->h[id]);
        inst->n[id] = inst->n[id] + (1.0 - exp(inst->dt * (((( -1.0))) / inst->ntau[id]))) * ( -(((inst->ninf[id])) / inst->ntau[id]) / (((( -1.0))) / inst->ntau[id]) - inst->n[id]);
    }
}
