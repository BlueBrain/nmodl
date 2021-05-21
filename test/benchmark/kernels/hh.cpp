#include <cmath>

struct hh_Instance  {
    double* __restrict__ gnabar;
    double* __restrict__ gkbar;
    double* __restrict__ gl;
    double* __restrict__ el;
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
    double* __restrict__ ion_ena;
    double* __restrict__ ion_ina;
    double* __restrict__ ion_dinadv;
    double* __restrict__ ion_ek;
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
    double t;
    double dt;
    double celsius;
    int secondorder;
    int node_count;
};

void nrn_state_hh_ext(void* __restrict__ mech){
    auto inst = static_cast<hh_Instance*>(mech);
    int id;
    int node_id, ena_id, ek_id;
    double v;
    for(id = 0; id<inst->node_count; id = ++id) {
        node_id = inst->node_index[id];
        ena_id = inst->ion_ena_index[id];
        ek_id = inst->ion_ek_index[id];
        v = inst->voltage[node_id];
        inst->ena[id] = inst->ion_ena[ena_id];
        inst->ek[id] = inst->ion_ek[ek_id];
        {
            double alpha, beta, sum, q10, vtrap_in_0, v_in_1;
            v_in_1 = v;
            q10 = 3*((inst->celsius-6.3)/10);
            alpha = .07*exp(-(v_in_1+65)/20);
            beta = 1/(exp(-(v_in_1+35)/10)+1);
            sum = alpha+beta;
            inst->htau[id] = 1/(q10*sum);
            inst->hinf[id] = alpha/sum;
            {
                double x_in_0, y_in_0;
                x_in_0 = alpha;
                y_in_0 = alpha;
                vtrap_in_0 = y_in_0*(1-x_in_0/y_in_0/2);
            }
            inst->hinf[id] = vtrap_in_0;
        }
        inst->m[id] = inst->m[id]+(1.0-exp(inst->dt*((((-1.0)))/inst->mtau[id])))*(-(((inst->minf[id]))/inst->mtau[id])/((((-1.0)))/inst->mtau[id])-inst->m[id]);
        inst->h[id] = inst->h[id]+(1.0-exp(inst->dt*((((-1.0)))/inst->htau[id])))*(-(((inst->hinf[id]))/inst->htau[id])/((((-1.0)))/inst->htau[id])-inst->h[id]);
        inst->n[id] = inst->n[id]+(1.0-exp(inst->dt*((((-1.0)))/inst->ntau[id])))*(-(((inst->ninf[id]))/inst->ntau[id])/((((-1.0)))/inst->ntau[id])-inst->n[id]);
    }
}
