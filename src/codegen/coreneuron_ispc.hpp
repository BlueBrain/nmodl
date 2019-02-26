/// list of data structures required for ISPC backend
/// added here just as placeholder

typedef int Datum;

struct ThreadDatum {
    int i;
    double* pval;
    void* _pvoid;
};

struct NetReceiveBuffer_t {
    int* _displ;
    int* _nrb_index;

    int* _pnt_index;
    int* _weight_index;
    double* _nrb_t;
    double* _nrb_flag;
    int _cnt;
    int _displ_cnt;
    int _size;
    int _pnt_offset;
};

struct Memb_list {
    int* nodeindices;
    int* _permute;
    double* data;
    Datum* pdata;
    ThreadDatum* _thread;
    NetReceiveBuffer_t* _net_receive_buffer;
    NetSendBuffer_t* _net_send_buffer;
    int nodecount;
    int _nodecount_padded;
    void* instance; // this is of type hh_Instance
};

typedef struct Point_process {
    int _i_instance;
    short _type;
    short _tid;
} Point_process;

struct NrnThread {
    double _t;
    double _dt;
    double cj;

    Memb_list** _ml_list;
    Point_process* pntprocs;
    double* weights;

    int n_pntproc, n_presyn, n_input_presyn, n_netcon, n_weight;

    int ncell;
    int end;
    int id;

    size_t _ndata, _nidata, _nvdata;
    double* _data;
    int* _idata;

    double* _actual_rhs;
    double* _actual_d;
    double* _actual_a;
    double* _actual_b;
    double* _actual_v;
    double* _actual_area;
    double* _actual_diam;
    double* _shadow_rhs;
    double* _shadow_d;
    int* _v_parent_index;
    int* _permute;

    int shadow_rhs_cnt;
    int compute_gpu;
    int stream_id;
    int _net_send_buffer_size;
    int _net_send_buffer_cnt;
    int* _net_send_buffer;
};
