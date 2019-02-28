/// list of data structures required for ISPC backend

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
    uniform int* nodeindices;
    uniform int* _permute;
    uniform double* data;
    uniform Datum* pdata;
    uniform ThreadDatum* _thread;
    uniform NetReceiveBuffer_t* _net_receive_buffer;
    // NetSendBuffer_t* _net_send_buffer; // not needed
    int nodecount;
    int _nodecount_padded;
    void* uniform instance;
};

struct Point_process {
    int _i_instance;
    int16 _type;
    int16 _tid;
};


struct NrnThread {
    double _t;
    double _dt;
    double cj;

    uniform Memb_list** uniform _ml_list;
    Point_process* uniform pntprocs;
    double* uniform weights;

    int n_pntproc, n_presyn, n_input_presyn, n_netcon, n_weight;

    int ncell;
    int end;
    int id;

    size_t _ndata, _nidata, _nvdata;
    double* uniform _data;
    int* uniform _idata;

    double* uniform _actual_rhs;
    double* uniform _actual_d;
    double* uniform _actual_a;
    double* uniform _actual_b;
    double* uniform _actual_v;
    double* uniform _actual_area;
    double* uniform _actual_diam;
    double* uniform _shadow_rhs;
    double* uniform _shadow_d;
    int* uniform _v_parent_index;
    int* uniform _permute;

    int shadow_rhs_cnt;
    int compute_gpu;
    int stream_id;
    //int _net_send_buffer_size;
    //int _net_send_buffer_cnt;
    //int* _net_send_buffer;
};

uniform double ispc_celsius;
