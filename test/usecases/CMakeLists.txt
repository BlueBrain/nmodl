set(NMODL_USECASE_DIRS
    at_time
    builtin_functions
    constant
    constructor
    electrode_current
    external
    function
    function_table
    global
    hodgkin_huxley
    kinetic
    linear
    morphology
    net_event
    net_move
    net_receive
    net_send
    neuron_variables
    nonspecific_current
    parameter
    point_process
    pointer
    procedure
    random
    solve
    state
    steady_state
    suffix
    table
    useion)

foreach(usecase ${NMODL_USECASE_DIRS})
  add_test(NAME usecase_${usecase}
           COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_test.sh ${CMAKE_BINARY_DIR}/bin/nmodl
                   ${CMAKE_CURRENT_SOURCE_DIR}/${usecase})
endforeach()
