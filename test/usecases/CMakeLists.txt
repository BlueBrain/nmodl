set(NMODL_USECASE_DIRS
    cnexp_scalar
    cnexp_array
    global_breakpoint
    point_process
    parameter
    func_proc
    func_proc_pnt)

foreach(usecase ${NMODL_USECASE_DIRS})
  add_test(NAME usecase_${usecase}
           COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_test.sh ${CMAKE_BINARY_DIR}/bin/nmodl
                   ${CMAKE_CURRENT_SOURCE_DIR}/${usecase})
endforeach()