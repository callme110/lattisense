add_executable(ckks_softmax_cpu ckks_softmax_cpu.cpp)
target_link_libraries(ckks_softmax_cpu PRIVATE lattisense)

set_target_properties(ckks_softmax_cpu PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../../lib"
    BUILD_WITH_INSTALL_RPATH OFF
)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ckks_softmax_cpu.py
               ${CMAKE_CURRENT_BINARY_DIR}/ckks_softmax_cpu.py COPYONLY)
