if(USE_MEMOPT)
    if(USE_CUQUANTUM)
        message(FATAL_ERROR "memopt and cuQuantum cannot be enabled at the same time")
    endif()

    set(QuEST_SRC_ARCHITECTURE_DEPENDENT
        ${CMAKE_CURRENT_SOURCE_DIR}/QuEST_gpu_memopt.cu
        PARENT_SCOPE
    )
else()
    if(USE_CUQUANTUM)
        set(GPU_CORE QuEST_cuQuantum.cu)
    else()
        set(GPU_CORE QuEST_gpu.cu)
    endif()

    set(QuEST_SRC_ARCHITECTURE_DEPENDENT
        ${CMAKE_CURRENT_SOURCE_DIR}/${GPU_CORE}
        ${CMAKE_CURRENT_SOURCE_DIR}/QuEST_gpu_common.cu
        PARENT_SCOPE
    )
endif()
