project(cuMat)
# Define 'cuMat_headers' variable to list all the header files
file(GLOB_RECURSE CUMAT_HEADERS src/*.h src/*.inl)
# Append "Core" "Dense" "Sparse" "IterativeLinearSolvers"
set(CUMAT_HEADERS "${CUMAT_HEADERS};Core;Dense;Sparse;IterativeLinearSolvers;")

add_library(cuMat INTERFACE)

#
# From here, the target 'moduleA' can be customised
#
target_include_directories(cuMat INTERFACE ${PROJECT_SOURCE_DIR}) # Transitively forwarded
target_include_directories(cuMat INTERFACE ${CUDA_INCLUDE_DIRS})
target_link_libraries(cuMat INTERFACE ${CUDA_LIBRARIES})
#install(TARGETS cuMat ...)

#
#  HACK: have the files showing in the IDE
#
option(CUMAT_SOURCE_LIBRARY "show the source headers as a library in the IDE" OFF)
if(CUMAT_SOURCE_LIBRARY)
  cuda_add_library(cuMat_ide ${CUMAT_HEADERS})
  target_include_directories(cuMat_ide INTERFACE ${CMAKE_SOURCE_DIR}) # Transitively forwarded
  target_include_directories(cuMat_ide INTERFACE ${CUDA_INCLUDE_DIRS})

  # source group by hierarchy
  source_group(TREE ${PROJECT_SOURCE_DIR} FILES ${CUMAT_HEADERS})
endif(CUMAT_SOURCE_LIBRARY)


