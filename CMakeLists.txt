cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_SUPPRESS_REGENERATION true)

project(MeshletMaker VERSION 0.5)

option(BUILD_STANDALONE "Build standalone" TRUE)


include_directories(
	${PROJECT_SOURCE_DIR}/libs/eigen3/
	${PROJECT_SOURCE_DIR}/libs/tinyobjloader/
	${PROJECT_SOURCE_DIR}/libs/glm/
    ${PROJECT_SOURCE_DIR}/libs/meshOptimizer/
)

set(SOURCES
	core/geometryProcessing.cpp
	core/idxBufCleaner.cpp
	core/meshletConverter.cpp
	core/meshletCompresser.cpp
	core/meshletMeshDescriptor.cpp
	core/meshletTaskDescriptor.cpp
)

set(HEADERS
	core/geometryProcessing.h
	core/mm_meshlet_builder.h
	core/meshlet_builder.hpp
	core/meshlet_util.hpp
	core/mm_structures.h
	core/meshletMaker.h
	core/settings.h
)

add_library(meshletMaker ${SOURCES} ${HEADERS})


set(TARGETS meshletMaker)

# Requires glfw3 lib for target platform
target_link_libraries(
	${TARGETS}
    ${PROJECT_SOURCE_DIR}/libs/meshOptimizer/meshoptimizer.lib
    )

install(TARGETS ${TARGETS} EXPORT meshletMakerTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES 
	core/meshletMaker.h DESTINATION include ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES 
	core/settings.h DESTINATION include ${CMAKE_INSTALL_INCLUDEDIR})

if (BUILD_STANDALONE)
	add_subdirectory(Standalone)
endif()
