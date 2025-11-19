cmake_minimum_required(VERSION 3.19)
cmake_policy(SET CMP0177 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_SUPPRESS_REGENERATION true)

option(ENABLE_HIGHFIVE "Enable HDF5 support via HighFive" FALSE)
option(BUILD_STANDALONE "Build standalone" FALSE)

if(ENABLE_HIGHFIVE)
	list(APPEND VCPKG_MANIFEST_FEATURES "hdf5-support")
endif()

project(meshletmaker VERSION 1.0)

find_package(glm REQUIRED)

if (ENABLE_HIGHFIVE)
    find_package(HighFive CONFIG REQUIRED)
endif()

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

add_library(meshletmaker STATIC ${SOURCES} ${HEADERS})

target_include_directories(meshletmaker
	PRIVATE
	${PROJECT_SOURCE_DIR}/libs/tinyobjloader/
)

target_link_libraries(meshletmaker 
	PRIVATE 
	glm::glm
)

if (ENABLE_HIGHFIVE)
    target_link_libraries(meshletmaker PRIVATE HighFive::HighFive)
    target_compile_definitions(meshletmaker PUBLIC HIGHFIVE_SUPPORT)
endif()

set(INCLUDES 
	core/MeshletMaker.h
	core/settings.h
)

install(FILES ${INCLUDES} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

if (BUILD_STANDALONE)
	add_subdirectory(Standalone)
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                 PROPERTY 
                 VS_STARTUP_PROJECT standalone
	)
endif()
