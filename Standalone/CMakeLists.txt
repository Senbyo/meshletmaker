cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_SUPPRESS_REGENERATION true)

include_directories(
	${PROJECT_SOURCE_DIR}/libs/tinyobjloader/
	${PROJECT_SOURCE_DIR}/core
	${PROJECT_SOURCE_DIR}
)

set(SOURCES
	main.cpp
)

set(HEADERS
)

set(TARGETS standalone)

add_executable(standalone ${SOURCES} ${HEADERS})


target_link_libraries(standalone
						meshletmaker
						${SYSTEM_LIBS})

install(TARGETS ${TARGETS} EXPORT meshletMakerTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)