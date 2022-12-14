cmake_minimum_required(VERSION 3.11 FATAL_ERROR)



project(FP VERSION 0.1 DESCRIPTION "Custom dataset following Feature Proposals " LANGUAGES CXX)


set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
	set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
		STRING "Choose the type of build." FORCE)
	# Set the possible values of build type for cmake-gui
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
		"Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(FPX fpx.cpp include/custom-datasets/fpx.hpp)


find_package(Torch REQUIRED)


target_include_directories(FPX PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	"$<INSTALL_INTERFACE:include>")

target_link_libraries(FPX PUBLIC  "${TORCH_LIBRARIES}")


install(TARGETS FPX EXPORT FPXTargets DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/)

install(EXPORT FPXTargets
	FILE FPXTargets.cmake
	DESTINATION lib/cmake/FPX
	)
include(CMakePackageConfigHelpers)
# generate the config file that is includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
	"${CMAKE_CURRENT_BINARY_DIR}/FPXConfig.cmake"
	INSTALL_DESTINATION "lib/cmake/FPX"
	NO_SET_AND_CHECK_MACRO
	NO_CHECK_REQUIRED_COMPONENTS_MACRO
	)
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/FPXConfigVersion.cmake"
	VERSION "0.1.1.0"
	COMPATIBILITY AnyNewerVersion
	)
install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/FPXConfig.cmake
	${CMAKE_CURRENT_BINARY_DIR}/FPXConfigVersion.cmake
	DESTINATION lib/cmake/FPX
	)
export(EXPORT FPXTargets
	FILE "${CMAKE_CURRENT_BINARY_DIR}/FPXTargets.cmake"
	)
execute_process(COMMAND "git" "log" "--format='%H'" "-n 1"
	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
	OUTPUT_VARIABLE  commit)
execute_process(COMMAND "git" "rev-parse" "--abbrev-ref" "HEAD"
	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
	OUTPUT_VARIABLE  branch)

FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt  ${PROJECT_NAME}:\n)
FILE(APPEND ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt  \t commit=${commit})
FILE(APPEND ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt  \t branch=${branch}\n)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt DESTINATION .conf)
