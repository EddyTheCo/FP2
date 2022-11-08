cmake_minimum_required(VERSION 3.11 FATAL_ERROR)



project(FP2 VERSION 0.1 DESCRIPTION "Custom dataset following Feature Proposal 2 FP2" LANGUAGES CXX)


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

add_library(FP2 fp2.cpp include/custom-datasets/fp2.hpp)


find_package(Torch REQUIRED)


target_include_directories(FP2 PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	"$<INSTALL_INTERFACE:include>")

target_link_libraries(FP2 PUBLIC  "${TORCH_LIBRARIES}")


install(TARGETS FP2 EXPORT FP2Targets DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/)

install(EXPORT FP2Targets
	FILE FP2Targets.cmake
	DESTINATION lib/cmake/FP2
	)
include(CMakePackageConfigHelpers)
# generate the config file that is includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
	"${CMAKE_CURRENT_BINARY_DIR}/FP2Config.cmake"
	INSTALL_DESTINATION "lib/cmake/FP2"
	NO_SET_AND_CHECK_MACRO
	NO_CHECK_REQUIRED_COMPONENTS_MACRO
	)
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/FP2ConfigVersion.cmake"
	VERSION "0.1.1.0"
	COMPATIBILITY AnyNewerVersion
	)
install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/FP2Config.cmake
	${CMAKE_CURRENT_BINARY_DIR}/FP2ConfigVersion.cmake
	DESTINATION lib/cmake/FP2
	)
export(EXPORT FP2Targets
	FILE "${CMAKE_CURRENT_BINARY_DIR}/FP2Targets.cmake"
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
