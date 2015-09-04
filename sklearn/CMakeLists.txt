if(NOT TARGET sx)
    find_package(sx REQUIRED)
endif()

file(GLOB_RECURSE files *.cpp *.h)

add_library(sklcpp ${files})
target_link_libraries(sklcpp PUBLIC sx)

if(MSVC)
	target_compile_definitions(sklcpp PRIVATE _SCL_SECURE_NO_WARNINGS)
endif()

target_include_directories(sklcpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/..>
        $<INSTALL_INTERFACE:include>
)

if(NOT UNIX AND NOT BUILD_SHARED_LIBS)
    set_target_properties(sklcpp PROPERTIES OUTPUT_NAME libsklcpp)
endif()

link_libraries(sklcpp)

add_executable(tree-playground tree/tests/tree-playground.cpp)
