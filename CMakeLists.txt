cmake_minimum_required(VERSION 3.5)
project(tf_test)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES cc/play.cc)


include_directories(
	/usr/local/include/tf
	)

add_executable(tf_test ${SOURCE_FILES})

target_link_libraries(tf_test
        /usr/local/lib/libtensorflow_cc.so
        /usr/local/lib/libtensorflow_framework.so
        )
