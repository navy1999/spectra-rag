cmake_minimum_required(VERSION 3.16)
project(pca_engine CXX)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

# Eigen's project registers its ~900-test suite with CTest when BUILD_TESTING is
# on (the executables are EXCLUDE_FROM_ALL, so ctest lists them all as "Not Run"
# and exits non-zero). Turn off subproject testing/docs; our own pca_tests below
# is registered via enable_testing()/add_test and is unaffected by this option.
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(json)

add_library(pca SHARED src/pca.cpp)
target_include_directories(pca PRIVATE include)
target_link_libraries(pca PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json)
target_compile_options(pca PRIVATE -O2 -march=native -fPIC)
set_target_properties(pca PROPERTIES BUILD_RPATH "$ORIGIN")

# Tests
enable_testing()
add_executable(test_pca test/test_pca.cpp)
target_include_directories(test_pca PRIVATE include)
target_link_libraries(test_pca PRIVATE pca Eigen3::Eigen nlohmann_json::nlohmann_json)
add_test(NAME pca_tests COMMAND test_pca)
