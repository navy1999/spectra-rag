#include "../include/pca.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

static void write_test_model(const char* path) {
    // PCA on [2,3,5,6]: mean=4, component=[1,0] (1D for test)
    // PC1 of [2,3,5,6] → centered=[-2,-1,1,2], PC1=[1,0,...], so proj=[-2,-1,1,2]
    // We use 4-dim embedding with 2-component PCA
    json j;
    j["components"] = {{1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}};
    j["mean"] = {0.0, 0.0, 0.0, 0.0};
    std::ofstream f(path);
    f << j.dump();
}

int tests_passed = 0;
int tests_total = 0;
#define CHECK(cond, msg) do { tests_total++; if(!(cond)) { printf("FAIL: %s\n", msg); return; } printf("PASS: %s\n", msg); tests_passed++; } while(0)

void test_not_loaded_initially() {
    CHECK(pca_is_loaded() == 0, "not loaded initially");
}

void test_load_model() {
    write_test_model("/tmp/test_model.json");
    int ret = pca_load_model("/tmp/test_model.json");
    CHECK(ret == 0, "load_model returns 0");
    CHECK(pca_is_loaded() == 1, "is_loaded after load");
}

void test_project_identity() {
    float emb[4] = {3.0f, 0.0f, 0.0f, 0.0f};
    double x, y;
    int ret = pca_project(emb, 4, &x, &y);
    CHECK(ret == 0, "project returns 0");
    CHECK(fabs(x - 3.0) < 1e-6, "PC1 correct (3.0)");
    CHECK(fabs(y - 0.0) < 1e-6, "PC2 correct (0.0)");
}

void test_project_pc2() {
    float emb[4] = {0.0f, 5.0f, 0.0f, 0.0f};
    double x, y;
    pca_project(emb, 4, &x, &y);
    CHECK(fabs(x - 0.0) < 1e-6, "PC1=0 for y-axis input");
    CHECK(fabs(y - 5.0) < 1e-6, "PC2=5 for y-axis input");
}

void test_wrong_dim() {
    float emb[3] = {1.0f, 2.0f, 3.0f};
    double x, y;
    int ret = pca_project(emb, 3, &x, &y);
    CHECK(ret != 0, "wrong dim returns error");
}

void test_bad_path() {
    pca_free();
    int ret = pca_load_model("/nonexistent/path.json");
    CHECK(ret != 0, "bad path returns error");
}

void test_free() {
    write_test_model("/tmp/test_model2.json");
    pca_load_model("/tmp/test_model2.json");
    pca_free();
    CHECK(pca_is_loaded() == 0, "not loaded after free");
}

int main() {
    printf("=== PCA Engine Tests ===\n");
    test_not_loaded_initially();
    test_load_model();
    test_project_identity();
    test_project_pc2();
    test_wrong_dim();
    test_bad_path();
    test_free();
    printf("\n%d/%d checks passed\n", tests_passed, tests_total);
    return (tests_passed == tests_total && tests_total > 0) ? 0 : 1;
}
