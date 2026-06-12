#include "../include/pca.h"
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <vector>

using namespace Eigen;
using json = nlohmann::json;

static std::mutex g_mutex;
static MatrixXd g_components;  // shape: (n_components, n_features)
static VectorXd g_mean;         // shape: (n_features,)
static bool g_loaded = false;

extern "C" {

int pca_load_model(const char* path) {
    std::lock_guard<std::mutex> lock(g_mutex);
    try {
        std::ifstream f(path);
        if (!f.is_open()) return -1;
        json j;
        f >> j;

        auto comp = j["components"].get<std::vector<std::vector<double>>>();
        auto mean = j["mean"].get<std::vector<double>>();

        int n_comp = comp.size();
        int n_feat = comp[0].size();

        g_components.resize(n_comp, n_feat);
        for (int i = 0; i < n_comp; ++i)
            for (int k = 0; k < n_feat; ++k)
                g_components(i, k) = comp[i][k];

        g_mean.resize(n_feat);
        for (int k = 0; k < n_feat; ++k)
            g_mean(k) = mean[k];

        g_loaded = true;
        return 0;
    } catch (...) {
        return -2;
    }
}

int pca_project(const float* embedding, int dim, double* out_x, double* out_y) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded || dim != (int)g_mean.size()) return -1;

    VectorXd x(dim);
    for (int i = 0; i < dim; ++i) x(i) = (double)embedding[i];
    VectorXd centered = x - g_mean;

    VectorXd projected = g_components * centered;
    *out_x = projected(0);
    *out_y = (projected.size() > 1) ? projected(1) : 0.0;
    return 0;
}

int pca_is_loaded(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_loaded ? 1 : 0;
}

void pca_free(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_components.resize(0, 0);
    g_mean.resize(0);
    g_loaded = false;
}

} // extern "C"
