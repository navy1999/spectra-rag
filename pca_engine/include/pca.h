#pragma once
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Load a PCA model from a JSON file.
 * @param path  Path to pca_model.json produced by scripts/fit_pca.py
 * @return 0 on success, non-zero on error
 */
int pca_load_model(const char* path);

/**
 * Project an embedding to 2D PCA space.
 * @param embedding  Float32 array of length `dim`
 * @param dim        Embedding dimensionality (e.g. 384)
 * @param out_x      Output first principal component
 * @param out_y      Output second principal component
 * @return 0 on success
 */
int pca_project(const float* embedding, int dim, double* out_x, double* out_y);

/** Returns 1 if a model is loaded, 0 otherwise. */
int pca_is_loaded(void);

/** Free model resources. */
void pca_free(void);

#ifdef __cplusplus
}
#endif
