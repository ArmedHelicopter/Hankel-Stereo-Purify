/* Batched SVD truncation + reconstruction via cuSOLVER.

Supports two usage patterns:
  1. One-shot:  mssa_svd_batch() — allocates/frees per call
  2. Persistent: mssa_svd_init(m, n, max_batch) → mssa_svd_run() → mssa_svd_cleanup()

All matrices are COLUMN-MAJOR.  Python wrapper handles C-contiguous ↔ col-major.

Build:
  cd src/cuda && /usr/local/cuda-12.8/bin/nvcc -shared -Xcompiler -fPIC \
    -o libmssa_svd.so mssa_svd.cu -lcudart -lcusolver -lcublas
*/

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,         \
                    __LINE__, cudaGetErrorString(err));                      \
            return -1;                                                      \
        }                                                                   \
    } while (0)

#define CHECK_CUSOLVER(call)                                                \
    do {                                                                    \
        cusolverStatus_t err = (call);                                      \
        if (err != CUSOLVER_STATUS_SUCCESS) {                               \
            fprintf(stderr, "cuSOLVER error at %s:%d: code %d\n",          \
                    __FILE__, __LINE__, (int)err);                          \
            return -2;                                                      \
        }                                                                   \
    } while (0)


/* ================================================================
 * Reconstruct truncated SVD on GPU.
 *
 * For each frame i, row r, col c (column-major m×n):
 *   out[i][r,c] = sum_{j=0}^{k-1} S[i][j] * U[i][r,j] * VT[i][j,c]
 *
 * U:  (m × mn) col-major   — left singular vectors
 * S:  (mn)      per frame   — singular values
 * VT: (mn × n)  col-major   — right singular vectors
 * out:(m × n)   col-major   — reconstructed matrix
 * ================================================================ */
__global__ void reconstruct_kernel(
    const double* __restrict__ U,
    const double* __restrict__ S,
    const double* __restrict__ VT,
    double* __restrict__ out,
    int N, int m, int n, int mn, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * m * n;
    if (idx >= total) return;

    int i   = idx / (m * n);
    int rem = idx % (m * n);
    /* Column-major indexing: idx = c * m + r */
    int c   = rem / m;
    int r   = rem % m;

    long long fU  = (long long)i * m * mn;
    long long fS  = (long long)i * mn;
    long long fVT = (long long)i * mn * n;

    double val = 0.0;
    for (int j = 0; j < k; j++) {
        val += S[fS + j] * U[fU + (long long)j * m + r] * VT[fVT + (long long)c * mn + j];
    }
    out[idx] = val;
}


/* ================================================================
 * Persistent context — pre-allocates GPU buffers.
 * ================================================================ */

static cusolverDnHandle_t g_solver = NULL;
static double *g_d_U = NULL, *g_d_S = NULL, *g_d_VT = NULL;
static double *g_d_input = NULL, *g_d_output = NULL;
static double *g_d_work = NULL, *g_d_rwork = NULL;
static int    *g_d_info = NULL;
static int     g_m = 0, g_n = 0, g_mn = 0, g_lwork = 0, g_max_batch = 0;
static size_t  g_mat_bytes = 0;

/* Forward declarations */
extern "C" void mssa_svd_cleanup(void);

extern "C" int mssa_svd_init(int m, int n, int max_batch)
{
    if (g_solver != NULL && g_m == m && g_n == n && g_max_batch >= max_batch)
        return 0;

    mssa_svd_cleanup();

    CHECK_CUSOLVER(cusolverDnCreate(&g_solver));

    g_m = m;  g_n = n;  g_max_batch = max_batch;
    g_mn = (m < n) ? m : n;

    CHECK_CUSOLVER(cusolverDnDgesvd_bufferSize(g_solver, m, n, &g_lwork));

    g_mat_bytes = (size_t)m * n * sizeof(double);
    size_t batch_mat = (size_t)max_batch * g_mat_bytes;
    size_t batch_U   = (size_t)max_batch * m * g_mn * sizeof(double);
    size_t batch_S   = (size_t)max_batch * g_mn * sizeof(double);
    size_t batch_VT  = (size_t)max_batch * g_mn * n * sizeof(double);

    CHECK_CUDA(cudaMalloc(&g_d_input,  batch_mat));
    CHECK_CUDA(cudaMalloc(&g_d_output, batch_mat));
    CHECK_CUDA(cudaMalloc(&g_d_U,      batch_U));
    CHECK_CUDA(cudaMalloc(&g_d_S,      batch_S));
    CHECK_CUDA(cudaMalloc(&g_d_VT,     batch_VT));
    CHECK_CUDA(cudaMalloc(&g_d_work,   g_lwork * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&g_d_rwork,  g_mn * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&g_d_info,   sizeof(int)));

    return 0;
}

extern "C" void mssa_svd_cleanup(void)
{
    if (g_d_input)  { cudaFree(g_d_input);  g_d_input  = NULL; }
    if (g_d_output) { cudaFree(g_d_output); g_d_output = NULL; }
    if (g_d_U)      { cudaFree(g_d_U);      g_d_U      = NULL; }
    if (g_d_S)      { cudaFree(g_d_S);      g_d_S      = NULL; }
    if (g_d_VT)     { cudaFree(g_d_VT);     g_d_VT     = NULL; }
    if (g_d_work)   { cudaFree(g_d_work);   g_d_work   = NULL; }
    if (g_d_rwork)  { cudaFree(g_d_rwork);  g_d_rwork  = NULL; }
    if (g_d_info)   { cudaFree(g_d_info);   g_d_info   = NULL; }
    if (g_solver)   { cusolverDnDestroy(g_solver); g_solver = NULL; }
    g_m = g_n = g_mn = g_lwork = g_max_batch = 0;
    g_mat_bytes = 0;
}

extern "C" int mssa_svd_upload(const double* input_host, int N)
{
    CHECK_CUDA(cudaMemcpy(g_d_input, input_host,
                          (size_t)N * g_mat_bytes, cudaMemcpyHostToDevice));
    return 0;
}

extern "C" int mssa_svd_run(int N, int rank)
{
    if (g_solver == NULL || N <= 0) return -1;
    if (rank > g_mn) rank = g_mn;

    for (int i = 0; i < N; i++) {
        double* Ai  = g_d_input  + (long long)i * g_m * g_n;
        double* Ui  = g_d_U      + (long long)i * g_m * g_mn;
        double* Si  = g_d_S      + (long long)i * g_mn;
        double* VTi = g_d_VT     + (long long)i * g_mn * g_n;

        CHECK_CUSOLVER(cusolverDnDgesvd(
            g_solver, 'S', 'S',
            g_m, g_n,
            Ai, g_m,
            Si,
            Ui, g_m,
            VTi, g_mn,
            g_d_work, g_lwork,
            g_d_rwork,
            g_d_info));
    }

    int total = N * g_m * g_n;
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;

    reconstruct_kernel<<<blocks, threads>>>(
        g_d_U, g_d_S, g_d_VT, g_d_output,
        N, g_m, g_n, g_mn, rank);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    return 0;
}

extern "C" int mssa_svd_download(double* output_host, int N)
{
    CHECK_CUDA(cudaMemcpy(output_host, g_d_output,
                          (size_t)N * g_mat_bytes, cudaMemcpyDeviceToHost));
    return 0;
}
