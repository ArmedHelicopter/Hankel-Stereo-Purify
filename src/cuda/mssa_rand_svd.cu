/* Randomized SVD + rank-k reconstruction via cuBLAS + cuSOLVER.

cusolverDnDgesvd fails for wide matrices (m < n) on some GPUs.
Workaround: transpose B before SVD.

For each matrix A ∈ R^{m×n} (column-major):
  1. Y = A @ Ω             cuBLAS dgemm
  2. Q = orth(Y)           cuSOLVER geqrf + orgqr
  3. B = Q^T @ A           cuBLAS dgemm  (kp × n)
  4. Bt = B^T              cuBLAS dgeam  (n × kp, tall)
  5. SVD(Bt) = U' σ Vt'   cuSOLVER gesvd (n × kp)
  6. Uhat = Vt'^T, VT = U'^T   cuBLAS dgeam (transpose)
  7. U_final = Q @ Uhat[:,:k]  cuBLAS dgemm
  8. out = (U * σ[:k]) @ VT[:k,:]  custom kernel

Build:
  cd src/cuda && /usr/local/cuda-12.8/bin/nvcc -shared -Xcompiler -fPIC \
    -o libmssa_rand_svd.so mssa_rand_svd.cu -lcudart -lcusolver -lcublas
*/

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <curand.h>
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

#define CHECK_CUBLAS(call)                                                  \
    do {                                                                    \
        cublasStatus_t err = (call);                                        \
        if (err != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error at %s:%d: code %d\n",            \
                    __FILE__, __LINE__, (int)err);                          \
            return -3;                                                      \
        }                                                                   \
    } while (0)

#define CHECK_CURAND(call)                                                  \
    do {                                                                    \
        curandStatus_t err = (call);                                        \
        if (err != CURAND_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuRAND error at %s:%d: code %d\n",            \
                    __FILE__, __LINE__, (int)err);                          \
            return -4;                                                      \
        }                                                                   \
    } while (0)


/* Reconstruct: out[c*m+r] = sum_{j<k} σ[j] * U[j*m+r] * VT[c*ldvt+j] */
__global__ void reconstruct_rsvd(
    const double* __restrict__ U,    /* m × k  col-major */
    const double* __restrict__ S,    /* k */
    const double* __restrict__ VT,   /* ldvt × n  col-major, first k rows used */
    double* __restrict__ out,        /* m × n  col-major */
    int m, int n, int k, int ldvt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * n;
    if (idx >= total) return;
    int c = idx / m;
    int r = idx % m;
    double val = 0.0;
    for (int j = 0; j < k; j++) {
        val += S[j] * U[(long long)j * m + r] * VT[(long long)c * ldvt + j];
    }
    out[idx] = val;
}


/* Persistent context */
static cusolverDnHandle_t g_solver = NULL;
static cublasHandle_t    g_blas   = NULL;
static curandGenerator_t g_rng    = NULL;

static int g_m = 0, g_n = 0, g_k = 0, g_p = 0, g_kp = 0;

/* Core working buffers */
static double *g_A_work  = NULL;  /* m × n */
static double *g_Omega   = NULL;  /* n × kp */
static double *g_Y       = NULL;  /* m × kp (also Q) */
static double *g_tau     = NULL;  /* kp */
static double *g_B       = NULL;  /* kp × n */
static double *g_Bt      = NULL;  /* n × kp (transposed B) */
static double *g_S       = NULL;  /* kp */
static double *g_Uprime  = NULL;  /* n × kp (from SVD of Bt) */
static double *g_Vtprime = NULL;  /* kp × kp */
static double *g_Uhat    = NULL;  /* kp × kp = Vt'^T */
static double *g_VT      = NULL;  /* kp × n = U'^T */
static double *g_U       = NULL;  /* m × k */
static double *g_output  = NULL;  /* m × n */

static double *g_work    = NULL;
static int     g_lwork   = 0;
static double *g_rwork   = NULL;
static int    *g_info    = NULL;

static double *g_A_batch  = NULL;
static double *g_out_batch = NULL;

extern "C" void mssa_rand_svd_cleanup(void);

extern "C" int mssa_rand_svd_init(int m, int n, int rank, int oversample)
{
    if (g_solver != NULL && g_m == m && g_n == n && g_k == rank && g_p == oversample)
        return 0;

    mssa_rand_svd_cleanup();

    g_m = m;  g_n = n;  g_k = rank;  g_p = oversample;
    g_kp = rank + oversample;

    CHECK_CUSOLVER(cusolverDnCreate(&g_solver));
    CHECK_CUBLAS(cublasCreate(&g_blas));
    CHECK_CURAND(curandCreateGenerator(&g_rng, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(g_rng, 42ULL));

    /* Workspace queries */
    int lwork_qr = 0, lwork_org = 0, lwork_svd = 0;
    {
        double *da = NULL, *db = NULL, *dt = NULL;
        CHECK_CUDA(cudaMalloc(&da, (size_t)m * g_kp * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&db, (size_t)g_kp * n * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&dt, (size_t)g_kp * sizeof(double)));
        CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(g_solver, m, g_kp, da, m, &lwork_qr));
        CHECK_CUSOLVER(cusolverDnDorgqr_bufferSize(g_solver, m, g_kp, g_kp, da, m, dt, &lwork_org));
        /* SVD on Bt: n × kp (tall) */
        CHECK_CUSOLVER(cusolverDnDgesvd_bufferSize(g_solver, n, g_kp, &lwork_svd));
        cudaFree(da); cudaFree(db); cudaFree(dt);
    }
    g_lwork = lwork_qr;
    if (lwork_org > g_lwork) g_lwork = lwork_org;
    if (lwork_svd > g_lwork) g_lwork = lwork_svd;

    /* Allocate */
    CHECK_CUDA(cudaMalloc(&g_A_work,  (size_t)m * n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&g_Omega,   (size_t)n * g_kp * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&g_Y,       (size_t)m * g_kp * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&g_tau,     (size_t)g_kp * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&g_B,       (size_t)g_kp * n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&g_Bt,      (size_t)n * g_kp * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&g_S,       (size_t)g_kp * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&g_Uprime,  (size_t)n * g_kp * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&g_Vtprime, (size_t)g_kp * g_kp * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&g_Uhat,    (size_t)g_kp * g_kp * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&g_VT,      (size_t)g_kp * n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&g_U,       (size_t)m * g_k * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&g_output,  (size_t)m * n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&g_work,    (size_t)g_lwork * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&g_rwork,   (size_t)g_kp * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&g_info,    sizeof(int)));

    /* Random projection Ω on GPU */
    CHECK_CURAND(curandGenerateNormalDouble(g_rng, g_Omega, (size_t)n * g_kp, 0.0, 1.0));

    return 0;
}

extern "C" void mssa_rand_svd_cleanup(void)
{
    if (g_A_work)   { cudaFree(g_A_work);   g_A_work   = NULL; }
    if (g_Omega)    { cudaFree(g_Omega);     g_Omega    = NULL; }
    if (g_Y)        { cudaFree(g_Y);         g_Y        = NULL; }
    if (g_tau)      { cudaFree(g_tau);       g_tau      = NULL; }
    if (g_B)        { cudaFree(g_B);         g_B        = NULL; }
    if (g_Bt)       { cudaFree(g_Bt);        g_Bt       = NULL; }
    if (g_S)        { cudaFree(g_S);         g_S        = NULL; }
    if (g_Uprime)   { cudaFree(g_Uprime);    g_Uprime   = NULL; }
    if (g_Vtprime)  { cudaFree(g_Vtprime);   g_Vtprime  = NULL; }
    if (g_Uhat)     { cudaFree(g_Uhat);      g_Uhat     = NULL; }
    if (g_VT)       { cudaFree(g_VT);        g_VT       = NULL; }
    if (g_U)        { cudaFree(g_U);         g_U        = NULL; }
    if (g_output)   { cudaFree(g_output);    g_output   = NULL; }
    if (g_work)     { cudaFree(g_work);      g_work     = NULL; }
    if (g_rwork)    { cudaFree(g_rwork);     g_rwork    = NULL; }
    if (g_info)     { cudaFree(g_info);      g_info     = NULL; }
    if (g_A_batch)  { cudaFree(g_A_batch);   g_A_batch  = NULL; }
    if (g_out_batch){ cudaFree(g_out_batch); g_out_batch = NULL; }
    if (g_rng)      { curandDestroyGenerator(g_rng); g_rng = NULL; }
    if (g_blas)     { cublasDestroy(g_blas); g_blas = NULL; }
    if (g_solver)   { cusolverDnDestroy(g_solver); g_solver = NULL; }
    g_m = g_n = g_k = g_p = g_kp = 0;
    g_lwork = 0;
}


static int _process_one(void)
{
    double alpha = 1.0, beta = 0.0;
    double one = 1.0, zero = 0.0;
    int info_host = 0;

    /* Step 1: Y = A @ Ω   (m×n) @ (n×kp) = (m×kp) */
    CHECK_CUBLAS(cublasDgemm(g_blas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        g_m, g_kp, g_n,
        &alpha,
        g_A_work, g_m,
        g_Omega,  g_n,
        &beta,
        g_Y,      g_m));

    /* Step 2: Q = orth(Y) via QR   (m×kp thin QR) */
    CHECK_CUSOLVER(cusolverDnDgeqrf(g_solver,
        g_m, g_kp, g_Y, g_m, g_tau,
        g_work, g_lwork, g_info));
    CHECK_CUDA(cudaMemcpy(&info_host, g_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_host != 0) return -10;

    CHECK_CUSOLVER(cusolverDnDorgqr(g_solver,
        g_m, g_kp, g_kp, g_Y, g_m, g_tau,
        g_work, g_lwork, g_info));
    CHECK_CUDA(cudaMemcpy(&info_host, g_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_host != 0) return -11;
    /* g_Y now holds Q (m × kp) */

    /* Step 3: B = Q^T @ A   (kp×m) @ (m×n) = (kp×n) */
    CHECK_CUBLAS(cublasDgemm(g_blas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        g_kp, g_n, g_m,
        &alpha,
        g_Y,   g_m,
        g_A_work, g_m,
        &beta,
        g_B,   g_kp));

    /* Step 4: Bt = B^T   (kp×n) → (n×kp) for cusolver compatibility */
    CHECK_CUBLAS(cublasDgeam(g_blas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        g_n, g_kp,
        &one, g_B, g_kp,
        &zero, (const double*)0, g_n,
        g_Bt, g_n));
    /* g_Bt: n × kp column-major */

    /* Step 5: SVD(Bt) = U' σ Vt'   (n×kp tall) → U'(n×kp), σ(kp), Vt'(kp×kp) */
    CHECK_CUSOLVER(cusolverDnDgesvd(g_solver,
        'S', 'S',
        g_n, g_kp,
        g_Bt, g_n,
        g_S,
        g_Uprime, g_n,      /* U': n × kp (with 'S': n × min(n,kp) = n × kp) */
        g_Vtprime, g_kp,    /* Vt': kp × kp (with 'S': kp × kp) */
        g_work, g_lwork,
        g_rwork,
        g_info));
    CHECK_CUDA(cudaMemcpy(&info_host, g_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_host != 0) return -12;

    /*
     * From SVD(Bt) = U' σ Vt':
     *   B = Vt'^T σ U'^T
     *   So: Uhat (for B) = Vt'^T  (kp × kp)
     *       VT (for B) = U'^T     (kp × n)
     *
     * Vt' is kp×kp col-major. Vt'^T = g_Vtprime transposed → store in g_Uhat.
     * U' is n×kp col-major. U'^T = g_Uprime transposed → store in g_VT.
     */

    /* Step 6a: Uhat = Vt'^T   (kp×kp) → dgeam with transa=T */
    CHECK_CUBLAS(cublasDgeam(g_blas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        g_kp, g_kp,
        &one, g_Vtprime, g_kp,
        &zero, (const double*)0, g_kp,
        g_Uhat, g_kp));

    /* Step 6b: VT = U'^T   (kp×n) → dgeam with transa=T */
    CHECK_CUBLAS(cublasDgeam(g_blas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        g_kp, g_n,
        &one, g_Uprime, g_n,
        &zero, (const double*)0, g_kp,
        g_VT, g_kp));
    /* g_VT: kp × n column-major, ldvt = kp */

    /* Step 7: U_final = Q @ Uhat[:,:k]   (m×kp) @ (kp×k) = (m×k) */
    CHECK_CUBLAS(cublasDgemm(g_blas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        g_m, g_k, g_kp,
        &alpha,
        g_Y,    g_m,     /* Q: m × kp */
        g_Uhat, g_kp,    /* Uhat: kp × kp, first k cols used */
        &beta,
        g_U,    g_m));   /* U: m × k */

    /* Step 8: Reconstruct   out = (U σ[:k]) @ VT[:k,:] */
    {
        int total = g_m * g_n;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        reconstruct_rsvd<<<blocks, threads>>>(
            g_U, g_S, g_VT, g_output, g_m, g_n, g_k, g_kp);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    return 0;
}


extern "C" int mssa_rand_svd_upload(const double* input_host, int N)
{
    size_t bytes = (size_t)N * g_m * g_n * sizeof(double);
    if (g_A_batch == NULL) {
        CHECK_CUDA(cudaMalloc(&g_A_batch, bytes));
    }
    CHECK_CUDA(cudaMemcpy(g_A_batch, input_host, bytes, cudaMemcpyHostToDevice));
    return 0;
}

extern "C" int mssa_rand_svd_run(int N)
{
    size_t mat_bytes = (size_t)g_m * g_n * sizeof(double);
    if (g_out_batch == NULL) {
        CHECK_CUDA(cudaMalloc(&g_out_batch, (size_t)N * mat_bytes));
    }
    int rc;
    for (int i = 0; i < N; i++) {
        CHECK_CUDA(cudaMemcpy(g_A_work, g_A_batch + (long long)i * g_m * g_n,
                              mat_bytes, cudaMemcpyDeviceToDevice));
        rc = _process_one();
        if (rc != 0) return rc;
        CHECK_CUDA(cudaMemcpy(g_out_batch + (long long)i * g_m * g_n,
                              g_output, mat_bytes, cudaMemcpyDeviceToDevice));
    }
    return 0;
}

extern "C" int mssa_rand_svd_download(double* output_host, int N)
{
    size_t bytes = (size_t)N * g_m * g_n * sizeof(double);
    CHECK_CUDA(cudaMemcpy(output_host, g_out_batch, bytes, cudaMemcpyDeviceToHost));
    return 0;
}
