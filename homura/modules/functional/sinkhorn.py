# based on Thomas Viehmann's implementation
# https://github.com/t-vi/pytorch-tvmisc/blob/master/wasserstein-distance/Pytorch_Wasserstein.ipynb
# Note that original `dest` is renamed to `cost`

cuda_source = """

#include <torch/extension.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>

using at::RestrictPtrTraits;
using at::PackedTensorAccessor;

#if defined(__HIP_PLATFORM_HCC__)
constexpr int WARP_SIZE = 64;
#else
constexpr int WARP_SIZE = 32;
#endif

// The maximum number of threads in a block
#if defined(__HIP_PLATFORM_HCC__)
constexpr int MAX_BLOCK_SIZE = 256;
#else
constexpr int MAX_BLOCK_SIZE = 512;
#endif

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int getMSB(int val) {
  return 31 - __clz(val);
}

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static int getNumThreads(int nElem) {
#if defined(__HIP_PLATFORM_HCC__)
  int threadSizes[5] = { 16, 32, 64, 128, MAX_BLOCK_SIZE };
#else
  int threadSizes[5] = { 32, 64, 128, 256, MAX_BLOCK_SIZE };
#endif
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}


template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}

// While this might be the most efficient sinkhorn step / logsumexp-matmul implementation I have seen,
// this is awfully inefficient compared to matrix multiplication and e.g. NVidia cutlass may provide
// many great ideas for improvement
template <typename scalar_t, typename index_t>
__global__ void sinkstep_kernel(
  // compute log v_bj = log nu_bj - logsumexp_i 1/lambda cost_ij - log u_bi
  // for this compute maxdiff_bj = max_i(1/lambda cost_ij - log u_bi)
  // i = reduction dim, using threadIdx.x
  PackedTensorAccessor<scalar_t, 2, RestrictPtrTraits, index_t> log_v,
  const PackedTensorAccessor<scalar_t, 2, RestrictPtrTraits, index_t> cost,
  const PackedTensorAccessor<scalar_t, 2, RestrictPtrTraits, index_t> log_nu,
  const PackedTensorAccessor<scalar_t, 2, RestrictPtrTraits, index_t> log_u,
  const scalar_t lambda) {

  using accscalar_t = scalar_t;

  __shared__ accscalar_t shared_mem[2 * WARP_SIZE];

  index_t b = blockIdx.y;
  index_t j = blockIdx.x;
  int tid = threadIdx.x;

  if (b >= log_u.size(0) || j >= log_v.size(1)) {
    return;
  }
  // reduce within thread
  accscalar_t max = -std::numeric_limits<accscalar_t>::infinity();
  accscalar_t sumexp = 0;

  if (log_nu[b][j] == -std::numeric_limits<accscalar_t>::infinity()) {
    if (tid == 0) {
      log_v[b][j] = -std::numeric_limits<accscalar_t>::infinity();
    }
    return;
  }

  for (index_t i = threadIdx.x; i < log_u.size(1); i += blockDim.x) {
    accscalar_t oldmax = max;
    accscalar_t value = -cost[i][j]/lambda + log_u[b][i];
    max = max > value ? max : value;
    if (oldmax == -std::numeric_limits<accscalar_t>::infinity()) {
      // sumexp used to be 0, so the new max is value and we can set 1 here,
      // because we will come back here again
      sumexp = 1;
    } else {
      sumexp *= exp(oldmax - max);
      sumexp += exp(value - max); // if oldmax was not -infinity, max is not either...
    }
  }

  // now we have one value per thread. we'll make it into one value per warp
  // first warpSum to get one value per thread to
  // one value per warp
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    accscalar_t o_max    = WARP_SHFL_XOR(max, 1 << i, WARP_SIZE);
    accscalar_t o_sumexp = WARP_SHFL_XOR(sumexp, 1 << i, WARP_SIZE);
    if (o_max > max) { // we're less concerned about divergence here
      sumexp *= exp(max - o_max);
      sumexp += o_sumexp;
      max = o_max;
    } else if (max != -std::numeric_limits<accscalar_t>::infinity()) {
      sumexp += o_sumexp * exp(o_max - max);
    }
  }

  __syncthreads();
  // this writes each warps accumulation into shared memory
  // there are at most WARP_SIZE items left because
  // there are at most WARP_SIZE**2 threads at the beginning
  if (tid % WARP_SIZE == 0) {
    shared_mem[tid / WARP_SIZE * 2] = max;
    shared_mem[tid / WARP_SIZE * 2 + 1] = sumexp;
  }
  __syncthreads();
  if (tid < WARP_SIZE) {
    max = (tid < blockDim.x / WARP_SIZE ? shared_mem[2 * tid] : -std::numeric_limits<accscalar_t>::infinity());
    sumexp = (tid < blockDim.x / WARP_SIZE ? shared_mem[2 * tid + 1] : 0);
  }
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    accscalar_t o_max    = WARP_SHFL_XOR(max, 1 << i, WARP_SIZE);
    accscalar_t o_sumexp = WARP_SHFL_XOR(sumexp, 1 << i, WARP_SIZE);
    if (o_max > max) { // we're less concerned about divergence here
      sumexp *= exp(max - o_max);
      sumexp += o_sumexp;
      max = o_max;
    } else if (max != -std::numeric_limits<accscalar_t>::infinity()) {
      sumexp += o_sumexp * exp(o_max - max);
    }
  }

  if (tid == 0) {
    log_v[b][j] = (max > -std::numeric_limits<accscalar_t>::infinity() ?
                   log_nu[b][j] - log(sumexp) - max : 
                   -std::numeric_limits<accscalar_t>::infinity());
  }
}

template <typename scalar_t>
torch::Tensor sinkstep_cuda_template(const torch::Tensor& cost, const torch::Tensor& log_nu, const torch::Tensor& log_u,
                                     const double lambda) {
  TORCH_CHECK(cost.is_cuda(), "need cuda tensors");
  TORCH_CHECK(cost.device() == log_nu.device() && cost.device() == log_u.device(), "need tensors on same GPU");
  TORCH_CHECK(cost.dim()==2 && log_nu.dim()==2 && log_u.dim()==2, "invalid sizes");
  TORCH_CHECK(cost.size(0) == log_u.size(1) &&
           cost.size(1) == log_nu.size(1) &&
           log_u.size(0) == log_nu.size(0), "invalid sizes");
  auto log_v = torch::empty_like(log_nu);
  using index_t = int32_t;

  auto log_v_a = log_v.packed_accessor<scalar_t, 2, RestrictPtrTraits, index_t>();
  auto cost_a = cost.packed_accessor<scalar_t, 2, RestrictPtrTraits, index_t>();
  auto log_nu_a = log_nu.packed_accessor<scalar_t, 2, RestrictPtrTraits, index_t>();
  auto log_u_a = log_u.packed_accessor<scalar_t, 2, RestrictPtrTraits, index_t>();

  auto stream = at::cuda::getCurrentCUDAStream();

  int tf = getNumThreads(log_u.size(1));
  dim3 blocks(log_v.size(1), log_u.size(0));
  dim3 threads(tf);

  sinkstep_kernel<<<blocks, threads, 2*WARP_SIZE*sizeof(scalar_t), stream>>>(
    log_v_a, cost_a, log_nu_a, log_u_a, static_cast<scalar_t>(lambda)
    );

  return log_v;
}

torch::Tensor sinkstep_cuda(const torch::Tensor& cost, const torch::Tensor& log_nu, const torch::Tensor& log_u,
                            const double lambda) {
    return AT_DISPATCH_FLOATING_TYPES(log_u.scalar_type(), "sinkstep", [&] {
       return sinkstep_cuda_template<scalar_t>(cost, log_nu, log_u, lambda);
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sinkstep", &sinkstep_cuda, "sinkhorn step");
}

"""

import math

import torch

wasserstein_ext = torch.utils.cpp_extension.load_inline("wasserstein", cpp_sources="", cuda_sources=cuda_source,
                                                        extra_cuda_cflags=["--expt-relaxed-constexpr"])


def _sinkstep(cost: torch.Tensor,
              log_nu: torch.Tensor,
              log_u: torch.Tensor,
              lam: float):
    # dispatch to optimized GPU implementation for GPU tensors, slow fallback for CPU
    if cost.is_cuda:
        return wasserstein_ext.sinkstep(cost, log_nu, log_u, lam)

    # CPU case
    assert cost.dim() == 2 and log_nu.dim() == 2 and log_u.dim() == 2
    assert cost.size(0) == log_u.size(1) and cost.size(1) == log_nu.size(1) and log_u.size(0) == log_nu.size(0)
    log_v = log_nu.clone()
    for b in range(log_u.size(0)):
        log_v[b] -= torch.logsumexp(-cost / lam + log_u[b, :, None], 1)
    return log_v


class _Sinkhorn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                source: torch.Tensor,
                dest: torch.Tensor,
                cost: torch.Tensor,
                lam: float = 1e-3,
                num_iteration: int = 100):
        assert source.dim() == 2 and dest.dim() == 2 and cost.dim() == 2
        bs = source.size(0)
        d1, d2 = cost.size()
        assert dest.size(0) == bs and source.size(1) == d1 and dest.size(1) == d2
        log_mu = source.log()
        log_nu = dest.log()
        log_u = torch.full_like(source, -math.log(d1))
        log_v = torch.full_like(dest, -math.log(d2))
        for i in range(num_iteration):
            log_v = _sinkstep(cost, log_nu, log_u, lam)
            log_u = _sinkstep(cost.t(), log_mu, log_v, lam)

        # this is slight abuse of the function. it computes (diag(exp(log_u))*Mt*exp(-Mt/gamma)*diag(exp(log_v))).sum()
        # in an efficient (i.e. no bxnxm tensors) way in log space
        distances = (-_sinkstep(-cost.log() + cost / lam, -log_v, log_u, 1.0)).logsumexp(1).exp()
        ctx.log_v = log_v
        ctx.log_u = log_u
        ctx.dist = cost
        ctx.lam = lam
        return distances

    @staticmethod
    def backward(ctx,
                 grad_out: torch.Tensor):
        return grad_out[:, None] * ctx.log_u * ctx.lam, grad_out[:, None] * ctx.log_v * ctx.lam, None, None, None


def sinkhorn(source: torch.Tensor,
             dest: torch.Tensor,
             cost: torch.Tensor,
             lam: float = 1e-3,
             num_iteration: int = 100):
    """ Batched sinkhorn iteration from https://arxiv.org/abs/1907.01729.

    :param source: The source histograms of `B x d_1`
    :param dest: The destination histograms of `B x d_2`
    :param cost: The cost matrix of `d_1 x d_2`
    :param lam:　The coefficient of the entropy term
    :param num_iteration: The number of iteration of Sinkhorn-Knopp iteration
    :return: distances
    """

    return _Sinkhorn.apply(source, dest, cost, lam, num_iteration)


def get_coupling(source: torch.Tensor,
                 dest: torch.Tensor,
                 cost: torch.Tensor,
                 lam: float = 1e-3,
                 num_iteration: int = 1000):
    assert source.dim() == 2 and dest.dim() == 2 and cost.dim() == 2
    bs = source.size(0)
    d1, d2 = cost.size()
    assert dest.size(0) == bs and source.size(1) == d1 and dest.size(1) == d2
    log_mu = source.log()
    log_nu = dest.log()
    log_u = torch.full_like(source, -math.log(d1))
    log_v = torch.full_like(dest, -math.log(d2))
    for i in range(num_iteration):
        log_v = _sinkstep(cost, log_nu, log_u, lam)
        log_u = _sinkstep(cost.t(), log_mu, log_v, lam)
    return (log_v[:, None, :] - cost / lam + log_u[:, :, None]).exp(z)
