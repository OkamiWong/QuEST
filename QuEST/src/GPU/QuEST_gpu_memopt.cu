#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <vector>

#include "../../../dependencies/optimize-cuda-memory-usage-v1/public/memopt.hpp"
#include "QuEST.h"
#include "QuEST_gpu_common.h"
#include "QuEST_internal.h"
#include "QuEST_precision.h"
#include "QuEST_validation.h"
#include "mt19937ar.h"

// General utilities
template <typename T>
void __check(T result, char const* const func, const char* const file, int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line, static_cast<unsigned int>(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) __check((val), #val, __FILE__, __LINE__)

typedef long long StateVecIndex_t;

// Memopt wrappers
namespace memopt_adapter {

template <typename T>
void allocateShardAndRegister(T** p, size_t s) {
  if (memopt::ConfigurationManager::getConfig().generic.useUM) {
    checkCudaErrors(cudaMallocManaged(p, s));
  } else {
    checkCudaErrors(cudaMalloc(p, s));
  }
  memopt::registerManagedMemoryAddress(*p, s);
  memopt::registerApplicationInput(*p);
  memopt::registerApplicationOutput(*p);
}

typedef std::function<void(Qureg, cudaStream_t)> Task;

std::vector<Task> tasks;

void registerAndExecuteTask(
  std::vector<StateVecIndex_t> ioShardIndices,
  Task task,
  Qureg qureg,
  cudaStream_t stream
) {
  auto taskId = tasks.size();
  tasks.push_back(task);

  std::vector<void*> inputs, outputs;
  for (auto i : ioShardIndices) {
    inputs.push_back(qureg.deviceStateVecShards[i].real);
    outputs.push_back(qureg.deviceStateVecShards[i].real);
  }

  memopt::annotateNextTask(taskId, inputs, outputs, stream);
  task(qureg, stream);
}

template <typename T>
void tryUpdatingAddress(T*& oldAddress, const std::map<void*, void*>& addressUpdateMap) {
  if (addressUpdateMap.count(oldAddress) > 0) {
    oldAddress = (T*)addressUpdateMap.at(oldAddress);
  }
}

void executeRandomTask(Qureg qureg, int taskId, std::map<void*, void*> addressUpdateMap, cudaStream_t stream) {
  for (StateVecIndex_t i = 0; i < qureg.numShards; i++) {
    tryUpdatingAddress(qureg.deviceStateVecShards[i].real, addressUpdateMap);
    qureg.deviceStateVecShards[i].imag = qureg.deviceStateVecShards[i].real + qureg.numAmpsPerShard;
  }
  tasks[taskId](qureg, stream);
}

template <typename T>
void moveDataBackToDevice(T*& oldAddress, const std::map<void*, void*>& managedDeviceArrayToHostArrayMap, size_t numAmpsPerShard) {
  auto newAddress = managedDeviceArrayToHostArrayMap.at(oldAddress);
  checkCudaErrors(cudaMalloc(&oldAddress, 2 * numAmpsPerShard * sizeof(qreal)));
  checkCudaErrors(cudaMemcpy(oldAddress, newAddress, 2 * numAmpsPerShard * sizeof(qreal), cudaMemcpyDefault));
  if (memopt::ConfigurationManager::getConfig().execution.useNvlink) {
    checkCudaErrors(cudaFree(newAddress));
  } else {
    checkCudaErrors(cudaFreeHost(newAddress));
  }
}

}  // namespace memopt_adapter

// QuEST speicifc definitions
constexpr int MAX_NUM_QUBITS = 64;
constexpr int MAX_NUM_PHASE_FUNC_OVERRIDES = 8;
constexpr int NUM_THREADS_PER_BLOCK = 128;

struct KernelParamQureg {
  long long numAmpsPerShard;
  int numShards;
  int numGlobalBits;
  int numLocalBits;
};

KernelParamQureg convertToKernelParamQureg(const Qureg& qureg) {
  KernelParamQureg kpq;
  kpq.numAmpsPerShard = qureg.numAmpsPerShard;
  kpq.numShards = qureg.numShards;
  kpq.numGlobalBits = qureg.numGlobalBits;
  kpq.numLocalBits = qureg.numLocalBits;
  return kpq;
}

__forceinline__ __device__ int getBit(StateVecIndex_t num, int index) {
  return (num >> index) & 1;
}

__forceinline__ __host__ __device__ int extractBit(const int locationOfBitFromRight, const StateVecIndex_t theEncodedNumber) {
  return (theEncodedNumber & (1LL << locationOfBitFromRight)) >> locationOfBitFromRight;
}

__forceinline__ __host__ __device__ StateVecIndex_t flipBit(const StateVecIndex_t number, const int index) {
  return (number ^ (1LL << index));
}

__forceinline__ __host__ __device__ StateVecIndex_t insertZeroBit(const StateVecIndex_t number, const int index) {
  StateVecIndex_t left, right;
  left = (number >> index) << index;
  right = number - left;
  return (left << 1) ^ right;
}

__forceinline__ __host__ __device__ StateVecIndex_t insertTwoZeroBits(const StateVecIndex_t number, const int bit1, const int bit2) {
  int small = (bit1 < bit2) ? bit1 : bit2;
  int big = (bit1 < bit2) ? bit2 : bit1;
  return insertZeroBit(insertZeroBit(number, small), big);
}

__forceinline__ __host__ __device__ StateVecIndex_t getGlobalIndex(Qureg* qureg, StateVecIndex_t index) {
  return index >> qureg->numLocalBits;
}

__forceinline__ __host__ __device__ StateVecIndex_t getLocalIndex(Qureg* qureg, StateVecIndex_t index) {
  return index & ((1 << qureg->numLocalBits) - 1);
}

__forceinline__ __host__ __device__ void splitIndex(Qureg* qureg, StateVecIndex_t index, StateVecIndex_t* globalIndex, StateVecIndex_t* localIndex) {
  *globalIndex = getGlobalIndex(qureg, index);
  *localIndex = getLocalIndex(qureg, index);
}

qreal statevec_getRealAmp(Qureg qureg, StateVecIndex_t index) {
  StateVecIndex_t globalIndex, localIndex;
  splitIndex(&qureg, index, &globalIndex, &localIndex);

  qreal el = 0;
  cudaMemcpy(
    &el,
    &(qureg.deviceStateVecShards[globalIndex].real[localIndex]),
    sizeof(qreal),
    cudaMemcpyDefault
  );
  return el;
}

qreal statevec_getImagAmp(Qureg qureg, StateVecIndex_t index) {
  StateVecIndex_t globalIndex, localIndex;
  splitIndex(&qureg, index, &globalIndex, &localIndex);

  qreal el = 0;
  cudaMemcpy(
    &el,
    &(qureg.deviceStateVecShards[globalIndex].imag[localIndex]),
    sizeof(qreal),
    cudaMemcpyDefault
  );
  return el;
}

void statevec_initZeroState(Qureg qureg) {
  // Set all amps to zero
  for (StateVecIndex_t i = 0; i < qureg.numShards; i++) {
    checkCudaErrors(cudaMemset(qureg.deviceStateVecShards[i].real, 0, 2 * qureg.numAmpsPerShard * sizeof(qreal)));
  }

  // Set |000...000>'s amp to 1
  qreal one = 1, zero = 0;
  checkCudaErrors(cudaMemcpy(&qureg.deviceStateVecShards[0].real[0], &one, sizeof(qreal), cudaMemcpyDefault));
  checkCudaErrors(cudaMemcpy(&qureg.deviceStateVecShards[0].imag[0], &zero, sizeof(qreal), cudaMemcpyDefault));
}

__global__ void statevec_hadamardLocalBitKernel(
  const __grid_constant__ KernelParamQureg qureg,
  const __grid_constant__ ComplexArray stateVecShard,
  int targetQubit
) {
  const StateVecIndex_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const StateVecIndex_t numTasks = qureg.numAmpsPerShard >> 1;
  if (idx >= numTasks) return;

  StateVecIndex_t indexUp = insertZeroBit(idx, targetQubit);
  StateVecIndex_t indexLo = flipBit(indexUp, targetQubit);

  qreal* stateVecReal = stateVecShard.real;
  qreal* stateVecImag = stateVecShard.imag;

  qreal *stateRealUp, *stateRealLo, *stateImagUp, *stateImagLo;
  stateRealUp = &stateVecReal[indexUp];
  stateImagUp = &stateVecImag[indexUp];
  stateRealLo = &stateVecReal[indexLo];
  stateImagLo = &stateVecImag[indexLo];

  qreal stateRealUpValue, stateRealLoValue, stateImagUpValue, stateImagLoValue;
  stateRealUpValue = *stateRealUp;
  stateImagUpValue = *stateImagUp;
  stateRealLoValue = *stateRealLo;
  stateImagLoValue = *stateImagLo;

  qreal factor = 1.0 / sqrt(2.0);

  *stateRealUp = factor * (stateRealUpValue + stateRealLoValue);
  *stateImagUp = factor * (stateImagUpValue + stateImagLoValue);
  *stateRealLo = factor * (stateRealUpValue - stateRealLoValue);
  *stateImagLo = factor * (stateImagUpValue - stateImagLoValue);
}

__global__ void statevec_hadamardGlobalBitKernel(
  const __grid_constant__ KernelParamQureg qureg,
  const __grid_constant__ ComplexArray stateVecShardUp,
  const __grid_constant__ ComplexArray stateVecShardLo,
  int targetQubit
) {
  const StateVecIndex_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= qureg.numAmpsPerShard) return;

  qreal *stateRealUp, *stateRealLo, *stateImagUp, *stateImagLo;
  stateRealUp = &stateVecShardUp.real[idx];
  stateImagUp = &stateVecShardUp.imag[idx];
  stateRealLo = &stateVecShardLo.real[idx];
  stateImagLo = &stateVecShardLo.imag[idx];

  qreal stateRealUpValue, stateRealLoValue, stateImagUpValue, stateImagLoValue;
  stateRealUpValue = *stateRealUp;
  stateImagUpValue = *stateImagUp;
  stateRealLoValue = *stateRealLo;
  stateImagLoValue = *stateImagLo;

  qreal factor = 1.0 / sqrt(2.0);

  *stateRealUp = factor * (stateRealUpValue + stateRealLoValue);
  *stateImagUp = factor * (stateImagUpValue + stateImagLoValue);
  *stateRealLo = factor * (stateRealUpValue - stateRealLoValue);
  *stateImagLo = factor * (stateImagUpValue - stateImagLoValue);
}

void memopt_statevec_hadamard(cudaStream_t stream, Qureg qureg, int targetQubit) {
  if (targetQubit < qureg.numLocalBits) {
    StateVecIndex_t numThreadsPerBlock, numBlocks;
    numThreadsPerBlock = NUM_THREADS_PER_BLOCK;
    numBlocks = ((qureg.numAmpsPerShard >> 1) + numThreadsPerBlock - 1) / numThreadsPerBlock;

    for (StateVecIndex_t i = 0; i < qureg.numShards; i++) {
      memopt_adapter::Task task = [=](Qureg q, cudaStream_t s) {
        statevec_hadamardLocalBitKernel<<<numBlocks, numThreadsPerBlock, 0, s>>>(
          convertToKernelParamQureg(q),
          q.deviceStateVecShards[i],
          targetQubit
        );
      };
      memopt_adapter::registerAndExecuteTask(
        {i},
        task,
        qureg,
        stream
      );
    }
  } else {
    StateVecIndex_t numThreadsPerBlock, numBlocks;
    numThreadsPerBlock = NUM_THREADS_PER_BLOCK;
    numBlocks = (qureg.numAmpsPerShard + numThreadsPerBlock - 1) / numThreadsPerBlock;

    for (StateVecIndex_t i = 0; i < (qureg.numShards >> 1); i++) {
      StateVecIndex_t globalIndexUp = insertZeroBit(i, targetQubit - qureg.numLocalBits);
      StateVecIndex_t globalIndexLo = flipBit(globalIndexUp, targetQubit - qureg.numLocalBits);
      memopt_adapter::Task task = [=](Qureg q, cudaStream_t s) {
        statevec_hadamardGlobalBitKernel<<<numBlocks, numThreadsPerBlock, 0, s>>>(
          convertToKernelParamQureg(q),
          q.deviceStateVecShards[globalIndexUp],
          q.deviceStateVecShards[globalIndexLo],
          targetQubit
        );
      };
      memopt_adapter::registerAndExecuteTask(
        {globalIndexUp, globalIndexLo},
        task,
        qureg,
        stream
      );
    }
  }

  memopt::endStage(stream);
}

__forceinline__ __device__ void setMultiRegPhaseInds(
  const KernelParamQureg* qureg,
  StateVecIndex_t* phaseInds, StateVecIndex_t fullIndex,
  const int* qubits, const int* numQubitsPerReg, int numRegs, enum bitEncoding encoding
) {
  size_t stride = blockDim.x;
  size_t offset = threadIdx.x;

  if (encoding == UNSIGNED) {
    int flatIndex = 0;
    for (int r = 0; r < numRegs; r++) {
      phaseInds[r * stride + offset] = 0LL;
      for (int q = 0; q < numQubitsPerReg[r]; q++)
        phaseInds[r * stride + offset] += (1LL << q) * getBit(fullIndex, qubits[flatIndex++]);
    }
  } else if (encoding == TWOS_COMPLEMENT) {
    // Not implemented
    __trap();
  }
}

__forceinline__ __device__ StateVecIndex_t getIndOfMultiRegPhaseOverride(
  const KernelParamQureg* qureg,
  StateVecIndex_t fullIndex,
  StateVecIndex_t* phaseInds, int numRegs,
  const StateVecIndex_t* overrideInds, int numOverrides
) {
  size_t stride = blockDim.x;
  size_t offset = threadIdx.x;

  int i;
  for (i = 0; i < numOverrides; i++) {
    int found = 1;
    for (int r = 0; r < numRegs; r++) {
      if (phaseInds[r * stride + offset] != overrideInds[i * numRegs + r]) {
        found = 0;
        break;
      }
    }

    if (found)
      break;
  }

  return i;
}

__forceinline__ __device__ qreal evalNormPhaseFunc(
  StateVecIndex_t* phaseInds, size_t stride, size_t offset,
  int numRegs, enum phaseFunc phaseFuncName, const qreal* params, int numParams
) {
  // Not implemented
  __trap();
}

__forceinline__ __device__ qreal evalProductPhaseFunc(
  StateVecIndex_t* phaseInds, size_t stride, size_t offset,
  int numRegs, enum phaseFunc phaseFuncName, const qreal* params, int numParams
) {
  // determine product of phase indices
  qreal prod = 1;
  for (int r = 0; r < numRegs; r++)
    prod *= phaseInds[r * stride + offset];

  // determine phase via phase function
  if (phaseFuncName == PRODUCT)
    return prod;

  if (phaseFuncName == INVERSE_PRODUCT)
    return (prod == 0.) ? params[0] : 1 / prod;  // smallest non-zero prod is +- 1

  if (phaseFuncName == SCALED_PRODUCT)
    return params[0] * prod;

  if (phaseFuncName == SCALED_INVERSE_PRODUCT)
    return (prod == 0.) ? params[1] : params[0] / prod;
}

__forceinline__ __device__ qreal evalDistancePhaseFunc(
  StateVecIndex_t* phaseInds, size_t stride, size_t offset,
  int numRegs, enum phaseFunc phaseFuncName, const qreal* params, int numParams
) {
  // Not implemented
  __trap();
}

__forceinline__ __device__ qreal getPhaseFromParamNamedFunc(
  const KernelParamQureg* qureg,
  StateVecIndex_t fullIndex,
  StateVecIndex_t* phaseInds, int numRegs,
  enum phaseFunc phaseFuncName, const qreal* params, int numParams
) {
  size_t stride = blockDim.x;
  size_t offset = threadIdx.x;

  if (
    phaseFuncName == NORM
    || phaseFuncName == INVERSE_NORM
    || phaseFuncName == SCALED_NORM
    || phaseFuncName == SCALED_INVERSE_NORM
    || phaseFuncName == SCALED_INVERSE_SHIFTED_NORM
  )
    return evalNormPhaseFunc(phaseInds, stride, offset, numRegs, phaseFuncName, params, numParams);

  if (
    phaseFuncName == PRODUCT
    || phaseFuncName == INVERSE_PRODUCT
    || phaseFuncName == SCALED_PRODUCT
    || phaseFuncName == SCALED_INVERSE_PRODUCT
  )
    return evalProductPhaseFunc(phaseInds, stride, offset, numRegs, phaseFuncName, params, numParams);

  if (
    phaseFuncName == DISTANCE
    || phaseFuncName == INVERSE_DISTANCE
    || phaseFuncName == SCALED_DISTANCE
    || phaseFuncName == SCALED_INVERSE_DISTANCE
    || phaseFuncName == SCALED_INVERSE_SHIFTED_DISTANCE
    || phaseFuncName == SCALED_INVERSE_SHIFTED_WEIGHTED_DISTANCE
  )
    return evalDistancePhaseFunc(phaseInds, stride, offset, numRegs, phaseFuncName, params, numParams);
}

__forceinline__ __device__ void applyPhaseToAmp(
  const ComplexArray* stateVecShard,
  StateVecIndex_t localIndex,
  qreal phase, int conj
) {
  phase *= (1 - 2 * conj);
  qreal c = cos(phase);
  qreal s = sin(phase);

  qreal re, im;
  re = stateVecShard->real[localIndex];
  im = stateVecShard->imag[localIndex];
  stateVecShard->real[localIndex] = re * c - im * s;
  stateVecShard->imag[localIndex] = re * s + im * c;
}

struct ApplyParamNamedPhaseFuncOverridesParams {
  int qubits[MAX_NUM_QUBITS];
  int numQubitsPerReg[MAX_NUM_QUBITS];
  int numRegs;
  enum bitEncoding encoding;
  enum phaseFunc phaseFuncName;
  qreal params[MAX_NUM_QUBITS + 2];
  int numParams;
  StateVecIndex_t overrideInds[MAX_NUM_PHASE_FUNC_OVERRIDES];
  qreal overridePhases[MAX_NUM_PHASE_FUNC_OVERRIDES];
  int numOverrides;
  int conj;
};

__global__ void
statevec_applyParamNamedPhaseFuncOverridesKernel(
  const __grid_constant__ KernelParamQureg qureg,
  const __grid_constant__ ApplyParamNamedPhaseFuncOverridesParams params,
  const __grid_constant__ ComplexArray stateVecShard,
  StateVecIndex_t globalIndex
) {
  extern __shared__ StateVecIndex_t phaseInds[];

  const StateVecIndex_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= qureg.numAmpsPerShard) return;

  StateVecIndex_t fullIndex = idx + (globalIndex << qureg.numLocalBits);

  // Determine phase indices (each thread has phaseInds[numRegs] sub-array)
  setMultiRegPhaseInds(&qureg, phaseInds, fullIndex, params.qubits, params.numQubitsPerReg, params.numRegs, params.encoding);

  // Determine if this phase index has an overriden value
  StateVecIndex_t overrideCaseIndex = getIndOfMultiRegPhaseOverride(&qureg, fullIndex, phaseInds, params.numRegs, params.overrideInds, params.numOverrides);

  // Determine the phase, or the overriden one
  qreal phase = 0;
  if (overrideCaseIndex < params.numOverrides)
    phase = params.overridePhases[overrideCaseIndex];
  else
    phase = getPhaseFromParamNamedFunc(&qureg, fullIndex, phaseInds, params.numRegs, params.phaseFuncName, params.params, params.numParams);

  // Modify amp to amp * exp(i phase)
  applyPhaseToAmp(&stateVecShard, idx, phase, params.conj);
}

void memopt_statevec_applyParamNamedPhaseFuncOverrides(
  cudaStream_t stream,
  Qureg qureg,
  ApplyParamNamedPhaseFuncOverridesParams params
) {
  StateVecIndex_t numThreadsPerBlock, numBlocks;
  numThreadsPerBlock = NUM_THREADS_PER_BLOCK;
  numBlocks = (qureg.numAmpsPerShard + numThreadsPerBlock - 1) / numThreadsPerBlock;

  for (StateVecIndex_t i = 0; i < qureg.numShards; i++) {
    memopt_adapter::Task task = [=](Qureg q, cudaStream_t s) {
      statevec_applyParamNamedPhaseFuncOverridesKernel<<<numBlocks, numThreadsPerBlock, numThreadsPerBlock * params.numRegs * sizeof(StateVecIndex_t), s>>>(
        convertToKernelParamQureg(q),
        params,
        q.deviceStateVecShards[i],
        i
      );
    };
    memopt_adapter::registerAndExecuteTask(
      {i},
      task,
      qureg,
      stream
    );
  }

  memopt::endStage(stream);
}

__global__ void statevec_swapQubitAmpsBothLocalKernel(
  const __grid_constant__ KernelParamQureg qureg,
  const __grid_constant__ ComplexArray stateVecShard,
  int qb1,
  int qb2
) {
  const StateVecIndex_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const StateVecIndex_t numTasks = qureg.numAmpsPerShard >> 2;
  if (idx >= numTasks) return;

  StateVecIndex_t ind00, ind01, ind10;
  qreal re01, re10, im01, im10;

  ind00 = insertTwoZeroBits(idx, qb1, qb2);
  ind01 = flipBit(ind00, qb1);
  ind10 = flipBit(ind00, qb2);

  re01 = stateVecShard.real[ind01];
  im01 = stateVecShard.imag[ind01];
  re10 = stateVecShard.real[ind10];
  im10 = stateVecShard.imag[ind10];

  stateVecShard.real[ind01] = re10;
  stateVecShard.imag[ind01] = im10;
  stateVecShard.real[ind10] = re01;
  stateVecShard.imag[ind10] = im01;
}

__global__ void statevec_swapQubitAmpsOneLocalOneGlobalKernel(
  const __grid_constant__ KernelParamQureg qureg,
  const __grid_constant__ ComplexArray stateVecShardUp,
  const __grid_constant__ ComplexArray stateVecShardLo,
  int qb1,
  int qb2
) {
  const StateVecIndex_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const StateVecIndex_t numTasks = qureg.numAmpsPerShard >> 1;
  if (idx >= numTasks) return;

  StateVecIndex_t localIndexUp, localIndexLo;
  qreal re01, re10, im01, im10;

  localIndexUp = insertZeroBit(idx, qb1);
  localIndexLo = flipBit(localIndexUp, qb1);

  re01 = stateVecShardUp.real[localIndexLo];
  im01 = stateVecShardUp.imag[localIndexLo];
  re10 = stateVecShardLo.real[localIndexUp];
  im10 = stateVecShardLo.imag[localIndexUp];

  stateVecShardUp.real[localIndexLo] = re10;
  stateVecShardUp.imag[localIndexLo] = im10;
  stateVecShardLo.real[localIndexUp] = re01;
  stateVecShardLo.imag[localIndexUp] = im01;
}

__global__ void statevec_swapQubitAmpsBothGlobalKernel(
  const __grid_constant__ KernelParamQureg qureg,
  const __grid_constant__ ComplexArray stateVecShard01,
  const __grid_constant__ ComplexArray stateVecShard10
) {
  const StateVecIndex_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const StateVecIndex_t numTasks = qureg.numAmpsPerShard;
  if (idx >= numTasks) return;

  qreal re01, re10, im01, im10;

  re01 = stateVecShard01.real[idx];
  im01 = stateVecShard01.imag[idx];
  re10 = stateVecShard10.real[idx];
  im10 = stateVecShard10.imag[idx];

  stateVecShard01.real[idx] = re10;
  stateVecShard01.imag[idx] = im10;
  stateVecShard10.real[idx] = re01;
  stateVecShard10.imag[idx] = im01;
}

void memopt_statevec_swapQubitAmps(cudaStream_t stream, Qureg qureg, int qb1, int qb2) {
  // Make sure qb1 < qb2
  if (qb2 < qb1) {
    int temp = qb1;
    qb1 = qb2;
    qb2 = temp;
  }

  if (qb2 < qureg.numLocalBits) {
    // Both are local bits
    StateVecIndex_t numThreadsPerBlock, numBlocks;
    numThreadsPerBlock = NUM_THREADS_PER_BLOCK;
    numBlocks = ((qureg.numAmpsPerShard >> 2) + numThreadsPerBlock - 1) / numThreadsPerBlock;

    for (StateVecIndex_t i = 0; i < qureg.numShards; i++) {
      memopt_adapter::Task task = [=](Qureg q, cudaStream_t s) {
        statevec_swapQubitAmpsBothLocalKernel<<<numBlocks, numThreadsPerBlock, 0, s>>>(
          convertToKernelParamQureg(q),
          q.deviceStateVecShards[i],
          qb1,
          qb2
        );
      };
      memopt_adapter::registerAndExecuteTask(
        {i},
        task,
        qureg,
        stream
      );
    }
  } else if (qb1 < qureg.numLocalBits) {
    // qb1 is local bit while qb2 is global bit
    StateVecIndex_t numThreadsPerBlock, numBlocks;
    numThreadsPerBlock = NUM_THREADS_PER_BLOCK;
    numBlocks = ((qureg.numAmpsPerShard >> 1) + numThreadsPerBlock - 1) / numThreadsPerBlock;

    for (StateVecIndex_t i = 0; i < (qureg.numShards >> 1); i++) {
      StateVecIndex_t globalIndexUp = insertZeroBit(i, qb2 - qureg.numLocalBits);
      StateVecIndex_t globalIndexLo = flipBit(globalIndexUp, qb2 - qureg.numLocalBits);
      memopt_adapter::Task task = [=](Qureg q, cudaStream_t s) {
        statevec_swapQubitAmpsOneLocalOneGlobalKernel<<<numBlocks, numThreadsPerBlock, 0, s>>>(
          convertToKernelParamQureg(q),
          q.deviceStateVecShards[globalIndexUp],
          q.deviceStateVecShards[globalIndexLo],
          qb1,
          qb2
        );
      };
      memopt_adapter::registerAndExecuteTask(
        {globalIndexUp, globalIndexLo},
        task,
        qureg,
        stream
      );
    }
  } else {
    // Both are global bits
    // qb1 is local bit while qb2 is global bit
    StateVecIndex_t numThreadsPerBlock, numBlocks;
    numThreadsPerBlock = NUM_THREADS_PER_BLOCK;
    numBlocks = (qureg.numAmpsPerShard + numThreadsPerBlock - 1) / numThreadsPerBlock;

    for (StateVecIndex_t i = 0; i < (qureg.numShards >> 2); i++) {
      StateVecIndex_t globalIndex00 = insertTwoZeroBits(i, qb1 - qureg.numLocalBits, qb2 - qureg.numLocalBits);
      StateVecIndex_t globalIndex01 = flipBit(globalIndex00, qb1 - qureg.numLocalBits);
      StateVecIndex_t globalIndex10 = flipBit(globalIndex00, qb2 - qureg.numLocalBits);

      memopt_adapter::Task task = [=](Qureg q, cudaStream_t s) {
        statevec_swapQubitAmpsBothGlobalKernel<<<numBlocks, numThreadsPerBlock, 0, s>>>(
          convertToKernelParamQureg(q),
          q.deviceStateVecShards[globalIndex01],
          q.deviceStateVecShards[globalIndex10]
        );
      };
      memopt_adapter::registerAndExecuteTask(
        {globalIndex01, globalIndex10},
        task,
        qureg,
        stream
      );
    }
  }

  memopt::endStage(stream);
}

cudaGraph_t captureCudaGraphForFullQFT(cudaStream_t stream, Qureg qureg) {
  // Does not support density matrix
  assert(!qureg.isDensityMatrix);

  checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  // Start with top/left-most qubit, work down
  for (int q = qureg.numQubitsInStateVec - 1; q >= 0; q--) {
    memopt_statevec_hadamard(stream, qureg, q);

    if (q == 0)
      break;

    ApplyParamNamedPhaseFuncOverridesParams params;
    params.numRegs = 2;
    params.numQubitsPerReg[0] = q;
    params.numQubitsPerReg[1] = 1;
    for (int i = 0; i < q + 1; i++)
      params.qubits[i] = i;

    params.numParams = 1;
    params.params[1] = M_PI / (1 << q);

    params.encoding = UNSIGNED;
    params.phaseFuncName = SCALED_PRODUCT;

    params.numOverrides = 0;

    params.conj = 0;

    memopt_statevec_applyParamNamedPhaseFuncOverrides(stream, qureg, params);
  }

  for (int i = 0; i < (qureg.numQubitsInStateVec / 2); i++) {
    int qb1 = i;
    int qb2 = qureg.numQubitsInStateVec - i - 1;

    memopt_statevec_swapQubitAmps(stream, qureg, qb1, qb2);
  }

  cudaGraph_t graph;
  checkCudaErrors(cudaStreamEndCapture(stream, &graph));
  return graph;
}

#ifdef __cplusplus
extern "C" {
#endif

// Copied from QuEST_gpu_common.cu
int GPUExists(void) {
  int deviceCount, device;
  int gpuDeviceCount = 0;
  struct cudaDeviceProp properties;
  cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
  if (cudaResultCode != cudaSuccess) deviceCount = 0;
  /* machines with no GPUs can still report one emulation device */
  for (device = 0; device < deviceCount; ++device) {
    cudaGetDeviceProperties(&properties, device);
    if (properties.major != 9999) { /* 9999 means emulation only */
      ++gpuDeviceCount;
    }
  }
  if (gpuDeviceCount)
    return 1;
  else
    return 0;
}

// Copied from QuEST_gpu_common.cu
void seedQuEST(QuESTEnv* env, unsigned long int* seedArray, int numSeeds) {
  // free existing seed array, if exists
  if (env->seeds != NULL)
    free(env->seeds);

  // record keys in permanent heap
  env->seeds = (unsigned long int*)malloc(numSeeds * sizeof *(env->seeds));
  for (int i = 0; i < numSeeds; i++)
    (env->seeds)[i] = seedArray[i];
  env->numSeeds = numSeeds;

  // pass keys to Mersenne Twister seeder
  init_by_array(seedArray, numSeeds);
}

QuESTEnv createQuESTEnv() {
  // Initialize memopt
  memopt::ConfigurationManager::exportDefaultConfiguration();
  memopt::ConfigurationManager::loadConfiguration();
  memopt::initializeCudaDevice();

  validateGPUExists(GPUExists(), __func__);

  QuESTEnv env;
  env.rank = 0;
  env.numRanks = 1;

  env.seeds = NULL;
  env.numSeeds = 0;
  seedQuESTDefault(&env);

  return env;
}

void destroyQuESTEnv(QuESTEnv env) {
  free(env.seeds);
}

void statevec_createQureg(Qureg* qureg, int numQubits, QuESTEnv env) {
  assert(numQubits > NUM_GLOBAL_BITS);

  const StateVecIndex_t numShards = 1L << NUM_GLOBAL_BITS;
  const StateVecIndex_t numTotalAmps = 1L << numQubits;
  const StateVecIndex_t numAmpsPerShard = numTotalAmps / numShards;

  qureg->numQubitsInStateVec = numQubits;
  qureg->numAmpsPerChunk = numTotalAmps;
  qureg->numAmpsPerShard = numAmpsPerShard;
  qureg->numAmpsTotal = numTotalAmps;
  qureg->chunkId = env.rank;
  qureg->numChunks = env.numRanks;
  qureg->numShards = numShards;
  qureg->numGlobalBits = NUM_GLOBAL_BITS;
  qureg->numLocalBits = numQubits - NUM_GLOBAL_BITS;
  qureg->isDensityMatrix = 0;

  for (StateVecIndex_t i = 0; i < qureg->numShards; i++) {
    // The real parts and imaginary parts are packed into one array
    memopt_adapter::allocateShardAndRegister(&(qureg->deviceStateVecShards[i].real), 2 * qureg->numAmpsPerShard * sizeof(qreal));
    qureg->deviceStateVecShards[i].imag = qureg->deviceStateVecShards[i].real + qureg->numAmpsPerShard;
  }
}

void statevec_destroyQureg(Qureg qureg, QuESTEnv env) {
  for (StateVecIndex_t i = 0; i < qureg.numShards; i++) {
    checkCudaErrors(cudaFree(qureg.deviceStateVecShards[i].real));
  }
}

void applyFullQFTWithMemopt(Qureg* qureg) {
  size_t totalShardSize = 0;
  for (const auto& [addr, size] : memopt::MemoryManager::managedMemoryAddressToSizeMap) {
    totalShardSize += size;
  }
  printf("totalShardSize (MiB) = %.6lf\n", (double)totalShardSize * 1e-6);

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));

  cudaGraph_t graph = captureCudaGraphForFullQFT(stream, *qureg);

  printf("Number of tasks = %llu\n", memopt_adapter::tasks.size());

  checkCudaErrors(cudaGraphDebugDotPrint(graph, "graph.dot", cudaGraphDebugDotFlagsVerbose));

  if (memopt::ConfigurationManager::getConfig().generic.optimize) {
    auto optimizedGraph = memopt::profileAndOptimize(graph);

    statevec_initZeroState(*qureg);

    float runningTime;
    std::map<void*, void*> managedDeviceArrayToHostArrayMap;
    memopt::executeOptimizedGraph(
      optimizedGraph,
      [=](int taskId, std::map<void*, void*> addressUpdate, cudaStream_t stream) {
        memopt_adapter::executeRandomTask(*qureg, taskId, addressUpdate, stream);
      },
      runningTime,
      managedDeviceArrayToHostArrayMap
    );

    printf("Total time used (s): %.6f\n", runningTime);

    for (StateVecIndex_t i = 0; i < qureg->numShards; i++) {
      memopt_adapter::moveDataBackToDevice(qureg->deviceStateVecShards[i].real, managedDeviceArrayToHostArrayMap, qureg->numAmpsPerShard);
      qureg->deviceStateVecShards[i].imag = qureg->deviceStateVecShards[i].real + qureg->numAmpsPerShard;
    }
    checkCudaErrors(cudaDeviceSynchronize());
  } else {
    if (memopt::ConfigurationManager::getConfig().generic.useUM) {
      size_t available = 1024ULL * 1024ULL * memopt::ConfigurationManager::getConfig().generic.availableMemoryForUMInMiB;
      memopt::reduceAvailableMemoryForUM(available);
    }

    memopt::PeakMemoryUsageProfiler profiler;
    memopt::CudaEventClock clock;
    profiler.start();

    cudaGraphExec_t graphExec;
    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph));
    clock.start(stream);
    checkCudaErrors(cudaGraphLaunch(graphExec, stream));
    clock.end();
    checkCudaErrors(cudaStreamSynchronize(stream));

    size_t peakMem = profiler.end();
    printf("Peak memory usage (MiB): %.6f\n", (double)peakMem / 1024.0 / 1024.0);
    printf("Total time used (s): %.6f\n", clock.getTimeInSeconds());

    if (memopt::ConfigurationManager::getConfig().generic.useUM) {
      memopt::resetAvailableMemoryForUM();
    }

    checkCudaErrors(cudaGraphExecDestroy(graphExec));
    checkCudaErrors(cudaGraphDestroy(graph));
    checkCudaErrors(cudaStreamDestroy(stream));
  }
}

#ifdef __cplusplus
}
#endif
