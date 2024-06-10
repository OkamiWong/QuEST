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
  checkCudaErrors(cudaMalloc(p, s));
  memopt::registerManagedMemoryAddress(*p, s);
  memopt::registerApplicationInput(*p);
  memopt::registerApplicationOutput(*p);
}

typedef std::function<void(Qureg, cudaStream_t)> Task;

std::vector<Task> tasks;

void registerAndExecuteTask(
  std::vector<StateVecIndex_t> inputShardIndices,
  std::vector<StateVecIndex_t> outputShardIndices,
  Task task,
  Qureg qureg,
  cudaStream_t stream
) {
  auto taskId = tasks.size();
  tasks.push_back(task);

  std::vector<void*> inputs, outputs;
  for (auto i : inputShardIndices) {
    inputs.push_back(qureg.deviceStateVecShards[i].real);
    inputs.push_back(qureg.deviceStateVecShards[i].imag);
  }
  for (auto i : outputShardIndices) {
    outputs.push_back(qureg.deviceStateVecShards[i].real);
    outputs.push_back(qureg.deviceStateVecShards[i].imag);
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
    tryUpdatingAddress(qureg.deviceStateVecShards[i].imag, addressUpdateMap);
  }
  tasks[taskId](qureg, stream);
}

}  // namespace memopt_adapter

// QuEST speicifc definitions
constexpr int MAX_NUM_QUBITS = 64;
constexpr int MAX_NUM_PHASE_FUNC_OVERRIDES = 8;

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
    checkCudaErrors(cudaMemset(qureg.deviceStateVecShards[i].real, 0, qureg.numAmpsPerShard * sizeof(qreal)));
    checkCudaErrors(cudaMemset(qureg.deviceStateVecShards[i].imag, 0, qureg.numAmpsPerShard * sizeof(qreal)));
  }

  // Set |000...000>'s amp to 1
  qreal one = 1, zero = 0;
  checkCudaErrors(cudaMemcpy(&qureg.deviceStateVecShards[0].real[0], &one, sizeof(qreal), cudaMemcpyDefault));
  checkCudaErrors(cudaMemcpy(&qureg.deviceStateVecShards[0].imag[0], &zero, sizeof(qreal), cudaMemcpyDefault));
}

__global__ void statevec_hadamardLocalBitKernel(Qureg qureg, StateVecIndex_t globalIndex, int targetQubit) {
  const StateVecIndex_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const StateVecIndex_t numTasks = qureg.numAmpsPerShard >> 1;
  if (idx >= numTasks) return;

  StateVecIndex_t indexUp = insertZeroBit(idx, targetQubit);
  StateVecIndex_t indexLo = flipBit(indexUp, targetQubit);

  qreal* stateVecReal = qureg.deviceStateVecShards[globalIndex].real;
  qreal* stateVecImag = qureg.deviceStateVecShards[globalIndex].imag;

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

__global__ void statevec_hadamardGlobalBitKernel(Qureg qureg, StateVecIndex_t globalIndex, int targetQubit) {
  const StateVecIndex_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= qureg.numAmpsPerShard) return;

  StateVecIndex_t globalIndexUp = globalIndex;
  StateVecIndex_t globalIndexLo = globalIndex | (1 << (targetQubit - qureg.numLocalBits));

  ComplexArray* stateVecShardUp = &qureg.deviceStateVecShards[globalIndexUp];
  ComplexArray* stateVecShardLo = &qureg.deviceStateVecShards[globalIndexLo];

  qreal *stateRealUp, *stateRealLo, *stateImagUp, *stateImagLo;
  stateRealUp = &stateVecShardUp->real[idx];
  stateImagUp = &stateVecShardUp->imag[idx];
  stateRealLo = &stateVecShardLo->real[idx];
  stateImagLo = &stateVecShardLo->imag[idx];

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
    numThreadsPerBlock = 128;
    numBlocks = ((qureg.numAmpsPerShard >> 1) + numThreadsPerBlock - 1) / numThreadsPerBlock;

    for (StateVecIndex_t i = 0; i < qureg.numShards; i++) {
      memopt_adapter::Task task = [=](Qureg q, cudaStream_t s) {
        statevec_hadamardLocalBitKernel<<<numBlocks, numThreadsPerBlock, 0, s>>>(
          q,
          i,
          targetQubit
        );
      };
      memopt_adapter::registerAndExecuteTask(
        {i},
        {i},
        task,
        qureg,
        stream
      );
    }
  } else {
    StateVecIndex_t numThreadsPerBlock, numBlocks;
    numThreadsPerBlock = 128;
    numBlocks = (qureg.numAmpsPerShard + numThreadsPerBlock - 1) / numThreadsPerBlock;

    for (StateVecIndex_t i = 0; i < (qureg.numShards >> 1); i++) {
      StateVecIndex_t globalIndexUp = insertZeroBit(i, targetQubit - qureg.numLocalBits);
      StateVecIndex_t globalIndexLo = flipBit(globalIndexUp, targetQubit - qureg.numLocalBits);
      memopt_adapter::Task task = [=](Qureg q, cudaStream_t s) {
        statevec_hadamardGlobalBitKernel<<<numBlocks, numThreadsPerBlock, 0, s>>>(
          q,
          globalIndexUp,
          targetQubit
        );
      };
      memopt_adapter::registerAndExecuteTask(
        {globalIndexUp, globalIndexLo},
        {globalIndexUp, globalIndexLo},
        task,
        qureg,
        stream
      );
    }
  }
}

__forceinline__ __device__ void setMultiRegPhaseInds(
  Qureg* qureg,
  StateVecIndex_t* phaseInds, StateVecIndex_t fullIndex,
  int* qubits, int* numQubitsPerReg, int numRegs, enum bitEncoding encoding
) {
  size_t stride = qureg->numAmpsPerShard;
  size_t offset = fullIndex & ((1 << qureg->numLocalBits) - 1);

  if (encoding == UNSIGNED) {
    int flatIndex = 0;
    for (int r = 0; r < numRegs; r++) {
      phaseInds[r * stride + offset] = 0LL;
      for (int q = 0; q < numQubitsPerReg[r]; q++)
        phaseInds[r * stride + offset] += (1LL << q) * getBit(fullIndex, qubits[flatIndex++]);
    }
  } else if (encoding == TWOS_COMPLEMENT) {
    int flatIndex = 0;
    for (int r = 0; r < numRegs; r++) {
      phaseInds[r * stride + offset] = 0LL;
      for (int q = 0; q < numQubitsPerReg[r] - 1; q++)
        phaseInds[r * stride + offset] += (1LL << q) * getBit(fullIndex, qubits[flatIndex++]);

      // Use final qubit to indicate sign
      if (getBit(fullIndex, qubits[flatIndex++]) == 1)
        phaseInds[r * stride + offset] -= (1LL << (numQubitsPerReg[r] - 1));
    }
  }
}

__forceinline__ __device__ StateVecIndex_t getIndOfMultiRegPhaseOverride(
  Qureg* qureg,
  StateVecIndex_t fullIndex,
  StateVecIndex_t* phaseInds, int numRegs,
  StateVecIndex_t* overrideInds, int numOverrides
) {
  size_t stride = qureg->numAmpsPerShard;
  size_t offset = fullIndex & ((1 << qureg->numLocalBits) - 1);

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
  int numRegs, enum phaseFunc phaseFuncName, qreal* params, int numParams
) {
  // determine norm
  qreal norm = 0;
  if (phaseFuncName == SCALED_INVERSE_SHIFTED_NORM) {
    for (int r = 0; r < numRegs; r++) {
      qreal dif = phaseInds[r * stride + offset] - params[2 + r];
      norm += dif * dif;
    }
  } else
    for (int r = 0; r < numRegs; r++)
      norm += phaseInds[r * stride + offset] * phaseInds[r * stride + offset];
  norm = sqrt(norm);

  // determine phase via phase function

  if (phaseFuncName == NORM)
    return norm;

  if (phaseFuncName == INVERSE_NORM)
    return (norm == 0.) ? params[0] : 1 / norm;  // smallest non-zero norm is 1

  if (phaseFuncName == SCALED_NORM)
    return params[0] * norm;

  if (
    phaseFuncName == SCALED_INVERSE_NORM || phaseFuncName == SCALED_INVERSE_SHIFTED_NORM
  )
    return (norm <= REAL_EPS) ? params[1] : params[0] / norm;  // unless shifted closer to zero
}

__forceinline__ __device__ qreal evalProductPhaseFunc(
  StateVecIndex_t* phaseInds, size_t stride, size_t offset,
  int numRegs, enum phaseFunc phaseFuncName, qreal* params, int numParams
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
  int numRegs, enum phaseFunc phaseFuncName, qreal* params, int numParams
) {
  // evaluate distance (depends on phase function)
  qreal dist = 0;
  if (phaseFuncName == SCALED_INVERSE_SHIFTED_DISTANCE) {
    for (int r = 0; r < numRegs; r += 2) {
      qreal dif = (phaseInds[r * stride + offset] - phaseInds[(r + 1) * stride + offset] - params[2 + r / 2]);
      dist += dif * dif;
    }
  } else if (phaseFuncName == SCALED_INVERSE_SHIFTED_WEIGHTED_DISTANCE) {
    for (int r = 0; r < numRegs; r += 2) {
      qreal dif = (phaseInds[r * stride + offset] - phaseInds[(r + 1) * stride + offset] - params[2 + r + 1]);
      dist += params[2 + r] * dif * dif;
    }
  } else
    for (int r = 0; r < numRegs; r += 2) {
      qreal dif = (phaseInds[(r + 1) * stride + offset] - phaseInds[r * stride + offset]);
      dist += dif * dif;
    }

  // if sqrt() arg of distance would be negative, set it to zero, to subsequently be set to the divergence param
  if (dist < 0)
    dist = 0;

  dist = sqrt(dist);

  if (phaseFuncName == DISTANCE)
    return dist;

  if (phaseFuncName == INVERSE_DISTANCE)
    return (dist == 0.) ? params[0] : 1 / dist;  // smallest non-zero dist is 1

  if (phaseFuncName == SCALED_DISTANCE)
    return params[0] * dist;

  if (
    phaseFuncName == SCALED_INVERSE_DISTANCE || phaseFuncName == SCALED_INVERSE_SHIFTED_DISTANCE || phaseFuncName == SCALED_INVERSE_SHIFTED_WEIGHTED_DISTANCE
  )
    return (dist <= REAL_EPS) ? params[1] : params[0] / dist;  // unless shifted closer
}

__forceinline__ __device__ qreal getPhaseFromParamNamedFunc(
  Qureg* qureg,
  StateVecIndex_t fullIndex,
  StateVecIndex_t* phaseInds, int numRegs,
  enum phaseFunc phaseFuncName, qreal* params, int numParams
) {
  size_t stride = qureg->numAmpsPerShard;
  size_t offset = fullIndex & ((1 << qureg->numLocalBits) - 1);

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
  Qureg* qureg,
  StateVecIndex_t globalIndex, StateVecIndex_t localIndex,
  qreal phase, int conj
) {
  phase *= (1 - 2 * conj);
  qreal c = cos(phase);
  qreal s = sin(phase);

  qreal re, im;
  re = qureg->deviceStateVecShards[globalIndex].real[localIndex];
  im = qureg->deviceStateVecShards[globalIndex].imag[localIndex];
  qureg->deviceStateVecShards[globalIndex].real[localIndex] = re * c - im * s;
  qureg->deviceStateVecShards[globalIndex].imag[localIndex] = re * s + im * c;
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

__global__ void statevec_applyParamNamedPhaseFuncOverridesKernel(
  Qureg qureg,
  ApplyParamNamedPhaseFuncOverridesParams params,
  StateVecIndex_t* phaseInds,
  StateVecIndex_t globalIndex
) {
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
  applyPhaseToAmp(&qureg, globalIndex, idx, phase, params.conj);
}

void memopt_statevec_applyParamNamedPhaseFuncOverrides(
  cudaStream_t stream,
  Qureg qureg,
  ApplyParamNamedPhaseFuncOverridesParams params
) {
  StateVecIndex_t numThreadsPerBlock, numBlocks;
  numThreadsPerBlock = 128;
  numBlocks = (qureg.numAmpsPerShard + numThreadsPerBlock - 1) / numThreadsPerBlock;

  for (StateVecIndex_t i = 0; i < qureg.numShards; i++) {
    memopt_adapter::Task task = [=](Qureg q, cudaStream_t s) {
      StateVecIndex_t* d_phaseInds;
      checkCudaErrors(cudaMallocAsync(&d_phaseInds, params.numRegs * q.numAmpsPerShard * sizeof *d_phaseInds, s));
      statevec_applyParamNamedPhaseFuncOverridesKernel<<<numBlocks, numThreadsPerBlock, 0, s>>>(
        q,
        params,
        d_phaseInds,
        i
      );
      checkCudaErrors(cudaFreeAsync(d_phaseInds, s));
    };
    memopt_adapter::registerAndExecuteTask(
      {i},
      {i},
      task,
      qureg,
      stream
    );
  }
}

__global__ void statevec_swapQubitAmpsBothLocalKernel(Qureg qureg, StateVecIndex_t globalIndex, int qb1, int qb2) {
  const StateVecIndex_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const StateVecIndex_t numTasks = qureg.numAmpsPerShard >> 2;
  if (idx >= numTasks) return;

  qreal* reVec = qureg.deviceStateVecShards[globalIndex].real;
  qreal* imVec = qureg.deviceStateVecShards[globalIndex].imag;

  StateVecIndex_t ind00, ind01, ind10;
  qreal re01, re10, im01, im10;

  ind00 = insertTwoZeroBits(idx, qb1, qb2);
  ind01 = flipBit(ind00, qb1);
  ind10 = flipBit(ind00, qb2);

  re01 = reVec[ind01];
  im01 = imVec[ind01];
  re10 = reVec[ind10];
  im10 = imVec[ind10];

  reVec[ind01] = re10;
  imVec[ind01] = im10;
  reVec[ind10] = re01;
  imVec[ind10] = im01;
}

__global__ void statevec_swapQubitAmpsOneLocalOneGlobalKernel(Qureg qureg, StateVecIndex_t globalIndex, int qb1, int qb2) {
  const StateVecIndex_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const StateVecIndex_t numTasks = qureg.numAmpsPerShard >> 1;
  if (idx >= numTasks) return;

  StateVecIndex_t globalIndexUp, globalIndexLo;
  StateVecIndex_t localIndexUp, localIndexLo;
  qreal re01, re10, im01, im10;

  globalIndexUp = globalIndex;
  globalIndexLo = flipBit(globalIndexUp, qb2 - qureg.numLocalBits);

  localIndexUp = insertZeroBit(idx, qb1);
  localIndexLo = flipBit(localIndexUp, qb1);

  re01 = qureg.deviceStateVecShards[globalIndexUp].real[localIndexLo];
  im01 = qureg.deviceStateVecShards[globalIndexUp].imag[localIndexLo];
  re10 = qureg.deviceStateVecShards[globalIndexLo].real[localIndexUp];
  im10 = qureg.deviceStateVecShards[globalIndexLo].imag[localIndexUp];

  qureg.deviceStateVecShards[globalIndexUp].real[localIndexLo] = re10;
  qureg.deviceStateVecShards[globalIndexUp].imag[localIndexLo] = im10;
  qureg.deviceStateVecShards[globalIndexLo].real[localIndexUp] = re01;
  qureg.deviceStateVecShards[globalIndexLo].imag[localIndexUp] = im01;
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
    numThreadsPerBlock = 128;
    numBlocks = ((qureg.numAmpsPerShard >> 2) + numThreadsPerBlock - 1) / numThreadsPerBlock;

    for (StateVecIndex_t i = 0; i < qureg.numShards; i++) {
      memopt_adapter::Task task = [=](Qureg q, cudaStream_t s) {
        statevec_swapQubitAmpsBothLocalKernel<<<numBlocks, numThreadsPerBlock, 0, s>>>(
          q,
          i,
          qb1,
          qb2
        );
      };
      memopt_adapter::registerAndExecuteTask(
        {i},
        {i},
        task,
        qureg,
        stream
      );
    }
  } else if (qb1 < qureg.numLocalBits) {
    // qb1 is local bit while qb2 is global bit
    StateVecIndex_t numThreadsPerBlock, numBlocks;
    numThreadsPerBlock = 128;
    numBlocks = ((qureg.numAmpsPerShard >> 1) + numThreadsPerBlock - 1) / numThreadsPerBlock;

    for (StateVecIndex_t i = 0; i < (qureg.numShards >> 1); i++) {
      StateVecIndex_t globalIndexUp = insertZeroBit(i, qb2 - qureg.numLocalBits);
      StateVecIndex_t globalIndexLo = flipBit(globalIndexUp, qb2 - qureg.numLocalBits);
      memopt_adapter::Task task = [=](Qureg q, cudaStream_t s) {
        statevec_swapQubitAmpsOneLocalOneGlobalKernel<<<numBlocks, numThreadsPerBlock, 0, s>>>(
          q,
          globalIndexUp,
          qb1,
          qb2
        );
      };
      memopt_adapter::registerAndExecuteTask(
        {globalIndexUp, globalIndexLo},
        {globalIndexUp, globalIndexLo},
        task,
        qureg,
        stream
      );
    }
  } else {
    // Both are global bits
    for (StateVecIndex_t i = 0; i < (qureg.numShards >> 2); i++) {
      StateVecIndex_t globalIndex00 = insertTwoZeroBits(i, qb1 - qureg.numLocalBits, qb2 - qureg.numLocalBits);
      StateVecIndex_t globalIndex01 = flipBit(globalIndex00, qb1 - qureg.numLocalBits);
      StateVecIndex_t globalIndex10 = flipBit(globalIndex00, qb2 - qureg.numLocalBits);

      memopt_adapter::Task task = [=](Qureg q, cudaStream_t s) {
        qreal *realTemp, *imagTemp;
        checkCudaErrors(cudaMallocAsync(&realTemp, q.numAmpsPerShard * sizeof(qreal), s));
        checkCudaErrors(cudaMallocAsync(&imagTemp, q.numAmpsPerShard * sizeof(qreal), s));

        checkCudaErrors(cudaMemcpyAsync(realTemp, q.deviceStateVecShards[globalIndex01].real, q.numAmpsPerShard * sizeof(qreal), cudaMemcpyDefault, s));
        checkCudaErrors(cudaMemcpyAsync(imagTemp, q.deviceStateVecShards[globalIndex01].imag, q.numAmpsPerShard * sizeof(qreal), cudaMemcpyDefault, s));
        checkCudaErrors(cudaMemcpyAsync(q.deviceStateVecShards[globalIndex01].real, q.deviceStateVecShards[globalIndex10].real, q.numAmpsPerShard * sizeof(qreal), cudaMemcpyDefault, s));
        checkCudaErrors(cudaMemcpyAsync(q.deviceStateVecShards[globalIndex01].imag, q.deviceStateVecShards[globalIndex10].imag, q.numAmpsPerShard * sizeof(qreal), cudaMemcpyDefault, s));
        checkCudaErrors(cudaMemcpyAsync(q.deviceStateVecShards[globalIndex10].real, realTemp, q.numAmpsPerShard * sizeof(qreal), cudaMemcpyDefault, s));
        checkCudaErrors(cudaMemcpyAsync(q.deviceStateVecShards[globalIndex10].imag, imagTemp, q.numAmpsPerShard * sizeof(qreal), cudaMemcpyDefault, s));

        checkCudaErrors(cudaFreeAsync(realTemp, s));
        checkCudaErrors(cudaFreeAsync(imagTemp, s));
      };
      memopt_adapter::registerAndExecuteTask(
        {globalIndex01, globalIndex10},
        {globalIndex01, globalIndex10},
        task,
        qureg,
        stream
      );
    }
  }
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

#define NUM_GLOBAL_BITS 4

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
  assert((1 << NUM_GLOBAL_BITS) <= MAX_NUM_SHARDS);
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
    memopt_adapter::allocateShardAndRegister(&(qureg->deviceStateVecShards[i].real), qureg->numAmpsPerShard * sizeof(qreal));
    memopt_adapter::allocateShardAndRegister(&(qureg->deviceStateVecShards[i].imag), qureg->numAmpsPerShard * sizeof(qreal));
  }
}

void statevec_destroyQureg(Qureg qureg, QuESTEnv env) {
  for (StateVecIndex_t i = 0; i < qureg.numShards; i++) {
    checkCudaErrors(cudaFree(qureg.deviceStateVecShards[i].real));
    checkCudaErrors(cudaFree(qureg.deviceStateVecShards[i].imag));
  }
}

void applyFullQFTWithMemopt(Qureg qureg) {
  size_t totalShardSize = 0;
  for (const auto& [addr, size] : memopt::MemoryManager::managedMemoryAddressToSizeMap) {
    totalShardSize += size;
  }
  printf("totalShardSize (MiB) = %.6lf\n", (double)totalShardSize * 1e-6);

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));

  cudaGraph_t graph = captureCudaGraphForFullQFT(stream, qureg);

  checkCudaErrors(cudaGraphDebugDotPrint(graph, "graph.dot", cudaGraphDebugDotFlagsVerbose));

  cudaGraphExec_t graphExec;
  checkCudaErrors(cudaGraphInstantiate(&graphExec, graph));
  checkCudaErrors(cudaGraphLaunch(graphExec, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  checkCudaErrors(cudaGraphExecDestroy(graphExec));
  checkCudaErrors(cudaGraphDestroy(graph));
  checkCudaErrors(cudaStreamDestroy(stream));
}

#ifdef __cplusplus
}
#endif
