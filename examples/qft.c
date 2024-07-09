#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "QuEST.h"

double getWallClockTimeInSeconds() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec + (double)t.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
  int numQubits;
  if (argc == 2) {
    numQubits = atoi(argv[1]);
  } else {
    numQubits = 8;
  }
  printf("numQubits = %d\n", numQubits);

  QuESTEnv env = createQuESTEnv();

  Qureg qureg = createQureg(numQubits, env);

  initZeroState(qureg);

  double start = getWallClockTimeInSeconds();

  applyFullQFT(qureg);

  double end = getWallClockTimeInSeconds();

  printf("Total time used (s): %.6lf\n", end - start);

  printf("prob of solution |%llx> = %.8e\n", 0, getProbAmp(qureg, 0));
  long long index = 0;
  for (int i = 0; i < numQubits; i++) {
    index = index + (1 << i);
    printf("prob of solution |%llx> = %.8e\n", index, getProbAmp(qureg, index));
  }

  destroyQureg(qureg, env);
  destroyQuESTEnv(env);
  return 0;
}