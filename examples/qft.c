#include <stdio.h>
#include <stdlib.h>

#include "QuEST.h"

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

  applyFullQFT(qureg);

  printf("prob of solution |%llx> = %.8lf\n", 0, getProbAmp(qureg, 0));
  printf("prob of solution |%llx> = %.8lf\n", ((unsigned long long)1 << numQubits) - 1, getProbAmp(qureg, 0));

  destroyQureg(qureg, env);
  destroyQuESTEnv(env);
  return 0;
}