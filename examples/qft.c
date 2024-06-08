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

  applyFullQFTWithMemopt(qureg, "config.json");

  printf("prob of solution |%llx> = %.8lf\n", 0, getProbAmp(qureg, 0));
  long long index = 0;
  for (int i = 0; i < numQubits; i++) {
    index = index + (1 << i);
    printf("prob of solution |%llx> = %.8lf\n", index, getProbAmp(qureg, index));
  }

  destroyQureg(qureg, env);
  destroyQuESTEnv(env);
  return 0;
}