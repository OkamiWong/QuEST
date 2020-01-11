
#include "catch.hpp"
#include "QuEST.h"
#include "QuEST_test_utils.hpp"

/** The default number of qubits in the registers created for unit testing 
 * (both statevectors and density matrices). Creation of non-NUM_QUBITS sized 
 * Quregs should be justified in a comment. 
 * Note that the smaller this number is, the fewer nodes can be employed in 
 * distribution testing, since each node must contain at least one amplitude.
 * Furthermore, the larger this number is, the greater the deviation of correct 
 * results from their expected value, due to numerical error; this is especially 
 * apparent for density matrices.
 */
#define NUM_QUBITS 5

/* allows concise use of Contains in catch's REQUIRE_THROWS_WITH */
using Catch::Matchers::Contains;



TEST_CASE( "fromComplex", "[data_structures]" ) {
    
    Complex a = {.real=.5, .imag=-.2};
    qcomp b = fromComplex(a);
    
    REQUIRE( a.real == real(b) );
    REQUIRE( a.imag == imag(b) );
}



TEST_CASE( "getStaticComplexMatrixN", "[data_structures]" ) {
    
    /* use of this function is illegal in C++ */
    SUCCEED( );
}



TEST_CASE( "toComplex", "[data_structures]" ) {
    
    qcomp a = .5 - .2i;
    Complex b = toComplex(a);
    
    REQUIRE( real(a) == b.real );
    REQUIRE( imag(a) == b.imag );
}



TEST_CASE( "createCloneQureg", "[data_structures]" ) {
    
    QuESTEnv env = createQuESTEnv();
    
    SECTION( "state-vector" ) {
        
        Qureg a = createQureg(NUM_QUBITS, env);    
        Qureg b = createCloneQureg(a, env);
        
        // check properties are the same
        REQUIRE( b.isDensityMatrix == a.isDensityMatrix );
        REQUIRE( b.numQubitsRepresented == a.numQubitsRepresented );
        REQUIRE( b.numQubitsInStateVec == a.numQubitsInStateVec );
        REQUIRE( b.numAmpsPerChunk == a.numAmpsPerChunk );
        REQUIRE( b.numAmpsTotal == a.numAmpsTotal );
        
        // check state-vector is the same (works for GPU and distributed)
        REQUIRE( areEqual(a, b) );  
        
        destroyQureg(a, env);
        destroyQureg(b, env);
    }
    SECTION( "density-matrix" ) {
        
        Qureg a = createDensityQureg(NUM_QUBITS, env);
        Qureg b = createCloneQureg(a, env);
        
        // check properties are the same
        REQUIRE( b.isDensityMatrix == a.isDensityMatrix );
        REQUIRE( b.numQubitsRepresented == a.numQubitsRepresented );
        REQUIRE( b.numQubitsInStateVec == a.numQubitsInStateVec );
        REQUIRE( b.numAmpsPerChunk == a.numAmpsPerChunk );
        REQUIRE( b.numAmpsTotal == a.numAmpsTotal );
        
        // check state-vector is the same (works for GPU and distributed)
        REQUIRE( areEqual(a, b) );  
        
        destroyQureg(a, env);
        destroyQureg(b, env);
    }
    destroyQuESTEnv(env);
}



TEST_CASE( "createComplexMatrixN", "[data_structures]" ) {
    
    SECTION( "correctness" ) {
        
        int numQb = GENERATE( range(1,10+1) );
        ComplexMatrixN m = createComplexMatrixN(numQb);
        
        // ensure elems are created and initialised to 0
        REQUIRE( areEqual(toQMatrix(m), getZeroMatrix(1<<numQb)) );
        
        destroyComplexMatrixN(m);
    }
    SECTION( "input validation" ) {
        
        SECTION( "number of qubits" ) {
            
            int numQb = GENERATE( -1, 0 );
            REQUIRE_THROWS_WITH( createComplexMatrixN(numQb), Contains("Invalid number of qubits") );
        }
    }
}



TEST_CASE( "createDensityQureg", "[data_structures]" ) {
    
    QuESTEnv env = createQuESTEnv();
    
    // must be at least one amplitude per node
    int minNumQb = calcLog2(env.numRanks) - 1; // density matrix has 2*numQb in state-vec
    if (minNumQb <= 0)
        minNumQb = 1;
    
    SECTION( "correctness" ) {
        
        // try 10 valid number of qubits
        int numQb = GENERATE_COPY( range(minNumQb, minNumQb+10) );
        Qureg reg = createDensityQureg(numQb, env);
        
        // ensure elems (CPU and/or GPU) are created, and reg begins in |0><0|
        QMatrix ref = getZeroMatrix(1<<numQb);
        ref[0][0] = 1; // |0><0|
        REQUIRE( areEqual(reg, ref) );
        
        destroyQureg(reg, env);
    }
    SECTION( "input validation") {
        
        SECTION( "number of qubits" ) {
            
            int numQb = GENERATE( -1, 0 );
            REQUIRE_THROWS_WITH( createDensityQureg(numQb, env), Contains("Invalid number of qubits") );
        }
        SECTION( "number of amplitudes" ) {
            
            // too many amplitudes to store in type
            int maxQb = (int) calcLog2(SIZE_MAX) - 1;
            REQUIRE_THROWS_WITH( createDensityQureg(maxQb+1, env), Contains("Too many qubits") && Contains("size_t type") );
            
            /* n-qubit density matrix contains 2^(2n) amplitudes 
             * so can be spread between at most 2^(2n) ranks
             */
            int minQb = GENERATE( range(3,10) );
            env.numRanks = (int) pow(2, 2*minQb);
            int numQb = GENERATE_COPY( range(1,minQb) );
            REQUIRE_THROWS_WITH( createDensityQureg(numQb, env), Contains("Too few qubits") );
        }
        SECTION( "available memory" ) {
            
            /* there is no reliable way to force the malloc statements to
             * fail, and hence trigger the matrixInit validation */
            SUCCEED( );
        }
    }
    destroyQuESTEnv(env);
}



TEST_CASE( "createQuESTEnv", "[data_structures]" ) {
    
    /* there is no meaningful way to test this */
    SUCCEED( );
}



TEST_CASE( "createQureg", "[data_structures]" ) {
    
    QuESTEnv env = createQuESTEnv();
    
    // must be at least one amplitude per node
    int minNumQb = calcLog2(env.numRanks);
    if (minNumQb == 0)
        minNumQb = 1;
    
    SECTION( "correctness" ) {
        
        // try 10 valid number of qubits
        int numQb = GENERATE_COPY( range(minNumQb, minNumQb+10) );
        Qureg reg = createQureg(numQb, env);
        
        // ensure elems (CPU and/or GPU) are created, and reg begins in |0>
        QVector ref = QVector(1<<numQb);
        ref[0] = 1; // |0>
        REQUIRE( areEqual(reg, ref) );
        
        destroyQureg(reg, env);
    }
    SECTION( "input validation") {
        
        SECTION( "number of qubits" ) {
            
            int numQb = GENERATE( -1, 0 );
            REQUIRE_THROWS_WITH( createQureg(numQb, env), Contains("Invalid number of qubits") );
        }
        SECTION( "number of amplitudes" ) {

            // too many amplitudes to store in type
            int maxQb = (int) calcLog2(SIZE_MAX);
            REQUIRE_THROWS_WITH( createQureg(maxQb+1, env), Contains("Too many qubits") && Contains("size_t type") );
            
            // too few amplitudes to distribute
            int minQb = GENERATE( range(2,10) );
            env.numRanks = (int) pow(2, minQb);
            int numQb = GENERATE_COPY( range(1,minQb) );
            REQUIRE_THROWS_WITH( createQureg(numQb, env), Contains("Too few qubits") );
        }
        SECTION( "available memory" ) {
            
            /* there is no reliable way to force the malloc statements to
             * fail, and hence trigger the matrixInit validation */
            SUCCEED( );
        }
    }
    destroyQuESTEnv(env);
}



TEST_CASE( "destroyComplexMatrixN", "[data_structures]" ) {
    
    SECTION( "correctness" ) {
        
        /* there is no meaningful way to test this */
        SUCCEED( );
    }
    SECTION( "input validation" ) {
        
        SECTION( "matrix not created" ) {
            
            /* this is an artificial test case since nothing in the QuEST API 
             * automatically sets un-initialised ComplexMatrixN fields to 
             * the NULL pointer.
             */
             ComplexMatrixN m;
             m.real = NULL;
             
             /* the error message is also somewhat unrelated, but oh well 
              */
             REQUIRE_THROWS_WITH( destroyComplexMatrixN(m), Contains("The ComplexMatrixN was not successfully created") );
        }
    }
}



TEST_CASE( "destroyQuESTEnv", "[data_structures]" ) {

    /* there is no meaningful way to test this */
    SUCCEED( );
}



TEST_CASE( "destroyQureg", "[data_structures]" ) {
    
    /* there is no meaningful way to test this.
     * We e.g. cannot check that the pointers are NULL because 
     * they are not updated; this function passes the struct by value,
     * not by reference. We also cannot reliably monitor the 
     * memory used in the heap at runtime.
     */
    SUCCEED( );
}



TEST_CASE( "initComplexMatrixN", "[data_structures]" ) {
    
    /* use of this function is illegal in C++ */
    SUCCEED( );
}
