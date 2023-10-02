#include <iostream>
#include <math.h>
#include "FOCUS_Class.cuh"
#include <string>

using namespace std;

int main(int argc, char **argv){

	// pointer to FOCUS class
	FOCUS *focus;
	focus = new FOCUS;

	// configure focus
	focus->readConfigFiles();

	// allocate memory (phase space and coherence 1D)
	focus->allocatePhaseSpace();
	focus->allocateCoherence1D();

	// fill phase space
	focus->fillPhaseSpace();

	// copy phase space on GPU memory
	focus->copyPhaseSpaceFromHostToDevice();
		
	// compute 1D coherence profile
	focus->coherence1D();

	// compute transverse coherence length
	focus->coherenceLength();

	// print results
	cout << "\n    Transverse coherence length: " << focus->getCohLength() << " um \n" << endl;

	// free memory
	focus->freePhaseSpace();
	focus->freeCoherence1D();

	// closing program
	delete focus;

	return 0;
}