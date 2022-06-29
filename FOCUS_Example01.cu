#include <iostream>
#include <math.h>
#include "FOCUS_Class.cuh"

using namespace std;

int main(int argc, char **argv){

	// pointer to FOCUS class
	FOCUS *focus;
	focus = new FOCUS;

	// configure focus
	focus->readConfigFiles();

	// allocate phaseSpace
	focus->allocatePhaseSpace();

	// fill phase space
	focus->fillPhaseSpace();

	// copy phase space on GPU memory
	focus->copyPhaseSpaceFromHostToDevice();

	// compute 1D coherence profile
	focus->coherence1D();

	// save
	focus->saveCoherence1D();
	focus->saveParameters();

	// free memory
	focus->freePhaseSpace();

	// closing program
	delete focus;

	return 0;
}