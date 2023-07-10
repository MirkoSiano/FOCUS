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

	// allocate memory (phase space and coherence 2D)
	focus->allocatePhaseSpace();
	focus->allocateCoherence2D();

	// fill phase space
	focus->fillPhaseSpace();

	// copy phase space on GPU memory
	focus->copyPhaseSpaceFromHostToDevice();

	// compute 2D coherence map
	focus->coherence2D();

	// save
	focus->saveCoherence2D();
	focus->saveParameters();

	// free memory
	focus->freePhaseSpace();
	focus->freeCoherence2D();

	// closing program
	delete focus;

	return 0;
}