#include <iostream>
#include <math.h>
#include "FOCUS_Class.cuh"

using namespace std;

int main(int argc, char **argv){

	// external phase space file
	if(argc!=2){
		cout << " Invalid call to "<< argv[0] << endl;
		cout << " Please type: " << argv[0] << " name_of_phase_space_file.txt" << endl;
		return 0;
	}

	// pointer to FOCUS class
	FOCUS *focus;
	focus = new FOCUS;

	// configure focus
	focus->readConfigFiles();

	// get NMC from file
	focus->getNMCFromFile(argv[1]);

	// allocate phase space
	focus->allocatePhaseSpace();

	// fill phase space
	focus->fillPhaseSpaceFromFile(argv[1]);

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