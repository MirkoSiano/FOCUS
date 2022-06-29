#include <iostream>
#include <math.h>
#include <fstream>
#include <string.h>

using namespace std;

class FOCUS{

	public:
		/********** CLASS CONSTRUCTOR AND DESTRUCTOR **********/

		// Default class constructor
		FOCUS();

		// Class destructor
		~FOCUS();

		/********** PUBLIC METHODS **********/

		// Update parameters
		void updateUndulatorParameters();
		void updateObserverParameters();
		void updateReducedParameters();
		void updateParameters();

		// Read configuration files
		void readConfigFileUndulator();
		void readConfigFileElectronBeam();
		void readConfigFileObserver();
		void readConfigFiles();

		// Allocate phase space
		void allocatePhaseSpace();
		void freePhaseSpace();

		// Fill phase space
		void getNMCFromFile(char *);
		void fillPhaseSpace();
		void fillPhaseSpaceFromFile(char *);
		void copyPhaseSpaceFromHostToDevice();

		// Coherence
		void coherence1D();

		// Saving
		void saveParameters();
		void saveCoherence1D();

	private:
		/********** PRIVATE ATTRIBUTES **********/

		// Physical constants
		const double pi = 3.14159265358979323846;
		const double c = 299792458.0; // m/s
		const double hbar = 6.582119569e-22; // MeV s

		// Undulator main parameters
		double undGamma;
		double undK;
		int undNW;
		double undLambdaW; // [mm]
		int undHarm;

		// Observation plane parameters
		double undObsLambda; // [nm]
		int dim;
		double pxl; // [um]
		double z; // [m]
		char plane[10];
		char reference_point[10];
		double x0; // [um]
		double y0; // [um]

		// Electron beam parameters
		double sigmaX; // [um]
		double sigmaXP; // [urad]
		double sigmaY; // [um]
		double sigmaYP; // [urad]
		double enSpread;
		int NMC;

		// Undulator utils
		double undLW; // [m]
		double undFundamentalLambda; // [nm]
		double undFundamentalOmega; // [s-1]
		double undFundamentalPhotonEnergy; // [keV]
		double undResLambda; // [nm]
		double undResOmega; // [s-1]
		double undResPhotonEnergy; // [keV]
		double undObsOmega; // [s-1]
		double undObsPhotonEnergy; // [keV]
		double undC;

		// Reduced parameters
		double undScalingTransverse; // [um]
		double undScalingAngular; // [urad]
		double undScalingEnSpread;
		double undHatZ;
		double undHatC;
		double undHatNX;
		double undHatNY;
		double undHatDX;
		double undHatDY;
		double undHatEnergySpread;

		// FOCUS variables
		double *phaseSpaceX;
		double *phaseSpaceXP;
		double *phaseSpaceY;
		double *phaseSpaceYP;
		double *phaseSpaceE;
		double *coherenceProfile;
		double *coordinatesProfile;
		double *devPhaseSpaceX;
		double *devPhaseSpaceXP;
		double *devPhaseSpaceY;
		double *devPhaseSpaceYP;
		double *devPhaseSpaceE;

		// Seeds for random number generators
		unsigned int seed1;
		unsigned int seed2;

		/********** PRIVATE METHODS **********/

		// Undulator methods
		void computeUndulatorLength();
		void computeFundamentalWavelength();
		void computeFundamentalFrequency();
		void computeFundamentalPhotonEnergy();
		void computeResonantWavelength();
		void computeResonantFrequency();
		void computeResonantPhotonEnergy();

		// Observer methods
		void computeObservedFrequency();
		void computeObservedPhotonEnergy();
		void computeDetuning();

		// Random number generators
		double randomNumber(unsigned int *);
		double boxMuller(unsigned int*,unsigned int*,double,double);

};