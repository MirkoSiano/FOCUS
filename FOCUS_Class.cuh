#include <iostream>
#include <math.h>
#include <fstream>
#include <string>

using namespace std;

class FOCUS{

	public:
		/********** CLASS CONSTRUCTOR AND DESTRUCTOR **********/

		// Default class constructor
		FOCUS();

		// Class destructor
		~FOCUS();

		/********** PUBLIC METHODS **********/

		// Write main FOCUS parameters
		void setUndGamma(double);
		void setUndK(double);
		void setUndLambdaW(double);
		void setUndNW(int);
		void setUndHarm(int);
		void setUndObsLambda(double);
		void setDim(int);
		void setPxl(double);
		void setZ(double);
		void setPlane(char *);
		void setReferencePoint(char *);
		void setX0(double);
		void setY0(double);
		void setSigmaX(double);
		void setSigmaXP(double);
		void setSigmaY(double);
		void setSigmaYP(double);
		void setEnSpread(double);
		void setNMC(int);

		// Read undulator main parameters
		// ...
		double getSigmaX();
		double getSigmaY();
		// ...

		// Read observation plane parameters
		// ...
		double getPxl();
		double getZ();
		// ...

		// Read electron beam parameters
		// ...
		// write methods here
		// ...

		// Read undulator utils
		// ...
		double getUndLW();
		// ...

		// Read coherence parameters
		// ...
		double getCohLength();
		// ...

		// Read reduced parameters
		// ...
		double getUndHatZ();
		double getUndHatNX();
		double getUndHatDX();
		// ...

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

		// Manage memory
		void allocatePhaseSpace();
		void freePhaseSpace();
		void allocateCoherence1D();
		void freeCoherence1D();
		void allocateCoherence2D();
		void freeCoherence2D();

		// Fill phase space
		void getNMCFromFile(char *);
		void fillPhaseSpace();
		void fillPhaseSpaceFromFile(char *);
		void copyPhaseSpaceFromHostToDevice();

		// Coherence
		void coherence1D();
		void coherence2D();
		void coherenceLength();

		// Saving
		void saveParameters();
		void saveParameters(string);
		void saveCoherence1D();
		void saveCoherence1D(string);
		void saveCoherence2D();

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
		char plane[4];
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

		double *devPhaseSpaceX;
		double *devPhaseSpaceXP;
		double *devPhaseSpaceY;
		double *devPhaseSpaceYP;
		double *devPhaseSpaceE;

		double *coherenceProfile;
		double *coordinatesProfile;

		//double **coherence2DMap;
		double *coherence2DMap;
		double **coordinatesX2D;
		double **coordinatesY2D;

		double cohLength;

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