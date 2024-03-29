#include "FOCUS_Class.cuh"
#include "FOCUS_Kernel.cuh"

/************************************************************/
/********** METHODS TO WRITE MAIN FOCUS PARAMETERS **********/
/************************************************************/

void FOCUS::setUndGamma(double value){
	undGamma = value;
}

void FOCUS::setUndK(double value){
	undK = value;
}

void FOCUS::setUndLambdaW(double value){
	undLambdaW = value;
}

void FOCUS::setUndNW(int value){
	undNW = value;
}

void FOCUS::setUndHarm(int value){
	undHarm = value;
}

void FOCUS::setUndObsLambda(double value){
	undObsLambda = value;
}

void FOCUS::setDim(int value){
	dim = value;
}

void FOCUS::setPxl(double value){
	pxl = value;
}

void FOCUS::setZ(double value){
	z = value;
}

void FOCUS::setPlane(char * value){
	strcpy(plane,value);

	// check plane
	if(strcmp(plane,"HOR")!=0){
		if(strcmp(plane,"VER")!=0){
			if(strcmp(plane,"2D")!=0){
				cout << " Invalid variable PLANE" << endl;
				exit(1);
			}
		}
	}
}

void FOCUS::setReferencePoint(char * value){
	strcpy(reference_point,value);

	// check reference point
	if(strcmp(reference_point,"MID_POINT")!=0){
		if(strcmp(reference_point,"OBS_POINT")!=0){
			cout << " Invalid variable REFERENCE POINT" << endl;
			exit(1);
		}
	}
}

void FOCUS::setX0(double value){
	x0 = value;
}

void FOCUS::setY0(double value){
	y0 = value;
}

void FOCUS::setSigmaX(double value){
	sigmaX = value;
}

void FOCUS::setSigmaXP(double value){
	sigmaXP = value;
}

void FOCUS::setSigmaY(double value){
	sigmaY = value;
}

void FOCUS::setSigmaYP(double value){
	sigmaYP = value;
}

void FOCUS::setEnSpread(double value){
	enSpread = value;
}

void FOCUS::setNMC(int value){
	NMC = value;
}

/****************************************************/
/********** READ MAIN UNDULATOR PARAMETERS **********/
/****************************************************/
// ...
// write methods here
// ...

/*******************************************************/
/********** READ OBSERVATION PLANE PARAMETERS **********/
/*******************************************************/
// ...
double FOCUS::getZ(){
	return z;
}

double FOCUS::getPxl(){
	return pxl;
}
// ...

/***************************************************/
/********** READ ELECTRON BEAM PARAMETERS **********/
/***************************************************/
// ...
double FOCUS::getSigmaX(){
	return sigmaX;
}

double FOCUS::getSigmaY(){
	return sigmaY;
}
// ...

/******************************************/
/********** READ UNDULATOR UTILS **********/
/******************************************/
// ...
double FOCUS::getUndLW(){
	return undLW;
};
// ...

/***********************************************/
/********** READ COHERENCE PARAMETERS **********/
/***********************************************/
// ...
double FOCUS::getCohLength(){
	return cohLength;
};
// ...

/*********************************************/
/********** READ REDUCED PARAMETERS **********/
/*********************************************/
// ...
double FOCUS::getUndHatZ(){
	return undHatZ;
}

double FOCUS::getUndHatNX(){
	return undHatNX;
}

double FOCUS::getUndHatDX(){
	return undHatDX;
}
// ...

/***************************************/
/********** CLASS CONSTRUCTOR **********/
/***************************************/

FOCUS::FOCUS(){
	cout << " Initializing FOCUS ... ";

	// Default values for NCD-SWEET undulator @ ALBA
	undGamma = 5831.714527050142;
	undK = 1.5589204391640006;
	undLambdaW = 21.6;
	undNW = 92;
	undHarm = 7;

	// Default values for observation plane
	computeFundamentalWavelength();
	computeResonantWavelength();
	undObsLambda = undResLambda;
	dim = 256;
	pxl = 1.0;
	z = 33.0;
	strcpy(plane,"VER");
	strcpy(reference_point,"MID_POINT");
	x0 = 0.0;
	y0 = 0.0;

	// Default values at NCD-SWEET at nominal coupling
	sigmaX = 130.0;
	sigmaXP = 48.0;
	sigmaY = 6.0;
	sigmaYP = 5.0;
	enSpread = 0.00105;
	NMC = 1000000;

	// Update parameters
	updateParameters();

	// Initialize seeds for random number generators
	seed1 = time(NULL);
	seed2 = seed1-1;

	cout << "done" << endl;
}

/**************************************/
/********** CLASS DESTRUCTOR **********/
/**************************************/

FOCUS::~FOCUS(){
	cout << "\n FOCUS successfully completed!" << endl;
}

/**************************************************************************************/
/********** METHODS TO COMPUTE UNDULATOR AND RADIATION (INTERNAL) PARAMETERS **********/
/**************************************************************************************/

void FOCUS::computeUndulatorLength(){
	undLW = double(undNW)*undLambdaW/1000.0;
}

void FOCUS::computeFundamentalWavelength(){
	undFundamentalLambda = undLambdaW*1000000.0*(1.0+undK*undK/2.0)/(2.0*undGamma*undGamma);
}

void FOCUS::computeFundamentalFrequency(){
	undFundamentalOmega = 2.0*pi*c*1000000000.0/undFundamentalLambda;
}

void FOCUS::computeFundamentalPhotonEnergy(){
	undFundamentalPhotonEnergy = hbar*undFundamentalOmega*1000.0;
}

void FOCUS::computeResonantWavelength(){
	undResLambda = undFundamentalLambda/double(undHarm);
}

void FOCUS::computeResonantFrequency(){
	undResOmega = undFundamentalOmega*double(undHarm);
}

void FOCUS::computeResonantPhotonEnergy(){
	undResPhotonEnergy = undFundamentalPhotonEnergy*double(undHarm);
}

void FOCUS::computeObservedFrequency(){
	undObsOmega = 2.0*pi*c*1000000000.0/undObsLambda;
}

void FOCUS::computeObservedPhotonEnergy(){
	undObsPhotonEnergy = hbar*undObsOmega*1000.0;
}

void FOCUS::computeDetuning(){
	undC = (undObsOmega-undResOmega)/undResOmega;
}

/**************************************************/
/********** METHODS TO UPDATE PARAMETERS **********/
/**************************************************/

void FOCUS::updateUndulatorParameters(){
	// check odd harmonic
	if(undHarm%2==0){
		cout << " Invalid harmonic number (even)" << endl;
		exit(1);
	}
	computeUndulatorLength();
	computeFundamentalWavelength();
	computeFundamentalFrequency();
	computeFundamentalPhotonEnergy();
	computeResonantWavelength();
	computeResonantFrequency();
	computeResonantPhotonEnergy();
}

void FOCUS::updateObserverParameters(){
	// if lambda_obs<0, lambda_obs = lambda_res
	if(undObsLambda<0.0){
		computeFundamentalWavelength();
		computeResonantWavelength();
		undObsLambda = undResLambda;
	}
	computeObservedFrequency();
	computeObservedPhotonEnergy();
	computeDetuning();
}

void FOCUS::updateReducedParameters(){
	undScalingTransverse = sqrt(undLW*c/undObsOmega)*1000000.0;
	undScalingAngular = sqrt(c/(undObsOmega*undLW))*1000000.0;
	undScalingEnSpread = 4.0*pi*double(undNW)*double(undHarm);
	undHatZ = z/undLW;
	undHatC = 2.0*pi*double(undNW)*double(undHarm)*undC;
	undHatNX = sigmaX*sigmaX/(undScalingTransverse*undScalingTransverse);
	undHatNY = sigmaY*sigmaY/(undScalingTransverse*undScalingTransverse);
	undHatDX = sigmaXP*sigmaXP/(undScalingAngular*undScalingAngular);
	undHatDY = sigmaYP*sigmaYP/(undScalingAngular*undScalingAngular);
	undHatEnergySpread = undScalingEnSpread*enSpread;
}

void FOCUS::updateParameters(){
	updateUndulatorParameters();
	updateObserverParameters();
	updateReducedParameters();
}

/*********************************************************/
/********** METHODS TO READ CONFIGURATION FILES **********/
/*********************************************************/

void FOCUS::readConfigFileUndulator(){
	char fileName[50] = "__configUndulator.txt";
	char tmpFlag[50];
	ifstream configFile;

	configFile.open(fileName);

	if(!configFile.is_open()){
		cout << " Unable to open " << fileName << endl;
		exit(1);
	}
	configFile >> tmpFlag >> undGamma;
	configFile >> tmpFlag >> undK;
	configFile >> tmpFlag >> undNW;
	configFile >> tmpFlag >> undLambdaW;
	configFile >> tmpFlag >> undHarm;
	
	configFile.close();
}

void FOCUS::readConfigFileElectronBeam(){
	char fileName[50] = "__configElectronBeam.txt";
	char tmpFlag[50];
	ifstream configFile;

	configFile.open(fileName);

	if(!configFile.is_open()){
		cout << " Unable to open " << fileName << endl;
		exit(1);
	}
	configFile >> tmpFlag >> sigmaX;
	configFile >> tmpFlag >> sigmaXP;
	configFile >> tmpFlag >> sigmaY;
	configFile >> tmpFlag >> sigmaYP;
	configFile >> tmpFlag >> enSpread;
	configFile >> tmpFlag >> NMC;
	
	configFile.close();
}

void FOCUS::readConfigFileObserver(){
	char fileName[50] = "__configObserver.txt";
	char tmpFlag[50];
	ifstream configFile;

	configFile.open(fileName);

	if(!configFile.is_open()){
		cout << " Unable to open " << fileName << endl;
		exit(1);
	}
	configFile >> tmpFlag >> dim;
	configFile >> tmpFlag >> pxl;
	configFile >> tmpFlag >> z;
	configFile >> tmpFlag >> undObsLambda;
	configFile >> tmpFlag >> plane;
	configFile >> tmpFlag >> reference_point;
	configFile >> tmpFlag >> x0;
	configFile >> tmpFlag >> y0;
	
	configFile.close();

	// check plane
	if(strcmp(plane,"HOR")!=0){
		if(strcmp(plane,"VER")!=0){
			if(strcmp(plane,"2D")!=0){
				cout << " Invalid variable PLANE" << endl;
				exit(1);
			}
		}
	}

	// check reference point
	if(strcmp(reference_point,"MID_POINT")!=0){
		if(strcmp(reference_point,"OBS_POINT")!=0){
			cout << " Invalid variable REFERENCE POINT" << endl;
			exit(1);
		}
	}
}

void FOCUS::readConfigFiles(){
	cout << " Configuring FOCUS ... ";

	readConfigFileUndulator();
	readConfigFileElectronBeam();
	readConfigFileObserver();
	updateParameters();

	cout << "done" << endl;
}

/****************************************************************************/
/********** METHODS TO MANAGE THE ELECTRON BEAM PHASE SPACE MEMORY **********/
/****************************************************************************/

void FOCUS::allocatePhaseSpace(){
	cout << " Allocating phase space memory ... ";

	phaseSpaceX = new double[NMC];
	phaseSpaceXP = new double[NMC];
	phaseSpaceY = new double[NMC];
	phaseSpaceYP = new double[NMC];
	phaseSpaceE = new double[NMC];
	cudaMalloc(&devPhaseSpaceX,NMC*sizeof(double));
	cudaMalloc(&devPhaseSpaceXP,NMC*sizeof(double));
	cudaMalloc(&devPhaseSpaceY,NMC*sizeof(double));
	cudaMalloc(&devPhaseSpaceYP,NMC*sizeof(double));
	cudaMalloc(&devPhaseSpaceE,NMC*sizeof(double));

	cout << "done" << endl;
}

void FOCUS::freePhaseSpace(){
	cout << " Freeing allocated phase space memory ... ";

	delete [] phaseSpaceX;
	delete [] phaseSpaceXP;
	delete [] phaseSpaceY;
	delete [] phaseSpaceYP;
	delete [] phaseSpaceE;
	cudaFree(devPhaseSpaceX);
	cudaFree(devPhaseSpaceXP);
	cudaFree(devPhaseSpaceY);
	cudaFree(devPhaseSpaceYP);
	cudaFree(devPhaseSpaceE);

	cout << "done" << endl;
}

void FOCUS::copyPhaseSpaceFromHostToDevice(){
	cout << " Moving phase space from CPU to GPU ... ";

	cudaMemcpy(devPhaseSpaceX,phaseSpaceX,NMC*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(devPhaseSpaceXP,phaseSpaceXP,NMC*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(devPhaseSpaceY,phaseSpaceY,NMC*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(devPhaseSpaceYP,phaseSpaceYP,NMC*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(devPhaseSpaceE,phaseSpaceE,NMC*sizeof(double),cudaMemcpyHostToDevice);

	cout << "done" << endl;
}

/********************************************************/
/********** METHODS TO MANAGE COHERENCE MEMORY **********/
/********************************************************/

void FOCUS::allocateCoherence1D(){
	cout << " Allocating coherence 1D memory ... ";

	coherenceProfile = new double[dim];
	coordinatesProfile = new double[dim];

	cout << "done" << endl;
}

void FOCUS::freeCoherence1D(){
	cout << " Freeing allocated coherence 1D memory ... ";

	delete [] coherenceProfile;
	delete [] coordinatesProfile;

	cout << "done" << endl;
}

void FOCUS::allocateCoherence2D(){
	cout << " Allocating coherence 2D memory ... ";

	//coherence2DMap = new double *[dim];
	coherence2DMap = new double[dim*dim];
	coordinatesX2D = new double *[dim];
	coordinatesY2D = new double *[dim];

	for(int i=0;i<dim;i++){
		//coherence2DMap[i] = new double[dim];
		coordinatesX2D[i] = new double[dim];
		coordinatesY2D[i] = new double[dim];
	}

	cout << "done" << endl;
}

void FOCUS::freeCoherence2D(){
	cout << " Freeing allocated coherence 2D memory ... ";

	for(int i=0;i<dim;i++){
		//delete [] coherence2DMap[i];
		delete [] coordinatesX2D[i];
		delete [] coordinatesY2D[i];
	}

	delete [] coherence2DMap;
	delete [] coordinatesX2D;
	delete [] coordinatesY2D;

	cout << "done" << endl;
}

/**********************************************************/
/********** METHODS FOR RANDOM NUMBER GENERATORS **********/
/**********************************************************/

double FOCUS::randomNumber(unsigned int *seed){
	const unsigned int m = (int)pow(2.,32);
	const unsigned int a = 1664525;
	const unsigned int b = 1013904223;
	unsigned int n = ((*seed)*a+b)%m;
	*seed = n;
	return (double)n/(m-1);
}

double FOCUS::boxMuller(unsigned int *s1, unsigned int *s2, double mean, double stdev){
	if((*s1)==(*s2)){
		(*s2)++;
	}
	//const double pi = 3.14159265358979323846;
	double u1 = randomNumber(s1);
	while(u1==0.0){
		u1 = randomNumber(s1);
	}
	double u2 = randomNumber(s2);
	return stdev*(sqrt(-2.*log(u1))*cos(2.*pi*u2))+mean;
}

/***********************************************************************/
/********** METHODS TO POPULATE THE ELECTRON BEAM PHASE SPACE **********/
/***********************************************************************/

void FOCUS::getNMCFromFile(char *fileName){
	ifstream phaseSpaceFile;
	double tmp;
	int n = 0;

	phaseSpaceFile.open(fileName);

	if(!phaseSpaceFile.is_open()){
		cout << " Unable to open " << fileName << endl;
		exit(1);
	}
	while(!phaseSpaceFile.eof()){
		phaseSpaceFile >> tmp >> tmp >> tmp >> tmp >> tmp;
		n ++;
	}
	
	phaseSpaceFile.close();

	NMC = n;
}

void FOCUS::fillPhaseSpace(){
	cout << " Filling phase space ... ";

	for(int i=0;i<NMC;i++){
		phaseSpaceX[i] = boxMuller(&seed1,&seed2,0,sigmaX)/undScalingTransverse;
		phaseSpaceXP[i] = boxMuller(&seed1,&seed2,0,sigmaXP)/undScalingAngular;
		phaseSpaceY[i] = boxMuller(&seed1,&seed2,0,sigmaY)/undScalingTransverse;
		phaseSpaceYP[i] = boxMuller(&seed1,&seed2,0,sigmaYP)/undScalingAngular;;
		phaseSpaceE[i] = boxMuller(&seed1,&seed2,0,enSpread)*undScalingEnSpread;
	}

	cout << "done" << endl;
}

void FOCUS::fillPhaseSpaceFromFile(char *fileName){
	cout << " Filling phase space (from file " << fileName << ") ... ";

	ifstream phaseSpaceFile;

	phaseSpaceFile.open(fileName);

	if(!phaseSpaceFile.is_open()){
		cout << " Unable to open " << fileName << endl;
		exit(1);
	}
	for(int i=0;i<NMC;i++){
		//phaseSpaceFile >> phaseSpaceX[i];
		//phaseSpaceFile >> phaseSpaceXP[i];
		//phaseSpaceFile >> phaseSpaceY[i];
		//phaseSpaceFile >> phaseSpaceYP[i];
		//phaseSpaceFile >> phaseSpaceE[i];

		phaseSpaceFile >> phaseSpaceX[i] >> phaseSpaceXP[i] >> phaseSpaceY[i] >> phaseSpaceYP[i] >> phaseSpaceE[i];
	}
	
	phaseSpaceFile.close();

	for(int i=0;i<NMC;i++){
		phaseSpaceX[i] = phaseSpaceX[i]*1000000.0/undScalingTransverse;
		phaseSpaceXP[i] = phaseSpaceXP[i]*1000000.0/undScalingAngular;
		phaseSpaceY[i] = phaseSpaceY[i]*1000000.0/undScalingTransverse;
		phaseSpaceYP[i] = phaseSpaceYP[i]*1000000.0/undScalingAngular;
		phaseSpaceE[i] = phaseSpaceE[i]*undScalingEnSpread;
	}

	cout << "done" << endl;
}

/**************************************************/
/********** COMPUTE COHERENCE 1D PROFILE **********/
/**************************************************/

void FOCUS::coherence1D(){
	 if(strcmp(plane,"2D")==0){
		cout << " Invalid variable PLANE for function coherence1D";
		exit(1);
	}

	cout << " Computing 1D coherence on GPU ... ";

	// Kernel configuration parameters
	int maxThreadsPerBlock = 1024;
	int numThreads = 256;
	int numBlocks = (NMC+numThreads-1)/numThreads;

	// Parameters on CPU
	int center = dim/2;
	double convertFromCoordinateToTheta = pxl/(z*undScalingAngular);
	double thetax0 = x0/(z*undScalingAngular);
	double thetay0 = y0/(z*undScalingAngular);
	double commonParam[5];
	double *deltaTheta;
	double *arrayThetaX1;
	double *arrayThetaX2;
	double *arrayThetaY1;
	double *arrayThetaY2;
	double theta1[2];
	double theta2[2];

	// Parameters on GPU
	double *devCommonParam;
	double *devTheta1;
	double *devTheta2;
	double *devCoherence;
	double *devRe1;
	double *devIm1;
	double *devRe2;
	double *devIm2;
	double *devReCSD;
	double *devImCSD;
	double *devI1;
	double *devI2;

	// Setting commonParam
	commonParam[0] = undHatZ;
	commonParam[1] = undHatZ/2.0;
	commonParam[2] = undHatC/2.0;
	commonParam[3] = undHatZ*undHatZ/(2.0*undHatZ-1.0);
	commonParam[4] = undHatZ*undHatZ/(2.0*undHatZ+1.0);

	// Allocate memory
	deltaTheta = new double[dim];
	arrayThetaX1 = new double[dim];
	arrayThetaX2 = new double[dim];
	arrayThetaY1 = new double[dim];
	arrayThetaY2 = new double[dim];
	
	cudaMalloc(&devCommonParam,5*sizeof(double));
	cudaMalloc(&devTheta1,2*sizeof(double));
	cudaMalloc(&devTheta2,2*sizeof(double));
	cudaMalloc(&devCoherence,dim*sizeof(double));
	cudaMalloc(&devRe1,NMC*sizeof(double));
	cudaMalloc(&devIm1,NMC*sizeof(double));
	cudaMalloc(&devRe2,NMC*sizeof(double));
	cudaMalloc(&devIm2,NMC*sizeof(double));
	cudaMalloc(&devReCSD,1*sizeof(double));
	cudaMalloc(&devImCSD,1*sizeof(double));
	cudaMalloc(&devI1,1*sizeof(double));
	cudaMalloc(&devI2,1*sizeof(double));

	// Copy commonParam from Host to Device
	cudaMemcpy(devCommonParam,&commonParam,5*sizeof(double),cudaMemcpyHostToDevice);

	// Compute values for deltaTheta and store results
	for(int i=0;i<dim;i++){
		deltaTheta[i] = double(i-center)*convertFromCoordinateToTheta;
		coordinatesProfile[i] = double(i-center)*pxl;
	}

	// Compute theta1 and theta2 for different modes
	if(strcmp(plane,"HOR")==0){
		if(strcmp(reference_point,"MID_POINT")==0){
			for(int i=0;i<dim;i++){
				arrayThetaX1[i] = thetax0-deltaTheta[i]/2.0;
				arrayThetaX2[i] = thetax0+deltaTheta[i]/2.0;
				arrayThetaY1[i] = thetay0;
				arrayThetaY2[i] = thetay0;
			}
		} else if(strcmp(reference_point,"OBS_POINT")==0){
			for(int i=0;i<dim;i++){
				arrayThetaX1[i] = thetax0;
				arrayThetaX2[i] = thetax0+deltaTheta[i];
				arrayThetaY1[i] = thetay0;
				arrayThetaY2[i] = thetay0;
			}
		}
	} else if(strcmp(plane,"VER")==0){
		if(strcmp(reference_point,"MID_POINT")==0){
			for(int i=0;i<dim;i++){
				arrayThetaX1[i] = thetax0;
				arrayThetaX2[i] = thetax0;
				arrayThetaY1[i] = thetay0-deltaTheta[i]/2.0;
				arrayThetaY2[i] = thetay0+deltaTheta[i]/2.0;
			}
		} else if(strcmp(reference_point,"OBS_POINT")==0){
			for(int i=0;i<dim;i++){
				arrayThetaX1[i] = thetax0;
				arrayThetaX2[i] = thetax0;
				arrayThetaY1[i] = thetay0;
				arrayThetaY2[i] = thetay0+deltaTheta[i];
			}
		}
	}

	// Cycle through observation points
	for(int i=0;i<dim;i++){

		// Set theta1 and theta2
		theta1[0] = arrayThetaX1[i];
		theta1[1] = arrayThetaY1[i];
		theta2[0] = arrayThetaX2[i];
		theta2[1] = arrayThetaY2[i];

		// Copy theta1 and theta2 from CPU to GPU
		cudaMemcpy(devTheta1,theta1,2*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(devTheta2,theta2,2*sizeof(double),cudaMemcpyHostToDevice);

		// Compute electric fields at observation points
		computeElectricField_FarField<<<numBlocks,numThreads>>>(devTheta1,devRe1,devIm1,devCommonParam,devPhaseSpaceX,devPhaseSpaceXP,devPhaseSpaceY,devPhaseSpaceYP,devPhaseSpaceE,NMC);
		computeElectricField_FarField<<<numBlocks,numThreads>>>(devTheta2,devRe2,devIm2,devCommonParam,devPhaseSpaceX,devPhaseSpaceXP,devPhaseSpaceY,devPhaseSpaceYP,devPhaseSpaceE,NMC);
		cudaDeviceSynchronize();

		// Compute individual terms for CSD, I1 and I2
		computeSpectralDegreeOfCoherence<<<numBlocks,numThreads>>>(devRe1,devIm1,devRe2,devIm2,NMC);
		cudaDeviceSynchronize();

		// Add array elements
		computeSum<<<1,maxThreadsPerBlock>>>(devRe1,devReCSD,NMC);
		computeSum<<<1,maxThreadsPerBlock>>>(devIm1,devImCSD,NMC);
		computeSum<<<1,maxThreadsPerBlock>>>(devRe2,devI1,NMC);
		computeSum<<<1,maxThreadsPerBlock>>>(devIm2,devI2,NMC);
		cudaDeviceSynchronize();

		// Compute |SDC|
		computeModulusSDC<<<1,1>>>(&(devCoherence[i]),devReCSD,devImCSD,devI1,devI2);
	}

	// Copy results from GPU to CPU
	cudaMemcpy(coherenceProfile,devCoherence,dim*sizeof(double),cudaMemcpyDeviceToHost);

	// Free memory
	delete [] deltaTheta;
	delete [] arrayThetaX1;
	delete [] arrayThetaX2;
	delete [] arrayThetaY1;
	delete [] arrayThetaY2;
	cudaFree(devCommonParam);
	cudaFree(devTheta1);
	cudaFree(devTheta2);
	cudaFree(devCoherence);
	cudaFree(devRe1);
	cudaFree(devIm1);
	cudaFree(devRe2);
	cudaFree(devIm2);
	cudaFree(devReCSD);
	cudaFree(devImCSD);
	cudaFree(devI1);
	cudaFree(devI2);

	cout << "done" << endl;
}

/**********************************************/
/********** COMPUTE COHERENCE 2D MAP **********/
/**********************************************/

void FOCUS::coherence2D(){
	cout << " Computing 2D coherence on GPU ... ";

	// Kernel configuration parameters
	int maxThreadsPerBlock = 1024;
	int numThreads = 256;
	int numBlocks = (NMC+numThreads-1)/numThreads;

	// Parameters on CPU
	int center = dim/2;
	double convertFromCoordinateToTheta = pxl/(z*undScalingAngular);
	double thetax0 = x0/(z*undScalingAngular);
	double thetay0 = y0/(z*undScalingAngular);
	double commonParam[5];
	double **deltaThetaX;
	double **deltaThetaY;
	double **arrayThetaX1;
	double **arrayThetaX2;
	double **arrayThetaY1;
	double **arrayThetaY2;
	double theta1[2];
	double theta2[2];

	// Parameters on GPU
	double *devCommonParam;
	double *devTheta1;
	double *devTheta2;
	double *devCoherence;
	double *devRe1;
	double *devIm1;
	double *devRe2;
	double *devIm2;
	double *devReCSD;
	double *devImCSD;
	double *devI1;
	double *devI2;

	// Setting commonParam
	commonParam[0] = undHatZ;
	commonParam[1] = undHatZ/2.0;
	commonParam[2] = undHatC/2.0;
	commonParam[3] = undHatZ*undHatZ/(2.0*undHatZ-1.0);
	commonParam[4] = undHatZ*undHatZ/(2.0*undHatZ+1.0);

	// Allocate memory
	deltaThetaX = new double *[dim];
	deltaThetaY = new double *[dim];
	arrayThetaX1 = new double *[dim];
	arrayThetaX2 = new double *[dim];
	arrayThetaY1 = new double *[dim];
	arrayThetaY2 = new double *[dim];
	for(int i=0;i<dim;i++){
		deltaThetaX[i] = new double[dim];
		deltaThetaY[i] = new double[dim];
		arrayThetaX1[i] = new double[dim];
		arrayThetaX2[i] = new double[dim];
		arrayThetaY1[i] = new double[dim];
		arrayThetaY2[i] = new double[dim];
	}
	
	cudaMalloc(&devCommonParam,5*sizeof(double));
	cudaMalloc(&devTheta1,2*sizeof(double));
	cudaMalloc(&devTheta2,2*sizeof(double));
	cudaMalloc(&devCoherence,dim*dim*sizeof(double));
	cudaMalloc(&devRe1,NMC*sizeof(double));
	cudaMalloc(&devIm1,NMC*sizeof(double));
	cudaMalloc(&devRe2,NMC*sizeof(double));
	cudaMalloc(&devIm2,NMC*sizeof(double));
	cudaMalloc(&devReCSD,1*sizeof(double));
	cudaMalloc(&devImCSD,1*sizeof(double));
	cudaMalloc(&devI1,1*sizeof(double));
	cudaMalloc(&devI2,1*sizeof(double));

	// Copy commonParam from Host to Device
	cudaMemcpy(devCommonParam,&commonParam,5*sizeof(double),cudaMemcpyHostToDevice);

	// Compute values for deltaTheta and store results
	for(int i=0;i<dim;i++){
		for(int j=0;j<dim;j++){
			deltaThetaX[i][j] = double(i-center)*convertFromCoordinateToTheta;
			deltaThetaY[i][j] = double(j-center)*convertFromCoordinateToTheta;
			coordinatesX2D[i][j] = double(i-center)*pxl;
			coordinatesY2D[i][j] = double(j-center)*pxl;
		}
	}

	// Compute theta1 and theta2 for different modes
	if(strcmp(plane,"HOR")==0){
		for(int i=0;i<dim;i++){
			for(int j=0;j<dim;j++){
				arrayThetaX1[i][j] = deltaThetaX[i][j];
				arrayThetaX2[i][j] = deltaThetaY[i][j];
				arrayThetaY1[i][j] = thetax0;
				arrayThetaY2[i][j] = thetay0;
			}
		}
	} else if(strcmp(plane,"VER")==0){
		for(int i=0;i<dim;i++){
			for(int j=0;j<dim;j++){
				arrayThetaX1[i][j] = thetax0;
				arrayThetaX2[i][j] = thetay0;
				arrayThetaY1[i][j] = deltaThetaX[i][j];
				arrayThetaY2[i][j] = deltaThetaY[i][j];
			}
		}
	} else{
		if(strcmp(reference_point,"MID_POINT")==0){
			for(int i=0;i<dim;i++){
				for(int j=0;j<dim;j++){
					arrayThetaX1[i][j] = thetax0-deltaThetaX[i][j]/2.0;
					arrayThetaX2[i][j] = thetax0+deltaThetaX[i][j]/2.0;
					arrayThetaY1[i][j] = thetay0-deltaThetaY[i][j]/2.0;
					arrayThetaY2[i][j] = thetay0+deltaThetaY[i][j]/2.0;
				}
			}
		} else if(strcmp(reference_point,"OBS_POINT")==0){
			for(int i=0;i<dim;i++){
				for(int j=0;j<dim;j++){
					arrayThetaX1[i][j] = thetax0;
					arrayThetaX2[i][j] = thetax0+deltaThetaX[i][j];
					arrayThetaY1[i][j] = thetay0;
					arrayThetaY2[i][j] = thetay0+deltaThetaY[i][j];
				}
			}
		}
	}

	// Cycle through observation points
	for(int i=0;i<dim;i++){
		for(int j=0;j<dim;j++){

			// Set theta1 and theta2
			theta1[0] = arrayThetaX1[i][j];
			theta1[1] = arrayThetaY1[i][j];
			theta2[0] = arrayThetaX2[i][j];
			theta2[1] = arrayThetaY2[i][j];

			// Copy theta1 and theta2 from CPU to GPU
			cudaMemcpy(devTheta1,theta1,2*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(devTheta2,theta2,2*sizeof(double),cudaMemcpyHostToDevice);

			// Compute electric fields at observation points
			computeElectricField_FarField<<<numBlocks,numThreads>>>(devTheta1,devRe1,devIm1,devCommonParam,devPhaseSpaceX,devPhaseSpaceXP,devPhaseSpaceY,devPhaseSpaceYP,devPhaseSpaceE,NMC);
			computeElectricField_FarField<<<numBlocks,numThreads>>>(devTheta2,devRe2,devIm2,devCommonParam,devPhaseSpaceX,devPhaseSpaceXP,devPhaseSpaceY,devPhaseSpaceYP,devPhaseSpaceE,NMC);
			cudaDeviceSynchronize();

			// Compute individual terms for CSD, I1 and I2
			computeSpectralDegreeOfCoherence<<<numBlocks,numThreads>>>(devRe1,devIm1,devRe2,devIm2,NMC);
			cudaDeviceSynchronize();

			// Add array elements
			computeSum<<<1,maxThreadsPerBlock>>>(devRe1,devReCSD,NMC);
			computeSum<<<1,maxThreadsPerBlock>>>(devIm1,devImCSD,NMC);
			computeSum<<<1,maxThreadsPerBlock>>>(devRe2,devI1,NMC);
			computeSum<<<1,maxThreadsPerBlock>>>(devIm2,devI2,NMC);
			cudaDeviceSynchronize();

			// Compute |SDC|
			computeModulusSDC<<<1,1>>>(&(devCoherence[i*dim+j]),devReCSD,devImCSD,devI1,devI2);
		}
	}

	// Copy results from GPU to CPU
	cudaMemcpy(coherence2DMap,devCoherence,dim*dim*sizeof(double),cudaMemcpyDeviceToHost);

	// Free memory
	for(int i=0;i<dim;i++){
		delete [] deltaThetaX[i];
		delete [] deltaThetaY[i];
		delete [] arrayThetaX1[i];
		delete [] arrayThetaX2[i];
		delete [] arrayThetaY1[i];
		delete [] arrayThetaY2[i];
	}
	delete [] deltaThetaX;
	delete [] deltaThetaY;
	delete [] arrayThetaX1;
	delete [] arrayThetaX2;
	delete [] arrayThetaY1;
	delete [] arrayThetaY2;
	cudaFree(devCommonParam);
	cudaFree(devTheta1);
	cudaFree(devTheta2);
	cudaFree(devCoherence);
	cudaFree(devRe1);
	cudaFree(devIm1);
	cudaFree(devRe2);
	cudaFree(devIm2);
	cudaFree(devReCSD);
	cudaFree(devImCSD);
	cudaFree(devI1);
	cudaFree(devI2);

	cout << "done" << endl;
}

/*********************************************************/
/********** COMPUTE TRANSVERSE COHERENCE LENGTH **********/
/*********************************************************/

void FOCUS::coherenceLength(){
	cout << " Computing transverse coherence length ... ";

	double sum = 0.;
	double temp;

	for(int i=0;i<dim;i++){
		temp = coherenceProfile[i]*coherenceProfile[i]/3;
		if(i!=0 && i!=(dim-1)){
			if(i%2==0){
				temp *= 2.;
			} else{
				temp *= 4.;
			}
		}
		sum += temp;
	}
	cohLength = sum*pxl;

	cout << "done" << endl;
}

/*********************************************/
/********** METHODS TO SAVE RESULTS **********/
/*********************************************/

void FOCUS::saveParameters(){
	cout << " Saving parameters ... ";

	char fileName[50] = "outputLog.txt";
	ofstream outputFile;

	outputFile.open(fileName);

	if(!outputFile.is_open()){
		cout << " Unable to open " << fileName << endl;
		exit(1);
	}

	outputFile << "/********** UNDULATOR PARAMETERS (INPUT) **********/" << endl;
	outputFile << " GAMMA: " << undGamma << endl;
	outputFile << " K: " << undK << endl;
	outputFile << " PERIOD_NUM: " << undNW << endl;
	outputFile << " PERIOD_LEN_[mm]: " << undLambdaW << endl;
	outputFile << " HARMONIC:" << undHarm << endl;
	outputFile << endl;
	outputFile << "/********** ELECTRON BEAM PARAMETERS (INPUT) **********/" << endl;
	outputFile << " SIGMAX_[um]: " << sigmaX << endl;
	outputFile << " SIGMAXP_[urad]: " << sigmaXP << endl;
	outputFile << " SIGMAY_[um]: " << sigmaY << endl;
	outputFile << " SIGMAYP_[um]: " << sigmaYP << endl;
	outputFile << " ENERGY_SPREAD: " << enSpread << endl;
	outputFile << " NMC: " << NMC << endl;
	outputFile << endl;
	outputFile << "/********** OBSERVER PARAMETERS (INPUT) **********/" << endl;
	outputFile << " MESH_SIZE: " << dim << endl;
	outputFile << " PIXEL_SIZE_[um]: " << pxl << endl;
	outputFile << " Z_[m]: " << z << endl;
	outputFile << " LAMBDA_OBS_[nm]: " << undObsLambda << endl;
	outputFile << " PLANE: " << plane << endl;
	outputFile << " REFERENCE_POINT: " << reference_point << endl;
	outputFile << " X0: " << x0 << endl;
	outputFile << " Y0: " << y0 << endl;
	outputFile << endl;
	outputFile << "/********** UNDULATOR PARAMETERS (DERIVED) **********/" << endl;
	outputFile << " UNDULATOR_LENGTH_[m]: " << undLW << endl;
	outputFile << " 1st_HARMONIC_LAMBDA[nm]: " << undFundamentalLambda << endl;
	outputFile << " 1st_HARMONIC_PHOTON_ENERGY_[keV]: " << undFundamentalPhotonEnergy << endl;
	outputFile << " RESONANT_LAMBDA_[nm]: " << undResLambda << endl;
	outputFile << " RESONANT_PHOTON_ENERGY_[keV]: " << undResPhotonEnergy << endl;
	outputFile << " OBSERVED_PHOTON_ENERGY_[keV]: " << undObsPhotonEnergy << endl;
	outputFile << " DETUNING: " << undC << endl;
	outputFile << endl;
	outputFile << "/********** REDUCED PARAMETERS **********/" << endl;
	outputFile << " SCALING_TRANSVERSE_[um]: " << undScalingTransverse << endl;
	outputFile << " SCALING_ANGULAR_[urad]: " << undScalingAngular << endl;
	outputFile << " HAT_Z: " << undHatZ << endl;
	outputFile << " HAT_C: " << undHatC << endl;
	outputFile << " HAT_N_X: " << undHatNX << endl;
	outputFile << " HAT_D_X: " << undHatDX << endl;
	outputFile << " HAT_N_Y: " << undHatNY << endl;
	outputFile << " HAT_D_Y: " << undHatDY << endl;
	outputFile << " HAT_ENERGY_SPREAD: " << undHatEnergySpread << endl;
	
	outputFile.close();

	cout << "done" << endl;
}

void FOCUS::saveParameters(string fileName){
	cout << " Saving parameters ... ";

	ofstream outputFile;

	outputFile.open(fileName);

	if(!outputFile.is_open()){
		cout << " Unable to open " << fileName << endl;
		exit(1);
	}

	outputFile << "/********** UNDULATOR PARAMETERS (INPUT) **********/" << endl;
	outputFile << " GAMMA: " << undGamma << endl;
	outputFile << " K: " << undK << endl;
	outputFile << " PERIOD_NUM: " << undNW << endl;
	outputFile << " PERIOD_LEN_[mm]: " << undLambdaW << endl;
	outputFile << " HARMONIC:" << undHarm << endl;
	outputFile << endl;
	outputFile << "/********** ELECTRON BEAM PARAMETERS (INPUT) **********/" << endl;
	outputFile << " SIGMAX_[um]: " << sigmaX << endl;
	outputFile << " SIGMAXP_[urad]: " << sigmaXP << endl;
	outputFile << " SIGMAY_[um]: " << sigmaY << endl;
	outputFile << " SIGMAYP_[um]: " << sigmaYP << endl;
	outputFile << " ENERGY_SPREAD: " << enSpread << endl;
	outputFile << " NMC: " << NMC << endl;
	outputFile << endl;
	outputFile << "/********** OBSERVER PARAMETERS (INPUT) **********/" << endl;
	outputFile << " MESH_SIZE: " << dim << endl;
	outputFile << " PIXEL_SIZE_[um]: " << pxl << endl;
	outputFile << " Z_[m]: " << z << endl;
	outputFile << " LAMBDA_OBS_[nm]: " << undObsLambda << endl;
	outputFile << " PLANE: " << plane << endl;
	outputFile << " REFERENCE_POINT: " << reference_point << endl;
	outputFile << " X0: " << x0 << endl;
	outputFile << " Y0: " << y0 << endl;
	outputFile << endl;
	outputFile << "/********** UNDULATOR PARAMETERS (DERIVED) **********/" << endl;
	outputFile << " UNDULATOR_LENGTH_[m]: " << undLW << endl;
	outputFile << " 1st_HARMONIC_LAMBDA[nm]: " << undFundamentalLambda << endl;
	outputFile << " 1st_HARMONIC_PHOTON_ENERGY_[keV]: " << undFundamentalPhotonEnergy << endl;
	outputFile << " RESONANT_LAMBDA_[nm]: " << undResLambda << endl;
	outputFile << " RESONANT_PHOTON_ENERGY_[keV]: " << undResPhotonEnergy << endl;
	outputFile << " OBSERVED_PHOTON_ENERGY_[keV]: " << undObsPhotonEnergy << endl;
	outputFile << " DETUNING: " << undC << endl;
	outputFile << endl;
	outputFile << "/********** REDUCED PARAMETERS **********/" << endl;
	outputFile << " SCALING_TRANSVERSE_[um]: " << undScalingTransverse << endl;
	outputFile << " SCALING_ANGULAR_[urad]: " << undScalingAngular << endl;
	outputFile << " HAT_Z: " << undHatZ << endl;
	outputFile << " HAT_C: " << undHatC << endl;
	outputFile << " HAT_N_X: " << undHatNX << endl;
	outputFile << " HAT_D_X: " << undHatDX << endl;
	outputFile << " HAT_N_Y: " << undHatNY << endl;
	outputFile << " HAT_D_Y: " << undHatDY << endl;
	outputFile << " HAT_ENERGY_SPREAD: " << undHatEnergySpread << endl;
	
	outputFile.close();

	cout << "done" << endl;
}

void FOCUS::saveCoherence1D(){
	cout << " Saving 1D coherence ... ";

	char fileName[50] = "outputCoherence1D.txt";
	ofstream outputFile;

	outputFile.open(fileName);

	if(!outputFile.is_open()){
		cout << " Unable to open " << fileName << endl;
		exit(1);
	}

	for(int i=0;i<dim;i++){
		outputFile << coordinatesProfile[i] << "\t" << coherenceProfile[i] << endl;
	}
	
	outputFile.close();

	cout << "done" << endl;
}

void FOCUS::saveCoherence1D(string fileName){
	cout << " Saving 1D coherence ... ";

	ofstream outputFile;

	outputFile.open(fileName);

	if(!outputFile.is_open()){
		cout << " Unable to open " << fileName << endl;
		exit(1);
	}

	for(int i=0;i<dim;i++){
		outputFile << coordinatesProfile[i] << "\t" << coherenceProfile[i] << endl;
	}
	
	outputFile.close();

	cout << "done" << endl;
}

void FOCUS::saveCoherence2D(){
	cout << " Saving 2D coherence ... ";

	char fileName[50] = "outputCoherence2D.txt";
	ofstream outputFile;

	outputFile.open(fileName);

	if(!outputFile.is_open()){
		cout << " Unable to open " << fileName << endl;
		exit(1);
	}

	for(int i=0;i<dim;i++){
		for(int j=0;j<dim;j++){
			outputFile << coordinatesX2D[i][j] << "\t" << coordinatesY2D[i][j] << "\t" << coherence2DMap[i*dim+j] << endl;
		}
	}
	
	outputFile.close();

	cout << "done" << endl;
}