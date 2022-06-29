#include "FOCUS_Kernel.cuh"

__global__ void computeElectricField_FarField(double *undTheta, double *undRe, double *undIm, double *undCommonParam, double *undPhaseSpaceX, double *undPhaseSpaceXP, double *undPhaseSpaceY, double *undPhaseSpaceYP, double *undPhaseSpaceE, int N){
	// Parameters
	int start;
	int stride;
	int particle;
	double undThetaXHat;
	double undThetaYHat;
	double undZeta2;
	double undPhiS;
	double undArg;
	double __SINC;

	// Configure grid-stride loop
	start = blockIdx.x*blockDim.x+threadIdx.x;
	stride = blockDim.x*gridDim.x;

	// Grid-stride loop (each electron contribution si computed in parallel)
	for(particle=start;particle<N;particle+=stride){
		undThetaXHat = undTheta[0]-undPhaseSpaceX[particle]/undCommonParam[0];
		undThetaYHat = undTheta[1]-undPhaseSpaceY[particle]/undCommonParam[0];
		undPhiS = (undThetaXHat*undThetaXHat+undThetaYHat*undThetaYHat)*undCommonParam[1];
		undThetaXHat -= undPhaseSpaceXP[particle];
		undThetaYHat -= undPhaseSpaceYP[particle];
		undZeta2 = undThetaXHat*undThetaXHat+undThetaYHat*undThetaYHat;
		undArg = undCommonParam[2]-0.5*undPhaseSpaceE[particle]+0.25*undZeta2;
		if(undArg==0.0){
			__SINC = 1.;
		} else{
			__SINC = (sin(undArg))/undArg;
		}
		undRe[particle] = cos(undPhiS)*__SINC;
		undIm[particle] = sin(undPhiS)*__SINC;
	}
}

__global__ void computeSpectralDegreeOfCoherence(double *undRe1, double *undIm1, double *undRe2, double *undIm2, int N){
	// Parameters
	int start;
	int stride;
	int particle;
	double ReCSD;
	double ImCSD;
	double I1;
	double I2;

	// Configure grid-stride loop
	start = blockIdx.x*blockDim.x+threadIdx.x;
	stride = blockDim.x*gridDim.x;

	// Grid-stride loop (each electron contribution si computed in parallel)
	for(particle=start;particle<N;particle+=stride){
		I1 = undRe1[particle]*undRe1[particle]+undIm1[particle]*undIm1[particle];	// INTENSITY 1
		I2 = undRe2[particle]*undRe2[particle]+undIm2[particle]*undIm2[particle];	// INTENSITY 2
		ReCSD = undRe1[particle]*undRe2[particle]+undIm1[particle]*undIm2[particle];
		ImCSD = undIm1[particle]*undRe2[particle]-undRe1[particle]*undIm2[particle];
		undRe1[particle] = ReCSD;
		undIm1[particle] = ImCSD;
		undRe2[particle] = I1;
		undIm2[particle] = I2;
	}
}

__global__ void computeSum(double *array, double *result, int N){
	// Parameters
	int i;
	int start;
	int skip;
	int stride;
	int tid;

	// Configure grid-stride loop
	start = blockIdx.x*blockDim.x+threadIdx.x;
	skip = blockDim.x*gridDim.x;
	stride = N;
	tid = N/2;

	// Add array elements in parallel
	while(tid>0){
		for(i=start;i<tid;i+=skip){
			array[i] += array[i+tid];
		}
		__syncthreads();
		if((stride%2)!=0){
			array[tid-1] += array[2*tid];
		}
		stride = tid;
		tid /= 2;
	}
	*result = array[0];
}

__global__ void computeModulusSDC(double *result, double *undReCSD, double *undImCSD, double *undI1, double *undI2){
	*result = sqrt((*undReCSD)*(*undReCSD)+(*undImCSD)*(*undImCSD))/(sqrt((*undI1)*(*undI2)));
}