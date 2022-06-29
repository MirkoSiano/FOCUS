#include <iostream>
#include <math.h>

using namespace std;

__global__ void computeElectricField_FarField(double *,double *,double *,double *,double *,double *,double *,double *,double *,int);
__global__ void computeSpectralDegreeOfCoherence(double *,double *,double *,double *,int);
__global__ void computeSum(double *,double *,int);
__global__ void computeModulusSDC(double *,double *,double *,double *,double*);