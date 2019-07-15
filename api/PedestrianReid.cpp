#include <PedestrianReid.h>
#include <math.h>

PedestrianReid::PedestrianReid(){
  gpuId=0;
  isGPU=true;
  batch_size=10;

}

float normalizeprocess(vector<float> &vFeature){
  float fNorm=0;
  for(int k=0;k<3328;k++){
    fNorm+=pow(vFeature[k],2)
  }

  return sqrt(fNorm);

}


int PedestrianReid::initPedestrian(const char *model_dir,int gpuID){


}