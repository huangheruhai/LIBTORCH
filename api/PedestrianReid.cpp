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
  string sModelDir(model_dir);

  gpuID=gpuID;
  if(gpuID==0){
    torch::DeviceType device_type;
    device_type=torch::kCUDA;
    torch::Device device(device_type,0);
    module=torch::jit::load(sModelDir+"/model_device0.pt",device)


  }
  if(gpuID==1){
    torch::DeviceType device_type;
    device_type=torch::kCUDA;
    torch::Device device(device_type,1);
    module=torch::jit::load(sModelDir+"/model_device0.pt",device)


  }
  return 0;
}

void PedestrianReid::Img2TensorInput(const cv::Mat& inputImg,torch::Tensor &output){
  if(!inputImg.data){
    cout<<"input empty "<<endl;


  }
  cv::Mat proProcessImg;
  cv::cbtColor(inputImg,proProcessImg,CV_BGR2RGB);
  proProcessImg.convertTo(proProcessImg,CV_32F,1.0/255);
  cv::resize(proProcessImg,proProcessImg,cv::Size(128,384));
  auto img_tensor=torch::from_blob(proProcessImg.data,{1,384,128,3});
  img_tensor=img_tensor.permute({0,3,1,2});
  img_tensor[0][0]=img_tensor[0][0].sub_(0.485).div_(0.229);
  img_tensor[0][1]=img_tensor[0][1].sub_(0.456).div_(0.224);
  img_tensor[0][2]=img_tensor[0][2].sub_(0.406).div_(0.225);
  output=img_tensor.clone();

}

std::vector<float> PedestrianReid::ProcessSingleImg(const cv::Mat&inputImg){
  std::vector<torch::jit::IValue> inputs;
  int featurelength=3328;
  float outvalues[featurelength];
  torch::Tensor img_var;
  Img2TensorInput(inputImg,img_var);
  inputs.push_back(img_var);
  torch::Tensor out=module->forward(inputs).toTuple()->elements()[0].toTensor();
  memove(outvalues,out.to(torch::kCPU).data<float>(),sizeof(outvalues));
  vector<float> outVec(outvalues,outvalues+featurelength);
  return outVec;


}

std::vector<vector<float>> PedestrianReid::ProcessBatchImg(const vector<cv::Mat>& inputImgs){
  std::vector<vector<float>> outVectors;
  std::vector<torch::Tensor> tInputVectors;
  std::vector<torch::jit::IValue> inputsTensor;
  torch::DeviceType device_type;
  device_type = torch::kCUDA;
  torch::Device device(device_type,gpuID_);
  for (int i=0;i<batch;i++){
  torch::Tensor tempTensor;
  Img2TensorInput(inputImgs[i],tempTensor);
  tInputVectors.push_back(tempTensor);

  }
  torch::Tensor batchTensor =torch::cat(tInputVectors);
  inputsTensor.push_back(batchTensor.to(device));
  torch::outTensor=module->forward(inputsTensor).toTuple()->elements()[0].toTensor();
  int featurelength=3328;
  float outvalues[featurelength];
  for (int i=0;i<batch;i++){
    memove(outvalues,out.to(torch::kCPU).data<float>(),sizeof(outvalues));
    vector<float> outVec(outvalues,outvalues+featurelength);
    outVectors.push_back(outVec);

  }

  return outVectors;

}

std::vector<vector<float>> PedestrianReid ::PrecessedBatchImagesNormalize(const vector<cv::Mat>& inputImgs){
  std::vector<vector<float>> outVectors=ProcessBatchImg(inputImgs);
  std::vector<vector<float >> outNormalized;
  int vlen=outVectors[0].size();
  for( int i=0;i<batch;i++){
    vector<float> oneVector=outVectors[i];
    float fnorm=normalizeprocess(oneVector);
    for(j=0;j<vlen;j++){
      oneVector/=fnorm;

    }
    outNormalized.push_back(oneVector);


  }

  return outNormalized;
}