#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <vector>
#include <time.h>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  // Deserialize the ScriptModule from a file using torch::jit::load().
  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);

  assert(module != nullptr);
  std::cout << "ok\n";
  torch::DeviceType device_type;
  device_type=torch::kCUDA;
  torch::Device device(device_type);
  module->to(device);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({5,3,384,128}).to(device))；

  auto output = module->forward(inputs).toTuple()->elements()[0].toTensor();

  auto startTime = std::chrono::high_resolution_clock::now();
  for(int i=0;i<1000;i++){
      auto output = module->forward(inputs).toTuple()->elements()[0].toTensor();

  }
  auto endTime=std::chrono::high_resolution_clock::now();
  float totalTime=std::chrono::duration<float,std::mill>(endTime-startTime).count();
  std::cout<<"Time used one image ( measured by chrono):"<<totalTime/1000<<" ms"<<std::endl;
  device_type = torch::kCPU;
  torch::Device device2(device_type);

  float normvalue[5][3328];
  float normvalue2[3328];
  for(int i=0;i<5;i++){
      memmove(normvalue[i],output[i].to(device2).data<float>(),sizeof(normvalue2));
  }
//  for(int i=0;i<3328;i++){
//  std::cout<<normvalue[4][i]<<",";
//
//  }



}