#include <torch/script.h>
#include <torch/torch.h>
#include <sys/time.h>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <fstream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class PedestrianReid{
 public:
  PedestrianReid();
  int initPedestrian(const char *model_dir,int gpuID);
  std::vector<float> PedestrianReid::ProcessSingleImg(const cv::Mat&inputImg);
  std::vector<vector<float>> PedestrianReid::PrecessedBatchImagesNormalize(const vector<cv::Mat>& inputImgs);
  void PedestrianReid::Img2TensorInput(const cv::Mat& inputImg,torch::Tensor &output);
 private:
  std::vector<vector<float>> PedestrianReid::ProcessBatchImg(const vector<cv::Mat>& inputImgs);
  std::shared_ptr<torch::jit::script::Module> module;
  bool isGPU_;
  int gpuID;
  int batch;





};