#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

#include <iostream>

namespace caffe {

template <typename Dtype>
__global__ void CroppingForward(const int count, const Dtype* in, Dtype* out,
    const int num, const int channel, const int height_in, const int width_in,
    const int startx, const int starty, const int height_out, const int width_out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < count) {
    int w = index % width_out;
    index /= width_out;
    int h = index % height_out;
    index /= height_out;
    int c = index % channel;
    index /= channel;
    out[((index * channel + c) * height_out + h) * width_out + w] =
        in[((index * channel + c) * height_in + h + starty) * width_in + w + startx];
  }
}

template <typename Dtype>
void CroppingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = top[0]->count();
  // First, set all data to be zero for the boundary pixels
  CroppingForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, NUM_, CHANNEL_, HEIGHT_IN_, WIDTH_IN_,
      STARTX_, STARTY_, HEIGHT_OUT_, WIDTH_OUT_);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void CroppingBackward(const int count, const Dtype* in, Dtype* out,
    const int num, const int channel, const int height_in, const int width_in,
    const int startx, const int starty, const int height_out, const int width_out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < count) {
    int w = index % width_out;
    index /= width_out;
    int h = index % height_out;
    index /= height_out;
    int c = index % channel;
    index /= channel;
    out[((index * channel + c) * height_in + h + starty) * width_in + w + startx] =
        in[((index * channel + c) * height_out + h) * width_out + w];
  }
}

template <typename Dtype>
void CroppingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = top[0]->count();
    CUDA_CHECK(cudaMemset(bottom_diff, 0, sizeof(Dtype) * (*bottom)[0]->count()));
    CroppingBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_diff, NUM_, CHANNEL_, HEIGHT_IN_, WIDTH_IN_,
        STARTX_, STARTY_, HEIGHT_OUT_, WIDTH_OUT_);
    CUDA_POST_KERNEL_CHECK;
  }
  return ;
}

}  // namespace caffe
