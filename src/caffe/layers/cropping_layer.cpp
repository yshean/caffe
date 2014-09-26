// Copyright 2014 Aravindh Mahendran
// adapted from Padding layer code

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

#include <iostream>

namespace caffe {

  template <typename Dtype>
  void CroppingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    STARTX_ = this->layer_param_.cropping_param().cropx() - 1; //Indexed at 0 not 1
    STARTY_ = this->layer_param_.cropping_param().cropy() - 1; //Indexed at 0 not 1
    HEIGHT_OUT_ = this->layer_param_.cropping_param().crop_height();
    WIDTH_OUT_ = this->layer_param_.cropping_param().crop_width();
    CHECK_EQ(bottom.size(), 1) << "Cropping Layer takes a single blob as input.";
    CHECK_EQ(top.size(), 1) << "Cropping Layer takes a single blob as output.";
    NUM_ = bottom[0]->num();
    CHANNEL_ = bottom[0]->channels();
    HEIGHT_IN_ = bottom[0]->height();
    WIDTH_IN_ = bottom[0]->width();
    top[0]->Reshape(NUM_, CHANNEL_, HEIGHT_OUT_, WIDTH_OUT_);
  }

template <typename Dtype>
void CroppingLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  STARTX_ = this->layer_param_.cropping_param().cropx() - 1; //Indexed at 0 not 1
  STARTY_ = this->layer_param_.cropping_param().cropy() - 1; //Indexed at 0 not 1
  HEIGHT_OUT_ = this->layer_param_.cropping_param().crop_height();
  WIDTH_OUT_ = this->layer_param_.cropping_param().crop_width();
  CHECK_EQ(bottom.size(), 1) << "Cropping Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "Cropping Layer takes a single blob as output.";
  NUM_ = bottom[0]->num();
  CHANNEL_ = bottom[0]->channels();
  HEIGHT_IN_ = bottom[0]->height();
  WIDTH_IN_ = bottom[0]->width();
  top[0]->Reshape(NUM_, CHANNEL_, HEIGHT_OUT_, WIDTH_OUT_);

}

template <typename Dtype>
void CroppingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  // In short, top[n, c, h, w] = bottom[n, c, h+starty, w+startx] if in range
  for (int n = 0; n < NUM_; ++n) {
    for (int c = 0; c < CHANNEL_; ++c) {
      for (int h = 0; h < HEIGHT_OUT_; ++h) {
        // copy the width part
        memcpy(
            top_data + ((n * CHANNEL_ + c) * HEIGHT_OUT_ + h)
                * WIDTH_OUT_,
            bottom_data + ((n * CHANNEL_ + c) * HEIGHT_IN_ + h + STARTY_) * WIDTH_IN_ + STARTX_,
            sizeof(Dtype) * WIDTH_OUT_);
      }
    }
  }
}

template <typename Dtype>
void CroppingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  memset(bottom_diff, 0, sizeof(Dtype) * bottom[0]->count());
  // In short,  bottom[n, c, h+starty-1, w+startx-1] = top[n, c, h, w] if in range
  for (int n = 0; n < NUM_; ++n) {
    for (int c = 0; c < CHANNEL_; ++c) {
      for (int h = 0; h < HEIGHT_OUT_; ++h) {
        // copy the width part
        memcpy(
            bottom_diff + ((n * CHANNEL_ + c) * HEIGHT_IN_ + h + STARTY_) * WIDTH_IN_ + STARTX_,
            top_diff + ((n * CHANNEL_ + c) * HEIGHT_OUT_ + h)
                * WIDTH_OUT_,
            sizeof(Dtype) * WIDTH_OUT_);
      }
    }
  }
  return ;
}

#ifdef CPU_ONLY
STUB_GPU(CroppingLayer);
#endif

INSTANTIATE_CLASS(CroppingLayer);


}  // namespace caffe
