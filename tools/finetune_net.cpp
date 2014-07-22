// Copyright 2014 BVLC and contributors.
//
// This is a simple script that allows one to quickly finetune a network.
// Usage:
//    finetune_net solver_proto_file pretrained_net

#include <string>
#include <iostream>
#include "caffe/caffe.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 3) {
    LOG(ERROR) << "Usage: finetune_net solver_proto_file pretrained_net";
    return 1;
  }
  std::cout << "SolverParam Generation..." << std::endl;
  SolverParameter solver_param;
  std::cout << "Reading Proto..." << std::endl;
  ReadProtoFromTextFileOrDie(argv[1], &solver_param);

  std::cout << "Starting optimization" << std::endl;
  
  SGDSolver<float> solver(solver_param);
  std::cout <<  "Loading from " << argv[2] << std::endl;
  LOG(INFO) << "Loading from " << argv[2];
  solver.net()->CopyTrainedLayersFrom(string(argv[2]));
  std::cout << "Optimization..." << std::endl;
  solver.Solve();
  std::cout << "Optimization Done" << std::endl;
  return 0;
}
