#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include <iostream>
#include <tensorflow/c/c_api.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>

int main() {
  using namespace tensorflow;
  using namespace tensorflow::ops;
  std::cout << "Hello from Tensorflow C library version" << TF_Version();

  Session *session1;
  Status status = NewSession(SessionOptions(), &session1);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }
  std::cout << "Session successfully created.\n";

  Scope root = Scope::NewRootScope();
  auto A = Const(root, {{3.f, 2.f}, {-1.f, 0.f}});
  auto b = Const(root, {{3.f, 5.f}});
  auto v = MatMul(root.WithOpName("v"), A, b, MatMul::TransposeB(true));
  std::vector<Tensor> outputs;
  // ClientSession is higher level API for building graph using C++
  ClientSession session(root);
  TF_CHECK_OK(session.Run({v}, &outputs));
  // Expect outputs[0] == [19; -3]
  LOG(INFO) << outputs[0].matrix<float>();
  return 0;
}
