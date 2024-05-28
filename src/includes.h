#pragma once
#include "defines.h"

#ifdef CUDA_ENABLED
#include "tensor.cu"
#include "multidim.cu"
// TODO: add more cuda files here once we make them
#endif
#ifdef CUDA_DISABLED
#include "tensor.h"
#endif