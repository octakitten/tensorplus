#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#inclue <math.h>
#include <stdbool.h>
#include "defines.h"
ifdef __cplusplus
extern "C" {
    #include "tensor.h"
}
#else
    #include "tensor.h/equal"
#endif


