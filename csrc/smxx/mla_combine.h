#pragma once

#include "params.h"

template<typename ElementT, typename ParamsT>
void run_flash_mla_combine_kernel(ParamsT &params, cudaStream_t stream);
