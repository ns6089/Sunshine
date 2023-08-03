#pragma once
#ifdef _WIN32

  #include "nvenc_base.h"

  #include <cuda.h>
  #include <cuda_runtime.h>

namespace nvenc {

  class nvenc_cuda final: public nvenc_base {
  public:
    nvenc_cuda(CUcontext cu_context);
    ~nvenc_cuda();

    cudaArray_t
    get_cuda_array();

  private:
    bool
    init_library() override;

    bool
    create_and_register_input_buffer() override;

    bool
    wait_for_async_event(uint32_t timeout_ms) override;

    using handle_t = util::safe_ptr_v2<void, BOOL, CloseHandle>;
    handle_t async_event_winapi;

    HMODULE dll = NULL;
    cudaArray_t cuda_array = nullptr;
  };

}  // namespace nvenc
#endif
