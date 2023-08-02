#pragma once

#include "display.h"

#include <comdef.h>

#include <cuda_d3d11_interop.h>
#include <cuda_runtime.h>

namespace platf::dxgi {

  class display_cuda_t: public display_base_t, public std::enable_shared_from_this<display_cuda_t> {
  public:
    _COM_SMARTPTR_TYPEDEF(IDXGIResource, IID_IDXGIResource);
    _COM_SMARTPTR_TYPEDEF(ID3D11Resource, IID_ID3D11Resource);

    ~display_cuda_t();

    int
    init(const ::video::config_t &config, const std::string &display_name);

    std::shared_ptr<img_t>
    alloc_img() override;

    int
    dummy_img(img_t *img) override { return -1; }

    int
    complete_img(img_t *img, bool dummy) override { return -1; }

    capture_e
    snapshot(const pull_free_image_cb_t &pull_free_image_cb, std::shared_ptr<platf::img_t> &img_out, std::chrono::milliseconds timeout, bool cursor_visible) override;

    std::vector<DXGI_FORMAT>
    get_supported_capture_formats() override { return { DXGI_FORMAT_B8G8R8A8_UNORM }; }

  private:
    int cuda_device = 0;
    // cudaStream_t capture_map_stream = nullptr;
    // cudaEvent_t capture_mapped_event = nullptr;

    struct cuda_array_deleter {
      void
      operator()(cudaArray_t p) {
        if (p) cudaFreeArray(p);
      }
    };

    struct cuda_img_t: img_t {
      std::unique_ptr<cudaArray, cuda_array_deleter> cuda_array;
      std::shared_ptr<display_cuda_t> display;
    };

    std::shared_ptr<cudaArray> capture_surface;

    struct encoder_instance_t {
      cudaStream_t cuda_stream;
      // cudaEvent_t capture_converted_event = nullptr;
      std::mutex capture_surface_pointer_mutex;
      std::shared_ptr<cudaArray> capture_surface;
    };
    // std::mutex encoders_mutex;
    std::list<encoder_instance_t> encoders;
  };

}  // namespace platf::dxgi
