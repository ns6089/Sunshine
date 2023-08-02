#include "display_cuda.h"

namespace platf::dxgi {

  display_cuda_t::~display_cuda_t() {
    // TODO: check thread
    // if (capture_mapped_event) cudaEventDestroy(capture_mapped_event);
    cudaDeviceReset();
  }

  int
  display_cuda_t::init(const ::video::config_t &config, const std::string &display_name) {
    auto base_error = display_base_t::init(config, display_name);
    if (base_error) return base_error;

    cudaError_t cuda_error;

    cuda_error = cudaD3D11GetDevice(&cuda_device, adapter.get());
    if (cuda_error != cudaSuccess) return 1;

    cuda_error = cudaSetDevice(cuda_device);
    if (cuda_error != cudaSuccess) return 1;

    // TODO: cudaStreamCreateWithPriority
    // cuda_error = cudaStreamCreateWithFlags(&capture_map_stream, cudaStreamNonBlocking);
    // if (cuda_error != cudaSuccess) return 1;

    // cuda_error = cudaEventCreate(&capture_mapped_event);
    // if (cuda_error != cudaSuccess) return 1;

    // TODO: remember thread

    return 0;
  }

  std::shared_ptr<img_t>
  display_cuda_t::alloc_img() {
    cudaError_t cuda_error;
    cudaArray_t cuda_array;
    // TODO: format from display
    cudaChannelFormatDesc cuda_channel_desc = cudaCreateChannelDesc<uchar4>();
    cuda_error = cudaMallocArray(&cuda_array, &cuda_channel_desc, width, height, cudaArraySurfaceLoadStore);
    if (cuda_error != cudaSuccess) return nullptr;

    auto new_img = std::make_shared<cuda_img_t>();
    new_img->display = shared_from_this();
    new_img->width = width;
    new_img->height = height;
    new_img->cuda_array = std::unique_ptr<cudaArray, cuda_array_deleter>(cuda_array, cuda_array_deleter());

    return new_img;
  }

  capture_e
  display_cuda_t::snapshot(const pull_free_image_cb_t &pull_free_image_cb, std::shared_ptr<platf::img_t> &img_out, std::chrono::milliseconds timeout, bool cursor_visible) {
    if (!dup.dup) return capture_e::error;
    // TODO: return if no encoders

    // TODO: check thread

    DXGI_OUTDUPL_FRAME_INFO dup_info = {};
    IDXGIResourcePtr dxgi_resource;
    HRESULT dup_result;
    dup_result = dup.dup->AcquireNextFrame(1000, &dup_info, &dxgi_resource);

    if (dup_result == DXGI_ERROR_WAIT_TIMEOUT) return capture_e::timeout;
    if (dup_result == DXGI_ERROR_ACCESS_LOST) return capture_e::reinit;
    if (dup_result != S_OK) return capture_e::error;

    if (dup_info.LastPresentTime.QuadPart == 0 || dup_info.AccumulatedFrames == 0) return capture_e::timeout;
    if (!dxgi_resource) return capture_e::error;

    ID3D11ResourcePtr d3d11_resource;
    if (FAILED(dxgi_resource->QueryInterface(IID_PPV_ARGS(&d3d11_resource)))) return capture_e::error;

    cudaError_t cuda_error;
    cudaGraphicsResource *cuda_resource;

    if (!capture_surface || capture_surface.use_count() > 1) {
      cudaArray_t new_surface;
      cudaChannelFormatDesc new_surface_desc = cudaCreateChannelDesc<uchar4>();
      cuda_error = cudaMallocArray(&new_surface, &new_surface_desc, width, height, cudaArraySurfaceLoadStore);
      if (cuda_error != cudaSuccess) return capture_e::error;
      capture_surface = std::shared_ptr<cudaArray>(new_surface, cuda_array_deleter());
    }

    cuda_error = cudaGraphicsD3D11RegisterResource(&cuda_resource, d3d11_resource, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    if (cuda_error != cudaSuccess) return capture_e::error;

    cuda_error = cudaGraphicsMapResources(1, &cuda_resource, 0);
    if (cuda_error != cudaSuccess) return capture_e::error;

    {
      cudaArray_t mapped_array;

      cuda_error = cudaGraphicsSubResourceGetMappedArray(&mapped_array, cuda_resource, 0, 0);
      if (cuda_error != cudaSuccess) return capture_e::error;

      cuda_error = cudaMemcpy2DArrayToArray(capture_surface.get(), 0, 0, mapped_array, 0, 0, width, height);
      if (cuda_error != cudaSuccess) return capture_e::error;
    }

    cuda_error = cudaGraphicsUnmapResources(1, &cuda_resource, 0);
    if (cuda_error != cudaSuccess) return capture_e::error;

    cuda_error = cudaGraphicsUnregisterResource(cuda_resource);
    if (cuda_error != cudaSuccess) return capture_e::error;

    dup_result = dup.dup->ReleaseFrame();
    if (dup_result == DXGI_ERROR_ACCESS_LOST) return capture_e::reinit;
    if (dup_result != S_OK && dup_result != DXGI_ERROR_INVALID_CALL) return capture_e::error;

    // cuda_error = cudaEventRecord(capture_mapped_event, capture_map_stream);
    // if (cuda_error != cudaSuccess) return capture_e::error;

    for (auto &encoder : encoders) {
      {
        std::scoped_lock lock(encoder.capture_surface_pointer_mutex);
        encoder.capture_surface = capture_surface;
      }
      // TODO: notify
    }

    /*
    {
      std::scoped_lock encoders_lock(encoders_mutex);

      for (auto& encoder : encoders) {
        cuda_error = cudaStreamWaitEvent(encoder.cuda_stream, capture_mapped_event);
        if (cuda_error != cudaSuccess) return capture_e::error;

        // TODO: processing

        cuda_error = cudaEventRecord(encoder.capture_converted_event, encoder.cuda_stream);
        if (cuda_error != cudaSuccess) return capture_e::error;

        cuda_error = cudaStreamWaitEvent(capture_map_stream, encoder.capture_converted_event);
        if (cuda_error != cudaSuccess) return capture_e::error;
      }
    }
    */

    // cuda_error = cudaStreamSynchronize(capture_map_stream);
    // if (cuda_error != cudaSuccess) return capture_e::error;

    return capture_e::ok;
  }

}  // namespace platf::dxgi
