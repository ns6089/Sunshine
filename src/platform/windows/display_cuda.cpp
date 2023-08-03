#include "display_cuda.h"

#include "src/nvenc/nvenc_config.h"
#include "src/nvenc/nvenc_cuda.h"
#include "src/nvenc/nvenc_utils.h"

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

    // int cuda_device;
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
    /*
    cudaError_t cuda_error;
    cudaArray_t cuda_array;
    // TODO: format from display
    cudaChannelFormatDesc cuda_channel_desc = cudaCreateChannelDesc<uchar4>();
    cuda_error = cudaMallocArray(&cuda_array, &cuda_channel_desc, width, height, cudaArraySurfaceLoadStore);
    if (cuda_error != cudaSuccess) return nullptr;
    */

    auto new_img = std::make_shared<cuda_img_t>();
    new_img->display = shared_from_this();
    new_img->width = width;
    new_img->height = height;

    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    // TODO: from display
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    if (device->CreateTexture2D(&desc, nullptr, &new_img->texture) != S_OK) return nullptr;

    cudaError_t cuda_error;
    cudaGraphicsResource_t cuda_resource;
    // cuda_error = cudaGraphicsD3D11RegisterResource(&cuda_resource, new_img->texture, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    cuda_error = cudaGraphicsD3D11RegisterResource(&cuda_resource, new_img->texture, cudaGraphicsRegisterFlagsNone);
    if (cuda_error != cudaSuccess) return nullptr;
    cuda_error = cudaGraphicsResourceSetMapFlags(cuda_resource, cudaGraphicsMapFlagsReadOnly);
    if (cuda_error != cudaSuccess) return nullptr;

    new_img->cuda_resource = std::unique_ptr<cudaGraphicsResource, cuda_resource_unregistrator>(cuda_resource, cuda_resource_unregistrator());

    return new_img;
  }

  capture_e
  display_cuda_t::capture(const push_captured_image_cb_t &push_captured_image_cb, const pull_free_image_cb_t &pull_free_image_cb, bool *cursor) {
    while (true) {
      if (!dup.dup) return capture_e::error;
      // TODO: return if no encoders

      // TODO: check thread

      DXGI_OUTDUPL_FRAME_INFO dup_info = {};
      IDXGIResourcePtr dxgi_resource;
      HRESULT dup_result;
      dup_result = dup.dup->AcquireNextFrame(1000, &dup_info, &dxgi_resource);

      if (dup_result == DXGI_ERROR_WAIT_TIMEOUT) {
        if (!push_captured_image_cb({}, false)) {
          BOOST_LOG(info) << "Returning on timeout";
          return capture_e::ok;
        }
        continue;

        // TODO: delete this
        /*
        {
          std::shared_ptr<platf::img_t> new_img;
          if (!pull_free_image_cb(new_img)) return capture_e::interrupted;
          auto new_cuda_img = (cuda_img_t *) new_img.get();
          if (!new_cuda_img || !new_cuda_img->texture) return capture_e::error;

          if (new_cuda_img->cuda_resource_mapped) {
            cudaError_t cuda_error;
            auto cuda_resource = new_cuda_img->cuda_resource.get();
            cuda_error = cudaGraphicsUnmapResources(1, &cuda_resource, 0);
            if (cuda_error != cudaSuccess) return capture_e::error;
            new_cuda_img->cuda_resource_mapped = false;
            new_cuda_img->cuda_array = nullptr;
          }
        }
        */
      }

      if (dup_result == DXGI_ERROR_ACCESS_LOST) return capture_e::reinit;
      if (dup_result != S_OK) return capture_e::error;

      if (dup_info.LastPresentTime.QuadPart == 0 || dup_info.AccumulatedFrames == 0) {
        // if (!push_captured_image_cb({}, false)) return capture_e::ok;
        dup.dup->ReleaseFrame();
        continue;
      }
      // if (dup_info.AccumulatedFrames == 0) return capture_e::timeout;
      if (!dxgi_resource) return capture_e::error;

      // ID3D11ResourcePtr d3d11_resource;
      ID3D11Texture2DPtr d3d11_resource;
      if (FAILED(dxgi_resource->QueryInterface(IID_PPV_ARGS(&d3d11_resource)))) return capture_e::error;

      // cudaError_t cuda_error;
      // cudaGraphicsResource *cuda_resource;

      std::shared_ptr<platf::img_t> new_img;
      if (!pull_free_image_cb(new_img)) return capture_e::interrupted;
      auto new_cuda_img = (cuda_img_t *) new_img.get();
      if (!new_cuda_img || !new_cuda_img->texture) return capture_e::error;

      if (new_cuda_img->cuda_resource_mapped) {
        cudaError_t cuda_error;
        auto cuda_resource = new_cuda_img->cuda_resource.get();
        cuda_error = cudaGraphicsUnmapResources(1, &cuda_resource, 0);
        if (cuda_error != cudaSuccess) return capture_e::error;
        new_cuda_img->cuda_resource_mapped = false;
        new_cuda_img->cuda_array = nullptr;
      }

      if (!device_ctx) return capture_e::error;
      device_ctx->CopyResource(new_cuda_img->texture, d3d11_resource);

      {
        cudaError_t cuda_error;
        auto cuda_resource = new_cuda_img->cuda_resource.get();
        cuda_error = cudaGraphicsMapResources(1, &cuda_resource, 0);
        if (cuda_error != cudaSuccess) return capture_e::error;
        new_cuda_img->cuda_resource_mapped = true;
        cuda_error = cudaGraphicsSubResourceGetMappedArray(&new_cuda_img->cuda_array, cuda_resource, 0, 0);
        if (cuda_error != cudaSuccess) return capture_e::error;
      }

      dup_result = dup.dup->ReleaseFrame();
      if (dup_result == DXGI_ERROR_ACCESS_LOST) return capture_e::reinit;
      if (dup_result != S_OK && dup_result != DXGI_ERROR_INVALID_CALL) return capture_e::error;

      if (!push_captured_image_cb(std::move(new_img), true)) return capture_e::ok;
    }

    /*
    cuda_error = cudaGraphicsD3D11RegisterResource(&cuda_resource, d3d11_resource.GetInterfacePtr(), cudaGraphicsRegisterFlagsSurfaceLoadStore);
    if (cuda_error != cudaSuccess) return capture_e::error;

    cuda_error = cudaGraphicsMapResources(1, &cuda_resource, 0);
    if (cuda_error != cudaSuccess) return capture_e::error;

    {
      cudaArray_t mapped_array;

      cuda_error = cudaGraphicsSubResourceGetMappedArray(&mapped_array, cuda_resource, 0, 0);
      if (cuda_error != cudaSuccess) return capture_e::error;

      cuda_error = cudaMemcpy2DArrayToArray(new_cuda_img->cuda_array.get(), 0, 0, mapped_array, 0, 0, width, height);
      if (cuda_error != cudaSuccess) return capture_e::error;
    }

    cuda_error = cudaGraphicsUnmapResources(1, &cuda_resource, 0);
    if (cuda_error != cudaSuccess) return capture_e::error;

    cuda_error = cudaGraphicsUnregisterResource(cuda_resource);
    if (cuda_error != cudaSuccess) return capture_e::error;
    */

    // cuda_error = cudaEventRecord(capture_mapped_event, capture_map_stream);
    // if (cuda_error != cudaSuccess) return capture_e::error;

    /*
    for (auto &encoder : encoders) {
      {
        std::scoped_lock lock(encoder.capture_surface_pointer_mutex);
        //encoder.capture_surface = capture_surface;
      }
      // TODO: notify
    }
    */

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
  }

  class cuda_nvenc_encode_device_t: public nvenc_encode_device_t {
  public:
    bool
    init_device(std::shared_ptr<display_cuda_t> display, CUcontext cu_context) {
      /*

      buffer_format = nvenc::nvenc_format_from_sunshine_format(pix_fmt);
      if (buffer_format == NV_ENC_BUFFER_FORMAT_UNDEFINED) {
        BOOST_LOG(error) << "Unexpected pixel format for NvENC ["sv << from_pix_fmt(pix_fmt) << ']';
        return false;
      }

      if (base.init(display, adapter_p, pix_fmt)) return false;
      // base.apply_colorspace(colorspace);

      nvenc_d3d = std::make_unique<nvenc::nvenc_d3d11>(base.device.get());
      nvenc = nvenc_d3d.get();
      */

      nvenc_cuda = std::make_unique<nvenc::nvenc_cuda>(cu_context);
      nvenc = nvenc_cuda.get();

      return true;
    }

    bool
    init_encoder(const ::video::config_t &client_config, const ::video::sunshine_colorspace_t &colorspace) override {
      if (!nvenc_cuda) return false;

      nvenc::nvenc_config nvenc_config;
      auto nvenc_colorspace = nvenc::nvenc_colorspace_from_sunshine_colorspace(colorspace);
      return nvenc_cuda->create_encoder(nvenc_config, client_config, nvenc_colorspace, NV_ENC_BUFFER_FORMAT_ARGB);
    }

    int
    convert(platf::img_t &img_base) override {
      if (!nvenc_cuda) return 1;
      auto &img = (display_cuda_t::cuda_img_t &) img_base;

      if (img.cuda_array) {
        cudaError_t cuda_error;
        // TODO: from format
        cuda_error = cudaMemcpy2DArrayToArray(nvenc_cuda->get_cuda_array(), 0, 0, img.cuda_array, 0, 0, img.width * 4, img.height);
        if (cuda_error != cudaSuccess) return 1;

        // REMOVE THIS!
        /*
        auto cuda_resource = img.cuda_resource.get();
        cuda_error = cudaGraphicsUnmapResources(1, &cuda_resource, 0);
        if (cuda_error != cudaSuccess) return 1;
        img.cuda_resource_mapped = false;
        img.cuda_array = nullptr;
        */
      }

      return 0;
    }

  private:
    // d3d_base_encode_device base;
    std::unique_ptr<nvenc::nvenc_cuda> nvenc_cuda;
    // std::shared_ptr<platf::img_t> last_img;
    // NV_ENC_BUFFER_FORMAT buffer_format = NV_ENC_BUFFER_FORMAT_UNDEFINED;
  };

  std::unique_ptr<nvenc_encode_device_t>
  display_cuda_t::make_nvenc_encode_device(pix_fmt_e pix_fmt) {
    cudaError_t cuda_error;
    cuda_error = cudaSetDevice(cuda_device);
    if (cuda_error != cudaSuccess) return nullptr;

    CUcontext cu_context;
    CUresult cu_error;
    cu_error = cuCtxGetCurrent(&cu_context);
    if (cu_error != CUDA_SUCCESS) return nullptr;

    auto device = std::make_unique<cuda_nvenc_encode_device_t>();
    if (!device->init_device(shared_from_this(), cu_context)) {
      return nullptr;
    }

    return device;
  }

}  // namespace platf::dxgi
