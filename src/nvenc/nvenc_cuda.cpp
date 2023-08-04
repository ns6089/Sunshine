#ifdef _WIN32
  #include "nvenc_cuda.h"

  #include "nvenc_utils.h"

namespace nvenc {

  nvenc_cuda::nvenc_cuda(CUcontext cu_context):
      async_event_winapi(CreateEvent(nullptr, FALSE, FALSE, nullptr)),
      nvenc_base(NV_ENC_DEVICE_TYPE_CUDA, cu_context) {
    async_event_handle = async_event_winapi.get();
  }

  nvenc_cuda::~nvenc_cuda() {
    if (encoder) destroy_encoder();

    if (cuda_array) cudaFreeArray(cuda_array);
    if (cuda_stream) cudaStreamDestroy(cuda_stream);

    if (dll) {
      FreeLibrary(dll);
      dll = NULL;
    }
  }

  cudaArray_t
  nvenc_cuda::get_cuda_array() {
    return cuda_array;
  }

  bool
  nvenc_cuda::init_library() {
    if (dll) return true;

  #ifdef _WIN64
    auto dll_name = "nvEncodeAPI64.dll";
  #else
    auto dll_name = "nvEncodeAPI.dll";
  #endif

    if ((dll = LoadLibraryEx(dll_name, NULL, LOAD_LIBRARY_SEARCH_SYSTEM32))) {
      if (auto create_instance = (decltype(NvEncodeAPICreateInstance) *) GetProcAddress(dll, "NvEncodeAPICreateInstance")) {
        auto new_nvenc = std::make_unique<NV_ENCODE_API_FUNCTION_LIST>();
        new_nvenc->version = NV_ENCODE_API_FUNCTION_LIST_VER;
        if (nvenc_failed(create_instance(new_nvenc.get()))) {
          BOOST_LOG(error) << "NvEncodeAPICreateInstance failed: " << last_error_string;
        }
        else {
          nvenc = std::move(new_nvenc);
          return true;
        }
      }
      else {
        BOOST_LOG(error) << "No NvEncodeAPICreateInstance in " << dll_name;
      }
    }
    else {
      BOOST_LOG(debug) << "Couldn't load NvEnc library " << dll_name;
    }

    if (dll) {
      FreeLibrary(dll);
      dll = NULL;
    }

    return false;
  }

  bool
  nvenc_cuda::create_and_register_input_buffer() {
    if (!cuda_array) {
      cudaError_t cuda_error;
      // TODO: format from display
      cudaChannelFormatDesc cuda_channel_desc = cudaCreateChannelDesc<uchar4>();
      cuda_error = cudaMallocArray(&cuda_array, &cuda_channel_desc, encoder_params.width, encoder_params.height, cudaArraySurfaceLoadStore);
      if (cuda_error != cudaSuccess) return false;
      // TODO: log error
    }

    /*
    if (!d3d_input_texture) {
      D3D11_TEXTURE2D_DESC desc = {};
      desc.Width = encoder_params.width;
      desc.Height = encoder_params.height;
      desc.MipLevels = 1;
      desc.ArraySize = 1;
      desc.Format = dxgi_format_from_nvenc_format(encoder_params.buffer_format);
      desc.SampleDesc.Count = 1;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_RENDER_TARGET;
      if (d3d_device->CreateTexture2D(&desc, nullptr, &d3d_input_texture) != S_OK) {
        BOOST_LOG(error) << "NvEnc: couldn't create input texture";
        return false;
      }
    }
    */

    if (!registered_input_buffer) {
      NV_ENC_REGISTER_RESOURCE register_resource = { NV_ENC_REGISTER_RESOURCE_VER };
      register_resource.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY;
      register_resource.width = encoder_params.width;
      register_resource.height = encoder_params.height;
      // TODO: from format
      // register_resource.pitch = encoder_params.width * 4;
      register_resource.resourceToRegister = cuda_array;
      register_resource.bufferFormat = encoder_params.buffer_format;
      register_resource.bufferUsage = NV_ENC_INPUT_IMAGE;

      if (nvenc_failed(nvenc->nvEncRegisterResource(encoder, &register_resource))) {
        BOOST_LOG(error) << "NvEncRegisterResource failed: " << last_error_string;
        return false;
      }

      registered_input_buffer = register_resource.registeredResource;
    }

    if (!cuda_stream) {
      cudaError_t cuda_error;
      cuda_error = cudaStreamCreateWithFlags(&cuda_stream, cudaStreamDefault);
      if (cuda_error != cudaSuccess) return false;
    }

    if (nvenc_failed(nvenc->nvEncSetIOCudaStreams(encoder, &cuda_stream, &cuda_stream))) {
      BOOST_LOG(error) << "NvEncSetIOCudaStreams failed: " << last_error_string;
      return false;
    }

    return true;
  }

  bool
  nvenc_cuda::wait_for_async_event(uint32_t timeout_ms) {
    return WaitForSingleObject(async_event_winapi.get(), timeout_ms) == WAIT_OBJECT_0;
  }

}  // namespace nvenc
#endif
