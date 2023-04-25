#ifdef _WIN32
  #include "nvenc_d3d11.h"

  #include "nvenc_utils.h"

namespace nvenc {

  nvenc_d3d11::nvenc_d3d11(ID3D11Device *d3d_device, uint32_t max_width, uint32_t max_height, NV_ENC_BUFFER_FORMAT buffer_format):
      nvenc_base(NV_ENC_DEVICE_TYPE_DIRECTX, d3d_device, max_width, max_height, buffer_format),
      d3d_device(d3d_device) {
  }

  nvenc_d3d11::~nvenc_d3d11() {
    destroy_base_resources();

    if (d3d_input_texture) {
      nvenc->nvEncUnregisterResource(encoder, d3d_input_texture_reg);
    }

    if (dll) {
      FreeLibrary(dll);
      dll = NULL;
    }
  }

  ID3D11Texture2D *
  nvenc_d3d11::get_input_texture() {
    return d3d_input_texture.GetInterfacePtr();
  }

  bool
  nvenc_d3d11::init_library() {
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
        if (create_instance(new_nvenc.get()) == NV_ENC_SUCCESS) {
          nvenc = std::move(new_nvenc);
          return true;
        }
        else {
          BOOST_LOG(error) << "NvEncodeAPICreateInstance failed";
        }
      }
    }

    if (dll) {
      FreeLibrary(dll);
      dll = NULL;
    }

    return false;
  }

  bool
  nvenc_d3d11::create_input_buffer() {
    if (d3d_input_texture) return false;

    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = max_width;
    desc.Height = max_height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = dxgi_format_from_nvenc_format(buffer_format);
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_RENDER_TARGET;
    desc.CPUAccessFlags = 0;
    if (d3d_device->CreateTexture2D(&desc, nullptr, &d3d_input_texture) != S_OK) {
      BOOST_LOG(error) << "Couldn't create input texture for NvENC";
      return false;
    }

    NV_ENC_REGISTER_RESOURCE register_resource = { NV_ENC_REGISTER_RESOURCE_VER };
    register_resource.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX;
    register_resource.width = max_width;
    register_resource.height = max_height;
    register_resource.resourceToRegister = d3d_input_texture.GetInterfacePtr();
    register_resource.bufferFormat = buffer_format;
    register_resource.bufferUsage = NV_ENC_INPUT_IMAGE;

    if (nvenc->nvEncRegisterResource(encoder, &register_resource) != NV_ENC_SUCCESS) {
      BOOST_LOG(error) << "NvEncRegisterResource failed: " << nvenc->nvEncGetLastErrorString(encoder);
      return false;
    }
    d3d_input_texture_reg = register_resource.registeredResource;

    return true;
  }

  NV_ENC_REGISTERED_PTR
  nvenc_d3d11::get_input_buffer() {
    return d3d_input_texture_reg;
  }

}  // namespace nvenc
#endif
