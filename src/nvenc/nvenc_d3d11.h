#pragma once
#ifdef _WIN32

  #include <comdef.h>
  #include <d3d11.h>

  #include "nvenc_base.h"

namespace nvenc {

  _COM_SMARTPTR_TYPEDEF(ID3D11Device, IID_ID3D11Device);
  _COM_SMARTPTR_TYPEDEF(ID3D11Texture2D, IID_ID3D11Texture2D);

  class nvenc_d3d11 final: public nvenc_base {
  public:
    nvenc_d3d11(ID3D11Device *d3d_device, uint32_t max_width, uint32_t max_height, NV_ENC_BUFFER_FORMAT buffer_format);
    ~nvenc_d3d11();

    ID3D11Texture2D *
    get_input_texture();

  private:
    bool
    init_library() override;

    bool
    create_input_buffer() override;

    NV_ENC_REGISTERED_PTR
    get_input_buffer() override;

    HMODULE dll = NULL;
    const ID3D11DevicePtr d3d_device;
    ID3D11Texture2DPtr d3d_input_texture;
    NV_ENC_REGISTERED_PTR d3d_input_texture_reg = nullptr;
  };

}  // namespace nvenc
#endif
