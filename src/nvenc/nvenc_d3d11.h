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
    nvenc_d3d11(ID3D11Device *d3d_device);
    ~nvenc_d3d11();

    ID3D11Texture2D *
    get_input_texture();

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
    const ID3D11DevicePtr d3d_device;
    ID3D11Texture2DPtr d3d_input_texture;
  };

}  // namespace nvenc
#endif
