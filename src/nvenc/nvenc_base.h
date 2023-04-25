#pragma once

#include "nvenc_colorspace.h"
#include "nvenc_config.h"
#include "nvenc_encoded_frame.h"

#include "src/video.h"

#include <third-party/nv-codec-headers/include/ffnvcodec/nvEncodeAPI.h>

namespace nvenc {

  class nvenc_base {
  public:
    nvenc_base(NV_ENC_DEVICE_TYPE device_type, void *device, uint32_t max_width, uint32_t max_height, NV_ENC_BUFFER_FORMAT buffer_format);
    virtual ~nvenc_base();

    nvenc_base(const nvenc_base &) = delete;
    nvenc_base &
    operator=(const nvenc_base &) = delete;

    bool
    create_encoder(const nvenc_config &config, const video::config_t &client_config, const nvenc_colorspace_t &colorspace);

    nvenc_encoded_frame
    encode_frame(uint64_t frame_index, bool force_idr);

    bool
    invalidate_ref_frames(uint64_t first_frame, uint64_t last_frame);

  protected:
    void
    destroy_base_resources();

    virtual bool
    init_library() = 0;

    virtual bool
    create_input_buffer() = 0;

    virtual NV_ENC_REGISTERED_PTR
    get_input_buffer() = 0;

    const NV_ENC_DEVICE_TYPE device_type;
    void *const device;
    const uint32_t max_width;
    const uint32_t max_height;
    const NV_ENC_BUFFER_FORMAT buffer_format;

    std::unique_ptr<NV_ENCODE_API_FUNCTION_LIST> nvenc;
    void *encoder = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;

  private:
    NV_ENC_OUTPUT_PTR output_bitstream = nullptr;

    uint64_t last_encoded_frame_index = 0;

    bool supporting_ref_frame_invalidation = true;
    bool ref_frame_invalidation_requested = false;
    std::pair<uint64_t, uint64_t> last_ref_frame_invalidation_range;
  };

}  // namespace nvenc
