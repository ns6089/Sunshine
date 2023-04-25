#include "nvenc_base.h"

namespace {
  GUID
  quality_preset_guid_from_number(unsigned number) {
    if (number > 7) number = 7;

    switch (number) {
      case 1:
      default:
        return NV_ENC_PRESET_P1_GUID;

      case 2:
        return NV_ENC_PRESET_P2_GUID;

      case 3:
        return NV_ENC_PRESET_P3_GUID;

      case 4:
        return NV_ENC_PRESET_P4_GUID;

      case 5:
        return NV_ENC_PRESET_P5_GUID;

      case 6:
        return NV_ENC_PRESET_P6_GUID;

      case 7:
        return NV_ENC_PRESET_P7_GUID;
    }
  };

  bool
  equal_guids(const GUID &guid1, const GUID &guid2) {
    return std::memcmp(&guid1, &guid2, sizeof(GUID)) == 0;
  }

  auto
  quality_preset_string_from_guid(const GUID &guid) {
    if (equal_guids(guid, NV_ENC_PRESET_P1_GUID)) {
      return "P1";
    }
    if (equal_guids(guid, NV_ENC_PRESET_P2_GUID)) {
      return "P2";
    }
    if (equal_guids(guid, NV_ENC_PRESET_P3_GUID)) {
      return "P3";
    }
    if (equal_guids(guid, NV_ENC_PRESET_P4_GUID)) {
      return "P4";
    }
    if (equal_guids(guid, NV_ENC_PRESET_P5_GUID)) {
      return "P5";
    }
    if (equal_guids(guid, NV_ENC_PRESET_P6_GUID)) {
      return "P6";
    }
    if (equal_guids(guid, NV_ENC_PRESET_P7_GUID)) {
      return "P7";
    }
    return "Unknown";
  }

}  // namespace

namespace nvenc {

  nvenc_base::nvenc_base(NV_ENC_DEVICE_TYPE device_type, void *device, uint32_t max_width, uint32_t max_height, NV_ENC_BUFFER_FORMAT buffer_format):
      device_type(device_type),
      device(device),
      max_width(max_width),
      max_height(max_height),
      buffer_format(buffer_format) {
  }

  nvenc_base::~nvenc_base() {
    // Use destroy_base_resources() instead
  }

  bool
  nvenc_base::create_encoder(const nvenc_config &config, const video::config_t &client_config, const nvenc_colorspace_t &colorspace) {
    if (encoder) return false;
    if (!nvenc && !init_library()) return false;

    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS session_params = { NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER };
    session_params.device = device;
    session_params.deviceType = device_type;
    session_params.apiVersion = NVENCAPI_VERSION;
    if (nvenc->nvEncOpenEncodeSessionEx(&session_params, &encoder) != NV_ENC_SUCCESS) {
      BOOST_LOG(error) << "NvEncOpenEncodeSessionEx failed";
      return false;
    }

    uint32_t encode_guid_count = 0;
    if (nvenc->nvEncGetEncodeGUIDCount(encoder, &encode_guid_count) != NV_ENC_SUCCESS) {
      BOOST_LOG(error) << "NvEncGetEncodeGUIDCount failed: " << nvenc->nvEncGetLastErrorString(encoder);
      return false;
    };

    std::vector<GUID> encode_guids(encode_guid_count);
    if (nvenc->nvEncGetEncodeGUIDs(encoder, encode_guids.data(), encode_guids.size(), &encode_guid_count) != NV_ENC_SUCCESS) {
      BOOST_LOG(error) << "NvEncGetEncodeGUIDs failed: " << nvenc->nvEncGetLastErrorString(encoder);
      return false;
    }

    NV_ENC_INITIALIZE_PARAMS init_params = { NV_ENC_INITIALIZE_PARAMS_VER };

    switch (client_config.videoFormat) {
      case 0:
        // H.264
        init_params.encodeGUID = NV_ENC_CODEC_H264_GUID;
        break;

      case 1:
        // HEVC
        init_params.encodeGUID = NV_ENC_CODEC_HEVC_GUID;
        break;
    }

    {
      auto search_predicate = [&](const GUID &guid) {
        return equal_guids(init_params.encodeGUID, guid);
      };
      if (std::find_if(encode_guids.begin(), encode_guids.end(), search_predicate) == encode_guids.end()) {
        // Video format is not supported by the encoder
        return false;
      }
    }

    auto get_encoder_cap = [&](NV_ENC_CAPS cap) {
      NV_ENC_CAPS_PARAM param = { NV_ENC_CAPS_PARAM_VER, cap };
      int value = 0;
      nvenc->nvEncGetEncodeCaps(encoder, init_params.encodeGUID, &param, &value);
      return value;
    };

    auto buffer_is_10bit = [&]() {
      return buffer_format == NV_ENC_BUFFER_FORMAT_YUV420_10BIT || buffer_format == NV_ENC_BUFFER_FORMAT_YUV444_10BIT;
    };

    auto buffer_is_yuv444 = [&]() {
      return buffer_format == NV_ENC_BUFFER_FORMAT_YUV444 || buffer_format == NV_ENC_BUFFER_FORMAT_YUV444_10BIT;
    };

    if (max_width > get_encoder_cap(NV_ENC_CAPS_WIDTH_MAX) || max_height > get_encoder_cap(NV_ENC_CAPS_HEIGHT_MAX)) {
      // Encoder doesn't support requested dimensions
      return false;
    }

    if (buffer_is_10bit() && !get_encoder_cap(NV_ENC_CAPS_SUPPORT_10BIT_ENCODE)) {
      // Encoder doesn't support 10-bit
      return false;
    }

    if (buffer_is_yuv444() && !get_encoder_cap(NV_ENC_CAPS_SUPPORT_YUV444_ENCODE)) {
      // Encoder doesn't support yuv444
      return false;
    }

    supporting_ref_frame_invalidation = get_encoder_cap(NV_ENC_CAPS_SUPPORT_REF_PIC_INVALIDATION);

    init_params.maxEncodeWidth = max_width;
    init_params.maxEncodeHeight = max_height;
    init_params.presetGUID = quality_preset_guid_from_number(config.quality_preset);
    init_params.tuningInfo = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
    init_params.enablePTD = 1;

    init_params.encodeWidth = client_config.width;
    init_params.darWidth = client_config.width;
    init_params.encodeHeight = client_config.height;
    init_params.darHeight = client_config.height;
    init_params.frameRateNum = client_config.framerate;
    init_params.frameRateDen = 1;

    NV_ENC_PRESET_CONFIG preset_config = { NV_ENC_PRESET_CONFIG_VER, { NV_ENC_CONFIG_VER } };
    if (nvenc->nvEncGetEncodePresetConfigEx(encoder, init_params.encodeGUID, init_params.presetGUID, init_params.tuningInfo, &preset_config) != NV_ENC_SUCCESS) {
      BOOST_LOG(error) << "NvEncGetEncodePresetConfigEx failed: " << nvenc->nvEncGetLastErrorString(encoder);
      return false;
    }

    NV_ENC_CONFIG enc_config = preset_config.presetCfg;
    enc_config.profileGUID = NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID;
    enc_config.gopLength = NVENC_INFINITE_GOPLENGTH;
    enc_config.frameIntervalP = 1;
    enc_config.rcParams.enableAQ = 0;
    enc_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
    enc_config.rcParams.zeroReorderDelay = 1;
    enc_config.rcParams.enableLookahead = 0;
    enc_config.rcParams.enableNonRefP = 1;
    enc_config.rcParams.lowDelayKeyFrameScale = config.keyframe_vbv_multiplier > 1 ? config.keyframe_vbv_multiplier : 1;
    enc_config.rcParams.multiPass = config.multipass == multipass_e::two_pass_full_res    ? NV_ENC_TWO_PASS_FULL_RESOLUTION :
                                    config.multipass == multipass_e::two_pass_quarter_res ? NV_ENC_TWO_PASS_QUARTER_RESOLUTION :
                                                                                            NV_ENC_MULTI_PASS_DISABLED;
    enc_config.rcParams.averageBitRate = client_config.bitrate * 1000;

    if (get_encoder_cap(NV_ENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE)) {
      enc_config.rcParams.vbvBufferSize = client_config.bitrate * 1000 / client_config.framerate;
    }

    auto set_common_format_config = [&](auto &format_config) {
      format_config.repeatSPSPPS = 1;
      format_config.idrPeriod = NVENC_INFINITE_GOPLENGTH;
      format_config.sliceMode = 3;
      format_config.sliceModeData = client_config.slicesPerFrame;
      if (buffer_is_yuv444()) {
        format_config.chromaFormatIDC = 3;
      }
      format_config.enableFillerDataInsertion = config.filler_data_insertion;
    };

    auto fill_vui = [&colorspace](auto &vui_config) {
      vui_config.videoSignalTypePresentFlag = 1;
      vui_config.videoFormat = NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED;
      vui_config.videoFullRangeFlag = colorspace.full_range;
      vui_config.colourDescriptionPresentFlag = 1;
      vui_config.colourPrimaries = colorspace.primaries;
      vui_config.transferCharacteristics = colorspace.tranfer_function;
      vui_config.colourMatrix = colorspace.matrix;
    };

    switch (client_config.videoFormat) {
      case 0: {
        // H.264
        enc_config.profileGUID = buffer_is_yuv444() ? NV_ENC_H264_PROFILE_HIGH_444_GUID : NV_ENC_H264_PROFILE_HIGH_GUID;
        auto &format_config = enc_config.encodeCodecConfig.h264Config;
        set_common_format_config(format_config);
        format_config.entropyCodingMode = get_encoder_cap(NV_ENC_CAPS_SUPPORT_CABAC) ? NV_ENC_H264_ENTROPY_CODING_MODE_CABAC : NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC;
        if (client_config.numRefFrames > 0) {
          format_config.maxNumRefFrames = client_config.numRefFrames;
        }
        else {
          format_config.maxNumRefFrames = 5;
        }
        if (format_config.maxNumRefFrames > 0 && !get_encoder_cap(NV_ENC_CAPS_SUPPORT_MULTIPLE_REF_FRAMES)) {
          format_config.maxNumRefFrames = 1;
          supporting_ref_frame_invalidation = false;
        }
        format_config.numRefL0 = NV_ENC_NUM_REF_FRAMES_1;
        fill_vui(format_config.h264VUIParameters);
        break;
      }

      case 1: {
        // HEVC
        auto &format_config = enc_config.encodeCodecConfig.hevcConfig;
        set_common_format_config(format_config);
        if (buffer_is_10bit()) {
          format_config.pixelBitDepthMinus8 = 2;
        }
        if (client_config.numRefFrames > 0) {
          format_config.maxNumRefFramesInDPB = client_config.numRefFrames;
        }
        else {
          format_config.maxNumRefFramesInDPB = 5;
        }
        if (format_config.maxNumRefFramesInDPB > 0 && !get_encoder_cap(NV_ENC_CAPS_SUPPORT_MULTIPLE_REF_FRAMES)) {
          format_config.maxNumRefFramesInDPB = 1;
          supporting_ref_frame_invalidation = false;
        }
        format_config.numRefL0 = NV_ENC_NUM_REF_FRAMES_1;
        fill_vui(format_config.hevcVUIParameters);
        break;
      }
    }

    init_params.encodeConfig = &enc_config;

    if (nvenc->nvEncInitializeEncoder(encoder, &init_params) != NV_ENC_SUCCESS) {
      BOOST_LOG(error) << "NvEncInitializeEncoder failed: " << nvenc->nvEncGetLastErrorString(encoder);
      return false;
    }

    NV_ENC_CREATE_BITSTREAM_BUFFER create_bitstream_buffer = { NV_ENC_CREATE_BITSTREAM_BUFFER_VER };
    if (nvenc->nvEncCreateBitstreamBuffer(encoder, &create_bitstream_buffer) != NV_ENC_SUCCESS) {
      BOOST_LOG(error) << "NvEncCreateBitstreamBuffer failed: " << nvenc->nvEncGetLastErrorString(encoder);
      return false;
    }
    output_bitstream = create_bitstream_buffer.bitstreamBuffer;

    if (!create_input_buffer()) {
      return false;
    }

    BOOST_LOG(info) << "Created NvENC encoder at " << quality_preset_string_from_guid(init_params.presetGUID);

    return true;
  }

  nvenc_encoded_frame
  nvenc_base::encode_frame(uint64_t frame_index, bool force_idr) {
    if (!encoder || !output_bitstream) {
      return {};
    }

    auto input_buffer = get_input_buffer();

    if (!input_buffer) {
      return {};
    }

    NV_ENC_PIC_PARAMS pic_params = { NV_ENC_PIC_PARAMS_VER };
    pic_params.inputWidth = width;
    pic_params.inputHeight = height;
    pic_params.encodePicFlags = force_idr ? NV_ENC_PIC_FLAG_FORCEIDR : 0;
    pic_params.inputTimeStamp = frame_index;
    pic_params.inputBuffer = input_buffer;
    pic_params.outputBitstream = output_bitstream;
    pic_params.bufferFmt = buffer_format;
    pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;

    if (nvenc->nvEncEncodePicture(encoder, &pic_params) != NV_ENC_SUCCESS) {
      BOOST_LOG(error) << "NvEncEncodePicture failed: " << nvenc->nvEncGetLastErrorString(encoder);
      return {};
    }

    NV_ENC_LOCK_BITSTREAM lock_bitstream = { NV_ENC_LOCK_BITSTREAM_VER };
    lock_bitstream.outputBitstream = output_bitstream;
    lock_bitstream.doNotWait = false;

    if (nvenc->nvEncLockBitstream(encoder, &lock_bitstream) != NV_ENC_SUCCESS) {
      BOOST_LOG(error) << "NvEncLockBitstream failed: " << nvenc->nvEncGetLastErrorString(encoder);
      return {};
    }

    auto data_pointer = (uint8_t *) lock_bitstream.bitstreamBufferPtr;
    nvenc_encoded_frame result {
      { data_pointer, data_pointer + lock_bitstream.bitstreamSizeInBytes },
      lock_bitstream.outputTimeStamp,
      lock_bitstream.pictureType == NV_ENC_PIC_TYPE_IDR,
      ref_frame_invalidation_requested,
    };

    if (ref_frame_invalidation_requested) {
      // Invalidation request has been fullfilled, and video network packet will be marked as such
      ref_frame_invalidation_requested = false;
    }

    last_encoded_frame_index = frame_index;

    if (result.idr) {
      BOOST_LOG(info) << "idr " << result.frame_index;
    }

    if (nvenc->nvEncUnlockBitstream(encoder, lock_bitstream.outputBitstream) != NV_ENC_SUCCESS) {
      BOOST_LOG(error) << "NvEncUnlockBitstream failed: " << nvenc->nvEncGetLastErrorString(encoder);
      return {};
    }

    return result;
  }

  bool
  nvenc_base::invalidate_ref_frames(uint64_t first_frame, uint64_t last_frame) {
    if (!encoder || !supporting_ref_frame_invalidation) return false;

    if (last_frame < first_frame || last_encoded_frame_index < first_frame || last_encoded_frame_index > first_frame + 100) {
      BOOST_LOG(error) << "invalidate_ref_frames " << first_frame << "-" << last_frame << " invalid range (current frame " << last_encoded_frame_index << ")";
      return false;
    }

    if (first_frame >= last_ref_frame_invalidation_range.first && last_frame <= last_ref_frame_invalidation_range.second) {
      BOOST_LOG(info) << "invalidate_ref_frames " << first_frame << "-" << last_frame << " predicted";
      return true;
    }

    BOOST_LOG(info) << "invalidate_ref_frames " << first_frame << "-" << last_frame << " predicting " << first_frame << "-" << last_encoded_frame_index;

    ref_frame_invalidation_requested = true;
    last_ref_frame_invalidation_range = { first_frame, last_encoded_frame_index };

    bool result = true;
    for (auto i = first_frame; i <= last_encoded_frame_index; i++) {
      if (nvenc->nvEncInvalidateRefFrames(encoder, i) != NV_ENC_SUCCESS) {
        BOOST_LOG(error) << "NvEncInvalidateRefFrames " << i << " failed: " << nvenc->nvEncGetLastErrorString(encoder);
        result = false;
      }
    }

    return result;
  }

  void
  nvenc_base::destroy_base_resources() {
    if (output_bitstream) {
      nvenc->nvEncDestroyBitstreamBuffer(encoder, output_bitstream);
    }
    if (encoder) {
      nvenc->nvEncDestroyEncoder(encoder);
    }
  }

}  // namespace nvenc
