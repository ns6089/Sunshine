#pragma once

namespace nvenc {

  enum class multipass_e {
    one_pass,
    two_pass_quarter_res,
    two_pass_full_res,
  };

  struct nvenc_config {
    unsigned quality_preset = 1;  // Quality preset from 1 to 7
    unsigned keyframe_vbv_multiplier = 1;  // Allows I-frames to break normal VBV constraints
    multipass_e multipass = multipass_e::one_pass;
    bool filler_data_insertion = false;
  };

}  // namespace nvenc
