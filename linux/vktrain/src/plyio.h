// PLY reading (initial point clouds) and writing (INRIA 3DGS gaussian PLY), and the live-preview .splat file:
// the same layouts as linux/splatworker/ply.py (write_gaussian_ply, read_point_cloud) and trainer.py
// export_preview_splat, so the worker, the server and the Mac read the Vulkan trainer's files unchanged.
#pragma once
#include <string>
#include <vector>

#include "scene.h"

namespace vkt {

// xyz (3 per point) and rgb in 0..1 (3 per point; empty when the file has no colours). Non-finite points dropped.
void readPointCloud(const std::string& path, std::vector<float>& xyz, std::vector<float>& rgb);

// Splats `keep` (all when empty) of p as a 3DGS PLY, written atomically (tmp + rename). SH coefficients above
// shDegree are written as zero; `up` (3 values, may be null) becomes "comment splatstudio_up x y z".
// Returns the number of splats written.
size_t writeGaussianPly(const std::string& path, const Params& p, int shDegree, const double* up,
                        const std::vector<uint8_t>* keep = nullptr);

constexpr size_t PREVIEW_MAX_SPLATS = 400000;
// antimatter15 .splat (32 bytes per splat, no SH): the maxSplats most visible splats (opacity × volume).
size_t writePreviewSplat(const std::string& path, const Params& p, size_t maxSplats = PREVIEW_MAX_SPLATS);

}  // namespace vkt
