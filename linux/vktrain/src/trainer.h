// The training loop of splatwork-vk-train: one dataset, one TrainingConfig, the behaviour of the CUDA worker's
// linux/splatworker/trainer.py train() on the Vulkan kernels of gpu_splat.*. Progress, previews, checkpoints,
// pause / resume and cancel follow subprocess_entry's protocol, as JSON lines on stdout (see main.cpp).
#pragma once
#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>

#include "config.h"
#include "nlohmann/json.hpp"
#include "vk.h"

namespace vkt {

// Requests from the worker (stdin commands or signals), polled once per step.
struct Control {
    std::atomic<bool> cancel{false}, pause{false}, checkpoint{false};
};

// JSON-lines output: {"type": "status" | "log" | "preview" | "checkpoint" | "done" | "paused" | "cancelled" |
// "error", ...} — the tuples subprocess_entry puts on its queue.
class Emitter {
public:
    explicit Emitter(FILE* out = stdout) : out_(out) {}
    void emit(const nlohmann::json& j);
    void log(const std::string& msg) { emit({{"type", "log"}, {"message", msg}}); }
    void status(nlohmann::json fields) { fields["type"] = "status"; emit(fields); }
private:
    FILE* out_;
    std::mutex mu_;
};

struct TrainOptions {
    std::string datasetDir;
    std::string outPath;          // result PLY
    std::string jobDir;           // preview.splat, checkpoint.ply, resume.vkstate (empty: none)
    std::string resumeFrom;       // --resume FILE: continue from this saved state
    int64_t previewEvery = -1;    // overrides previewInterval when ≥ 0
    ContextOptions ctx;
    bool opacityReset = false;    // adaptive strategy: textbook opacity reset (gsplat 1.5's never fires, see trainer.cpp)
    int64_t pauseAt = -1;         // tests: pause before this step, as if "pause" had arrived
    int64_t traceEvery = 0;       // tests: {"type": "trace", step, loss, psnr, numGaussians} every N steps
    std::string resumePath() const;   // where a pause saves the state
};

enum ExitCode { EXIT_DONE = 0, EXIT_ERROR = 1, EXIT_USAGE = 2, EXIT_PAUSED = 3, EXIT_CANCELLED = 4 };

int runTraining(const TrainOptions& opt, const TrainConfig& cfg, Control& ctl, Emitter& em);

}  // namespace vkt
