import Foundation

/// Pause / Resume for a run whose training happens in several places at once (block-parallel, or one worker job):
/// it reaches every Metal trainer on this Mac and every worker job that is running, and anything that starts
/// while paused starts paused.
///
/// A worker pause saves the job's state and frees the GPU; a worker resumes another computer's paused job by
/// itself after an hour (`REMOTE_PAUSE_LIMIT` on the worker), so a long pause re-sends it (see `jobUpdated`).
final class PauseController: @unchecked Sendable {
    private let lock = NSLock()
    private var paused = false
    private var trainers: [ObjectIdentifier: MetalTrainer] = [:]
    private var jobs: [String: WorkerClient] = [:]
    private var lastPauseSent: [String: Date] = [:]

    var isPaused: Bool { lock.lock(); defer { lock.unlock() }; return paused }

    func set(paused value: Bool) {
        lock.lock()
        paused = value
        let ts = Array(trainers.values), js = jobs
        if value { for id in js.keys { lastPauseSent[id] = Date() } }
        lock.unlock()
        for t in ts { t.setPaused(value) }
        for (id, client) in js { Task.detached { await self.send(value, id, client) } }
    }

    // MARK: Metal trainers on this Mac

    func add(_ t: MetalTrainer) {
        lock.lock(); trainers[ObjectIdentifier(t)] = t; let p = paused; lock.unlock()
        if p { t.setPaused(true) }
    }
    func remove(_ t: MetalTrainer) { lock.lock(); trainers.removeValue(forKey: ObjectIdentifier(t)); lock.unlock() }

    // MARK: Worker jobs

    func add(job id: String, on client: WorkerClient) {
        lock.lock(); jobs[id] = client; let p = paused; if p { lastPauseSent[id] = Date() }; lock.unlock()
        if p { Task.detached { await self.send(true, id, client) } }
    }
    func remove(job id: String) { lock.lock(); jobs.removeValue(forKey: id); lastPauseSent.removeValue(forKey: id); lock.unlock() }

    /// From the job's status polls: while paused, a job the worker put back in the queue or started again (it
    /// auto-resumes after an hour, or the pause arrived while it was still uploading) is paused again, at most
    /// once a minute.
    func jobUpdated(_ id: String, state: JobState) {
        lock.lock()
        guard paused, let client = jobs[id], state != .paused, !state.isTerminal,
              Date().timeIntervalSince(lastPauseSent[id] ?? .distantPast) > 60 else { lock.unlock(); return }
        lastPauseSent[id] = Date()
        lock.unlock()
        Task.detached { await self.send(true, id, client) }
    }

    /// Workers without pause support answer 404/409; that's ignored (the job just keeps running). Resume is
    /// refused (409) while the job is still saving its state, so it's retried for up to a minute.
    private func send(_ pause: Bool, _ id: String, _ client: WorkerClient) async {
        if pause { _ = try? await client.pauseJob(id); return }
        for _ in 0..<30 where !isPaused {      // paused again meanwhile: leave it paused
            do { try await client.resumeJob(id); return } catch {
                guard (error as? WorkerError)?.status == 409 else { return }
                if let s = try? await client.job(id), s.state.isTerminal { return }
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
        }
    }
}
