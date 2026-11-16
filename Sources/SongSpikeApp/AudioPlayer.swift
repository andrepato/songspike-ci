import AVFoundation
import Foundation

/// Minimal WAV playback for the last generated song.
@MainActor
final class AudioPlayer: NSObject, ObservableObject {
    @Published private(set) var isPlaying = false
    @Published private(set) var loadedURL: URL?

    private var player: AVAudioPlayer?

    func load(url: URL) {
        stop()
        do {
            player = try AVAudioPlayer(contentsOf: url)
            player?.delegate = self
            loadedURL = url
        } catch {
            player = nil
            loadedURL = nil
        }
    }

    func togglePlayPause() {
        guard let player else { return }
        if player.isPlaying {
            player.pause()
            isPlaying = false
        } else {
            player.play()
            isPlaying = true
        }
    }

    func stop() {
        player?.stop()
        player = nil
        isPlaying = false
        loadedURL = nil
    }

    fileprivate func playbackFinished() {
        isPlaying = false
    }
}

extension AudioPlayer: AVAudioPlayerDelegate {
    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in
            self.playbackFinished()
        }
    }
}
