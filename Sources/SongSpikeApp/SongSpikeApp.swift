// The app target only builds where SwiftUI exists. The #else branch lets
// `swift build` on a Linux host still link the package (same pattern as
// Perotta), even though the spike itself only means anything on a device.
#if canImport(SwiftUI)
import SwiftUI

@main
struct SongSpikeApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
#else
@main
enum SongSpikeApp {
    static func main() {
        print("Song Spike needs SwiftUI; build it for iOS with xtool.")
    }
}
#endif
