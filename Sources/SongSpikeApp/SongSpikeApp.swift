// The app target only builds where SwiftUI exists. The #else branch lets
// `swift build` on a Linux host still link the package (same pattern as
// Perotta), even though the spike itself only means anything on a device.
#if canImport(SwiftUI)
import SwiftUI
#if canImport(UIKit)
import UIKit

/// Minimal app delegate: only the background-URL-session events hook,
/// so iOS can hand the model downloader the completion handler it must
/// invoke once every pending background download event is delivered.
final class SpikeAppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        handleEventsForBackgroundURLSession identifier: String,
        completionHandler: @escaping () -> Void
    ) {
        ModelDownloader.shared.setBackgroundCompletionHandler(completionHandler, for: identifier)
    }
}
#endif

@main
struct SongSpikeApp: App {
    #if canImport(UIKit)
    @UIApplicationDelegateAdaptor(SpikeAppDelegate.self) private var appDelegate
    #endif

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
