import Foundation

struct DictationTextDeliveryPolicy {
    let shouldCopyToClipboard: Bool
    let shouldTypeExternally: Bool

    init(
        deliversText: Bool,
        accessibilityTrusted: Bool,
        isFluidFrontmost: Bool,
        sendsExistingDraft: Bool = false,
        copyPreference: Bool,
        aiFailed: Bool = false
    ) {
        self.shouldCopyToClipboard = deliversText && !sendsExistingDraft &&
            (!accessibilityTrusted || copyPreference || aiFailed)
        self.shouldTypeExternally = deliversText && accessibilityTrusted && !isFluidFrontmost
    }
}
