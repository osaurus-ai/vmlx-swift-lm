// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Iter 46: regression tests for `UserInput` init semantics.
//
// Motivation: iter 45 found that `UserInput(prompt: String, images: [...])`
// silently dropped the images because Swift's `didSet` observer on the
// `prompt` property does NOT fire during init, and the init didn't
// manually assign `self.images` / `self.videos`. Every VLM processor
// branches on `input.images.isEmpty` to decide whether to run the vision
// path, so the bug manifested as "VL model hallucinates against unseen
// image" — an incredibly subtle user-facing correctness failure.
//
// These tests pin the invariant as pure value-type property checks: no
// model required, runs in milliseconds, would have caught iter 45's bug
// on the first CI run.

import CoreImage
import Foundation
@preconcurrency import MLX
import XCTest

@testable import MLXLMCommon

final class UserInputInitSemanticsTests: XCTestCase {

    /// Deterministic 4×4 CIImage — no pixel content needed; only identity.
    private func dummyImage() -> UserInput.Image {
        let bytes = [UInt8](repeating: 200, count: 4 * 4 * 4)
        let data = Data(bytes)
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let ci = CIImage(
            bitmapData: data, bytesPerRow: 4 * 4,
            size: .init(width: 4, height: 4),
            format: .RGBA8, colorSpace: cs
        )
        return .ciImage(ci)
    }

    // MARK: - init(prompt: String, images:, videos:)

    /// The exact bug iter 45 closed. `UserInput(prompt: "...", images: [img])`
    /// must expose the image on `self.images`, not only inside the
    /// wrapped chat message.
    func testPromptImagesInitExposesImagesTopLevel() {
        let img = dummyImage()
        let input = UserInput(prompt: "hi", images: [img])

        XCTAssertFalse(input.images.isEmpty,
            "UserInput.images must be populated after `init(prompt:images:)`. " +
            "If this test fails, VLM processors will hit the text-only branch " +
            "and silently discard every image passed in.")
        XCTAssertEqual(input.images.count, 1,
            "Exactly one image was passed in — exactly one must be exposed.")
    }

    /// Videos take the same init path and must round-trip the same way.
    func testPromptVideosInitExposesVideosTopLevel() {
        let video = UserInput.Video.url(URL(fileURLWithPath: "/tmp/nonexistent.mov"))
        let input = UserInput(prompt: "hi", videos: [video])
        XCTAssertEqual(input.videos.count, 1,
            "UserInput.videos must be populated after `init(prompt:videos:)`.")
    }

    /// Mixed prompt + image + video: both media collections populated.
    func testPromptImagesAndVideosInitExposesBoth() {
        let img = dummyImage()
        let video = UserInput.Video.url(URL(fileURLWithPath: "/tmp/x.mov"))
        let input = UserInput(prompt: "hi", images: [img], videos: [video])
        XCTAssertEqual(input.images.count, 1)
        XCTAssertEqual(input.videos.count, 1)
    }

    /// Multiple images — loop preserves count.
    func testPromptImagesInitPreservesAllImages() {
        let imgs = (0..<5).map { _ in dummyImage() }
        let input = UserInput(prompt: "hi", images: imgs)
        XCTAssertEqual(input.images.count, 5)
    }

    /// No-image construction must ALSO produce empty (no accidental extra
    /// images from init-side defaults).
    func testPromptOnlyInitLeavesImagesEmpty() {
        let input = UserInput(prompt: "hi")
        XCTAssertTrue(input.images.isEmpty,
            "Text-only prompt construction must not fabricate images.")
        XCTAssertTrue(input.videos.isEmpty)
    }

    // MARK: - init(chat:) — already correct, guard against regression

    /// The `init(chat:)` overload was already correct prior to iter 45
    /// (had explicit `self.images = chat.reduce(...)`). Pin it here so
    /// no later refactor accidentally removes that extraction.
    func testChatInitExtractsImagesFromMessages() {
        let img = dummyImage()
        let chat: [Chat.Message] = [
            .system("system msg"),
            .user("user msg", images: [img]),
        ]
        let input = UserInput(chat: chat)
        XCTAssertEqual(input.images.count, 1,
            "Chat messages carrying images must be flattened onto self.images.")
    }

    // MARK: - didSet still fires on subsequent mutations

    /// Once the UserInput exists, mutating `.prompt` (not through init)
    /// MUST trigger `didSet` to re-extract images. Confirm that behaviour
    /// is preserved.
    func testPromptMutationTriggersImageReExtraction() {
        var input = UserInput(prompt: "hi")
        XCTAssertTrue(input.images.isEmpty)

        let img = dummyImage()
        input.prompt = .chat([
            .user("now with image", images: [img])
        ])

        XCTAssertEqual(input.images.count, 1,
            "Assigning `self.prompt = .chat(...)` after init must trigger " +
            "didSet re-extraction of images. Without this, multi-turn chats " +
            "that swap the prompt mid-conversation would lose images.")
    }

    /// Switching from chat-with-images back to a non-chat prompt should
    /// NOT clear images (the didSet branches on .chat only). This pins
    /// the current behaviour — callers that need to clear images on
    /// prompt-type change must do so explicitly.
    func testPromptSwitchToTextPromptDoesNotClearImages() {
        let img = dummyImage()
        var input = UserInput(prompt: "seed", images: [img])
        XCTAssertEqual(input.images.count, 1)

        input.prompt = .text("plain text prompt now")
        XCTAssertEqual(input.images.count, 1,
            "Current contract: switching to .text does not reset images. " +
            "Changing this would be a semantic break — update this test " +
            "intentionally.")
    }

    // MARK: - init(messages:)

    /// The `messages:` init was already correct; guard regression.
    func testMessagesInitExposesImages() {
        let img = dummyImage()
        let input = UserInput(messages: [], images: [img])
        XCTAssertEqual(input.images.count, 1)
    }

    // MARK: - init(prompt: Prompt, images:)

    /// The preconfigured-Prompt init was already correct; guard regression.
    func testPromptEnumInitExposesImages() {
        let img = dummyImage()
        let input = UserInput(
            prompt: .text("text"),
            images: [img]
        )
        XCTAssertEqual(input.images.count, 1)
    }
}
