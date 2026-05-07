// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Testing

@testable import MLXLMCommon

@Suite("Runtime acceleration flag")
struct RuntimeAccelerationTests {
    @Test("flag parser accepts public values and aliases")
    func testFlagParsing() {
        #expect(AccelerationMode(flagValue: "metal") == .metal)
        #expect(AccelerationMode(flagValue: "gpu") == .metal)
        #expect(AccelerationMode(flagValue: "auto") == .auto)
        #expect(AccelerationMode(flagValue: "ane-coreml") == .aneCoreML)
        #expect(AccelerationMode(flagValue: "coreml_ane") == .aneCoreML)
        #expect(AccelerationMode(flagValue: "bogus") == .invalid("bogus"))
    }

    @Test("environment lookup defaults to metal")
    func testEnvironmentLookup() {
        #expect(AccelerationRuntime.requestedMode(environment: [:]) == .metal)
        #expect(AccelerationRuntime.requestedMode(environment: [
            AccelerationMode.environmentVariable: "auto"
        ]) == .auto)
        #expect(AccelerationRuntime.requestedMode(environment: [
            AccelerationMode.environmentVariable: "ane-coreml"
        ]) == .aneCoreML)
    }

    @Test("text decode auto keeps MLX Metal without a validated island")
    func testTextDecodeAutoFallsBackToMetal() throws {
        let decision = try AccelerationRuntime.resolveTextDecode(.auto)
        #expect(decision == .metal(reason: "no-validated-coreml-island"))
    }

    @Test("text decode ane-coreml fails closed without a validated island")
    func testTextDecodeAneCoreMLRequiresIsland() {
        do {
            _ = try AccelerationRuntime.resolveTextDecode(.aneCoreML)
            Issue.record("ane-coreml must not silently fall back to Metal")
        } catch let error as AccelerationError {
            #expect(error == .noValidatedCoreMLIsland(
                mode: .aneCoreML, target: .textDecode))
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test("invalid flag fails closed")
    func testInvalidFlagFailsClosed() {
        do {
            _ = try AccelerationRuntime.resolveTextDecode(.invalid("neural"))
            Issue.record("invalid accelerator flag must fail")
        } catch let error as AccelerationError {
            #expect(error == .invalidMode("neural"))
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test("GenerateParameters can carry an explicit accelerator request")
    func testGenerateParametersCarriesMode() {
        let params = GenerateParameters(accelerationMode: .aneCoreML)
        #expect(params.accelerationMode == .aneCoreML)
    }
}
