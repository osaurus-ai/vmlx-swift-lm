// Copyright © 2026 osaurus.

import MLX
import MLXLMCommon
@testable import MLXLLM
import XCTest

final class ZayaForwardContractTests: XCTestCase {
    func testZayaModelSatisfiesLanguageModelArrayForwardForBatchEngine() {
        var text = ZayaTextConfiguration()
        text.hiddenSize = 4
        text.numHiddenLayers = 0
        text.numAttentionHeads = 1
        text.numKeyValueHeads = 1
        text.numQueryGroups = 1
        text.ccaNumQHeads = 1
        text.kvChannels = 4
        text.numExperts = 1
        text.moeRouterTopk = 1
        text.maxPositionEmbeddings = 32
        text.vocabSize = 16
        text.ffnHiddenSize = 4
        text.tieWordEmbeddings = true

        var configuration = ZayaConfiguration()
        configuration.textConfig = text

        let concrete = ZayaModel(configuration, moe: nil)
        let erased: any LanguageModel = concrete
        let input = MLXArray([[Int32(1), Int32(2)]])

        let logits = erased(input, cache: [])
        eval(logits)

        XCTAssertEqual(logits.dim(0), 1)
        XCTAssertEqual(logits.dim(1), 2)
        XCTAssertEqual(logits.dim(2), text.vocabSize)
    }
}
