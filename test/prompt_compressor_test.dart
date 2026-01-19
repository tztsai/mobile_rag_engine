// test/prompt_compressor_test.dart
//
// Unit tests for PromptCompressor service

import 'package:flutter_test/flutter_test.dart';
import 'package:mobile_rag_engine/services/prompt_compressor.dart';
import 'package:mobile_rag_engine/src/rust/api/compression_utils.dart';

void main() {
  group('CompressionLevel', () {
    test('should have correct index values', () {
      expect(CompressionLevel.minimal.index, 0);
      expect(CompressionLevel.balanced.index, 1);
      expect(CompressionLevel.aggressive.index, 2);
    });
  });

  group('CompressedContext', () {
    test('should calculate ratio correctly', () {
      const context = CompressedContext(
        text: 'Compressed text',
        originalChars: 100,
        compressedChars: 50,
        ratio: 0.5,
        estimatedTokensSaved: 12,
        includedChunks: [],
      );

      expect(context.ratio, 0.5);
      expect(context.estimatedTokensSaved, 12);
    });

    test('toString should include ratio percentage', () {
      const context = CompressedContext(
        text: 'Test',
        originalChars: 100,
        compressedChars: 75,
        ratio: 0.75,
        estimatedTokensSaved: 6,
        includedChunks: [],
      );

      expect(context.toString(), contains('75.0%'));
    });
  });

  group('CompressionOptions', () {
    test('should create with correct parameters', () {
      const options = CompressionOptions(
        removeStopwords: true,
        removeDuplicates: true,
        language: 'ko',
        level: 1,
      );

      expect(options.removeStopwords, true);
      expect(options.removeDuplicates, true);
      expect(options.language, 'ko');
      expect(options.level, 1);
    });

    test('should have equality based on properties', () {
      const options1 = CompressionOptions(
        removeStopwords: true,
        removeDuplicates: true,
        language: 'ko',
        level: 1,
      );
      const options2 = CompressionOptions(
        removeStopwords: true,
        removeDuplicates: true,
        language: 'ko',
        level: 1,
      );

      expect(options1, options2);
    });
  });

  // Phase 2 tests
  group('ScoredSentence', () {
    test('should create with correct properties', () {
      const scored = ScoredSentence(
        sentence: 'Test sentence.',
        similarity: 0.85,
        index: 3,
      );

      expect(scored.sentence, 'Test sentence.');
      expect(scored.similarity, 0.85);
      expect(scored.index, 3);
    });

    test('toString should include similarity', () {
      const scored = ScoredSentence(
        sentence: 'Hello',
        similarity: 0.756,
        index: 0,
      );

      expect(scored.toString(), contains('0.756'));
    });
  });

  group('sqrt helper', () {
    test('should calculate square root correctly', () {
      expect(sqrt(4.0), closeTo(2.0, 0.001));
      expect(sqrt(9.0), closeTo(3.0, 0.001));
      expect(sqrt(2.0), closeTo(1.414, 0.01));
    });

    test('should handle zero and negative values', () {
      expect(sqrt(0.0), 0.0);
      expect(sqrt(-1.0), 0.0);
    });
  });
}
