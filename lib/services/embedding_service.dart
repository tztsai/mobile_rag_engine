// lib/services/embedding_service.dart
import 'dart:typed_data';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:mobile_rag_engine/src/rust/api/tokenizer.dart';

/// Dart-based embedding service
/// Rust tokenizer + Flutter ONNX Runtime combination
class EmbeddingService {
  static OrtSession? _session;
  static bool _isModelAvailable = false;

  /// Debug mode flag
  static bool debugMode = false;

  /// Check if embedding model is available
  static bool get isModelAvailable => _isModelAvailable;

  /// Initialize ONNX model
  static Future<void> init(Uint8List modelBytes) async {
    OrtEnv.instance.init();
    final sessionOptions = OrtSessionOptions();
    _session = OrtSession.fromBuffer(modelBytes, sessionOptions);
    _isModelAvailable = true;
  }

  /// Convert text to embedding vector.
  ///
  /// NOTE: Dimension is determined by the model, NOT fixed.
  /// Common dimensions:
  /// - 384: nomic-embed-text-v1.5, bge-small-en-v1.5
  /// - 768: bge-base-en-v1.5, e5-base-v2
  /// - 1024: bge-large-en-v1.5, e5-large-v2
  ///
  /// When model is not available, returns a zero vector (384-dimensional).
  /// This allows the database to function before model download.
  static Future<List<double>> embed(String text) async {
    if (!_isModelAvailable) {
      // Return zero vector when model not available
      // 384 is the dimension of nomic-embed-text-v1.5 (default model)
      return List<double>.filled(384, 0.0);
    }

    if (_session == null) {
      throw Exception("EmbeddingService not initialized. Call init() first.");
    }

    // 1. Tokenize with Rust tokenizer
    final tokenIds = tokenize(text: text);

    if (debugMode) {
      print('[DEBUG] Text: "$text"');
      print('[DEBUG] Token IDs: $tokenIds (length: ${tokenIds.length})');
    }

    // 2. Generate attention_mask
    final seqLen = tokenIds.length;
    final attentionMask = List<int>.filled(seqLen, 1);

    // 3. Create ONNX input tensors
    final inputIdsData = Int64List.fromList(
      tokenIds.map((e) => e.toInt()).toList(),
    );
    final attentionMaskData = Int64List.fromList(
      attentionMask.map((e) => e.toInt()).toList(),
    );
    // BGE-m3 / many modern embedding models do not use token_type_ids
    // Passing it to a model that doesn't expect it causes an ONNX runtime error.

    final shape = [1, seqLen];

    final inputIdsTensor = OrtValueTensor.createTensorWithDataList(
      inputIdsData,
      shape,
    );
    final attentionMaskTensor = OrtValueTensor.createTensorWithDataList(
      attentionMaskData,
      shape,
    );

    // 4. Run inference
    final inputs = {
      'input_ids': inputIdsTensor,
      'attention_mask': attentionMaskTensor,
      // 'token_type_ids' removed
    };

    final runOptions = OrtRunOptions();
    final outputs = await _session!.runAsync(runOptions, inputs);

    // 5. Extract results and apply mean pooling
    final outputTensor = outputs?[0];
    if (outputTensor == null) {
      throw Exception("ONNX inference returned null output");
    }

    final outputData = outputTensor.value as List;

    if (debugMode) {
      print('[DEBUG] Output shape: ${_getShape(outputData)}');
    }

    // [1, seq_len, hidden_size] -> mean pooling -> [hidden_size]
    List<double> embedding;
    if (outputData.isNotEmpty && outputData[0] is List) {
      // 3D output: [batch, seq_len, hidden]
      final batchData = outputData[0] as List;
      if (batchData.isNotEmpty && batchData[0] is List) {
        final hiddenSize = (batchData[0] as List).length;
        embedding = List<double>.filled(hiddenSize, 0.0);

        // Apply mean pooling over all tokens (with attention mask)
        // Includes CLS and SEP - sentence-transformers default behavior
        int count = 0;
        for (int t = 0; t < batchData.length; t++) {
          // Only include tokens with attention_mask == 1
          if (t < attentionMask.length && attentionMask[t] == 1) {
            final tokenEmb = batchData[t] as List;
            for (int h = 0; h < hiddenSize; h++) {
              embedding[h] += (tokenEmb[h] as num).toDouble();
            }
            count++;
          }
        }

        if (count > 0) {
          for (int h = 0; h < hiddenSize; h++) {
            embedding[h] /= count;
          }
        }

        if (debugMode) {
          print('[DEBUG] Embedding (first 5): ${embedding.take(5).toList()}');
        }
      } else {
        // 2D output: [batch, hidden]
        embedding = (batchData).map((e) => (e as num).toDouble()).toList();
      }
    } else {
      // 1D output: [hidden]
      embedding = outputData.map((e) => (e as num).toDouble()).toList();
    }

    // 6. Release resources
    inputIdsTensor.release();
    attentionMaskTensor.release();
    runOptions.release();
    for (final output in outputs ?? []) {
      output?.release();
    }

    return embedding;
  }

  /// Batch embed multiple texts (sequential processing)
  ///
  /// ONNX session doesn't support concurrent access, so processing is sequential
  ///
  /// [texts]: List of texts to embed
  /// [onProgress]: Progress callback (completed count, total count)
  static Future<List<List<double>>> embedBatch(
    List<String> texts, {
    int concurrency = 1, // Sequential due to ONNX session limitation
    void Function(int completed, int total)? onProgress,
  }) async {
    if (!_isModelAvailable) {
      // Return zero vectors when model not available
      // 384 is the dimension of nomic-embed-text-v1.5 (default model)
      return texts.map((_) => List<double>.filled(384, 0.0)).toList();
    }

    if (_session == null) {
      throw Exception("EmbeddingService not initialized. Call init() first.");
    }

    if (texts.isEmpty) return [];

    final results = <List<double>>[];

    // Sequential processing (ONNX session is not thread-safe)
    for (var i = 0; i < texts.length; i++) {
      final embedding = await embed(texts[i]);
      results.add(embedding);
      onProgress?.call(i + 1, texts.length);
    }

    return results;
  }

  /// Release resources
  static void dispose() {
    _session?.release();
    OrtEnv.instance.release();
  }

  /// Get array shape as string
  static String _getShape(dynamic data) {
    if (data is! List) return 'scalar';
    if (data.isEmpty) return '[0]';
    if (data[0] is! List) return '[${data.length}]';
    if (data[0].isEmpty) return '[${data.length}, 0]';
    if (data[0][0] is! List) return '[${data.length}, ${data[0].length}]';
    return '[${data.length}, ${data[0].length}, ${data[0][0].length}]';
  }
}
