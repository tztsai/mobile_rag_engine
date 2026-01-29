/// Unified RAG engine with simplified initialization.
///
/// This class combines tokenizer, embedding model, and RAG service
/// initialization into a single `initialize()` call.
///
/// ## Quick Start
///
/// ```dart
/// import 'package:mobile_rag_engine/mobile_rag_engine.dart';
///
/// // NOTE: For most apps, use [MobileRag] singleton instead.
/// // It wraps this class and handles global access.
///
/// // If you need a standalone engine instance:
/// final rag = await RagEngine.initialize(
///   config: RagConfig.fromAssets(
///     tokenizerAsset: 'assets/tokenizer.json',
///     modelAsset: 'assets/model.onnx',
///   ),
/// );
///
/// // Use the engine
/// await rag.addDocument('Your document text here');
/// await rag.rebuildIndex();
/// final result = await rag.search('query', tokenBudget: 2000);
/// ```
library;

import 'dart:io';

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

import '../src/rust/api/tokenizer.dart';
import '../src/rust/api/source_rag.dart' show SourceStats;
import '../src/rust/api/db_pool.dart';
import 'embedding_service.dart';
import 'source_rag_service.dart';
import 'context_builder.dart';
import '../src/rust/api/hybrid_search.dart' as hybrid;
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';
import '../src/rust/frb_generated.dart';

/// Unified RAG engine with simplified initialization.
///
/// Wraps [SourceRagService] with automatic dependency initialization.
class RagEngine {
  static bool _isRustInitialized = false;

  /// Ensures RustLib is initialized (safe to call multiple times).
  static Future<void> _ensureRustInitialized() async {
    if (_isRustInitialized) return;

    if (Platform.isMacOS) {
      await RustLib.init(
        externalLibrary: ExternalLibrary.process(iKnowHowToUseIt: true),
      );
    } else {
      await RustLib.init();
    }
    _isRustInitialized = true;
  }

  final SourceRagService _ragService;

  /// Path to the SQLite database.
  final String dbPath;

  /// Vocabulary size of the loaded tokenizer.
  int? _vocabSize;
  int get vocabSize => _vocabSize ?? getVocabSize();

  RagEngine._({required SourceRagService ragService, required this.dbPath})
    : _ragService = ragService;

  /// Initialize RagEngine with all dependencies.
  ///
  /// This method handles:
  /// 1. Copying tokenizer asset to documents directory
  /// 2. Initializing the tokenizer
  /// 3. Loading the ONNX embedding model
  /// 4. Initializing the RAG database
  ///
  /// [config] - Configuration containing asset paths and options.
  /// [onProgress] - Optional callback for initialization status updates.
  ///
  /// Example:
  /// ```dart
  /// final rag = await RagEngine.initialize(
  ///   config: RagConfig.fromAssets(
  ///     tokenizerAsset: 'assets/tokenizer.json',
  ///     modelAsset: 'assets/model.onnx',
  ///   ),
  ///   onProgress: (status) => setState(() => _status = status),
  /// );
  /// ```
  static Future<RagEngine> initialize({
    String? tokenizerAsset,
    String? modelAsset,
    String? tokenizerPath,
    String? modelPath,
    String? databaseName,
    int maxChunkChars = 500,
    int overlapChars = 50,
    void Function(String status)? onProgress,
  }) async {
    // 0. Auto-initialize Rust library (safe to call multiple times)
    await _ensureRustInitialized();

    // 1. Get app documents directory
    final dir = await getApplicationDocumentsDirectory();

    try {
      // 2. Copy tokenizer asset to documents directory
      if (tokenizerPath == null) {
        tokenizerPath = "${dir.path}/tokenizer.json";
        await _copyAssetToFile(tokenizerAsset!, tokenizerPath);
      }

      // 3. Load ONNX embedding model from file or asset
      final modelBytes = modelPath == null
          ? (await rootBundle.load(modelAsset!)).buffer.asUint8List()
          : await File(modelPath).readAsBytes();

      // 4. Initialize tokenizer and embedding service
      await initTokenizerAndEmbedding(
        tokenizerPath: tokenizerPath,
        modelBytes: modelBytes,
        onProgress: onProgress,
      );
    } catch (e) {
      onProgress?.call(
        'Warning: Failed to initialize embedding model. Vector search will be disabled.',
      );
    }

    // 5. Initialize database connection pool
    onProgress?.call('Initializing connection pool...');
    final dbPath = "${dir.path}/${databaseName ?? 'rag.sqlite'}";
    await initDbPool(dbPath: dbPath, maxSize: 4);

    // 6. Initialize RAG service
    onProgress?.call('Initializing database...');
    final ragService = SourceRagService(
      dbPath: dbPath,
      maxChunkChars: maxChunkChars,
      overlapChars: overlapChars,
    );
    await ragService.init();

    onProgress?.call('Ready!');
    return RagEngine._(ragService: ragService, dbPath: dbPath);
  }

  /// Copy asset file to filesystem if it doesn't exist.
  static Future<void> _copyAssetToFile(
    String assetPath,
    String filePath,
  ) async {
    final file = File(filePath);
    if (!await file.exists()) {
      final data = await rootBundle.load(assetPath);
      await file.writeAsBytes(data.buffer.asUint8List());
    }
  }

  /// Initialize tokenizer and embedding service from file paths.
  ///
  /// This method can be called to initialize or reinitialize the tokenizer
  /// and embedding model from files on the filesystem. This is useful when:
  /// - First initializing the RAG engine
  /// - Models were downloaded after app launch
  ///
  /// Returns the vocabulary size of the tokenizer.
  ///
  /// [tokenizerPath] - Path to tokenizer.json file
  /// [modelBytes] - ONNX model bytes
  /// [onProgress] - Optional callback for status updates
  static Future<void> initTokenizerAndEmbedding({
    required String tokenizerPath,
    required Uint8List modelBytes,
    void Function(String status)? onProgress,
  }) async {
    onProgress?.call('Initializing tokenizer...');
    await initTokenizer(tokenizerPath: tokenizerPath);
    onProgress?.call('Loading embedding model...');
    await EmbeddingService.init(modelBytes);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Delegated methods from SourceRagService
  // ─────────────────────────────────────────────────────────────────────────

  /// Add a document with automatic chunking and embedding.
  ///
  /// The document is:
  /// 1. Split into chunks based on file type (auto-detected from [filePath])
  /// 2. Each chunk is embedded using the loaded model
  /// 3. Source and chunks are stored in the database
  ///
  /// Remember to call [rebuildIndex] after adding documents for optimal
  /// search performance.
  Future<SourceAddResult> addDocument(
    String content, {
    String? metadata,
    String? filePath,
    ChunkingStrategy? strategy,
    void Function(int done, int total)? onProgress,
  }) => _ragService.addSourceWithChunking(
    content,
    metadata: metadata,
    filePath: filePath,
    strategy: strategy,
    onProgress: onProgress,
  );

  /// Search for relevant chunks and assemble context for LLM.
  ///
  /// [query] - The search query text.
  /// [topK] - Number of top results to return (default: 10).
  /// [tokenBudget] - Maximum tokens for assembled context (default: 2000).
  /// [strategy] - Context assembly strategy (default: relevanceFirst).
  /// [adjacentChunks] - Include N chunks before/after matches (default: 0).
  /// [singleSourceMode] - Only include chunks from most relevant source.
  ///
  /// Returns empty result if embedding model is not available.
  Future<RagSearchResult> search(
    String query, {
    int topK = 10,
    int tokenBudget = 2000,
    ContextStrategy strategy = ContextStrategy.relevanceFirst,
    int adjacentChunks = 0,
    bool singleSourceMode = false,
  }) async {
    if (!isModelAvailable) {
      return RagSearchResult(chunks: [], context: AssembledContext.empty);
    }
    return _ragService.search(
      query,
      topK: topK,
      tokenBudget: tokenBudget,
      strategy: strategy,
      adjacentChunks: adjacentChunks,
      singleSourceMode: singleSourceMode,
    );
  }

  /// Hybrid search combining vector and keyword (BM25) search.
  ///
  /// Uses Reciprocal Rank Fusion (RRF) to combine semantic and keyword results.
  ///
  /// Returns empty result if embedding model is not available.
  Future<List<hybrid.HybridSearchResult>> searchHybrid(
    String query, {
    int topK = 10,
    double vectorWeight = 0.5,
    double bm25Weight = 0.5,
  }) async {
    if (!isModelAvailable) return [];
    return _ragService.searchHybrid(
      query,
      topK: topK,
      vectorWeight: vectorWeight,
      bm25Weight: bm25Weight,
    );
  }

  /// Hybrid search with context assembly for LLM.
  ///
  /// Returns empty result if embedding model is not available.
  Future<RagSearchResult> searchHybridWithContext(
    String query, {
    int topK = 10,
    int tokenBudget = 2000,
    ContextStrategy strategy = ContextStrategy.relevanceFirst,
    double vectorWeight = 0.5,
    double bm25Weight = 0.5,
  }) async {
    if (!isModelAvailable) {
      return RagSearchResult(chunks: [], context: AssembledContext.empty);
    }
    return _ragService.searchHybridWithContext(
      query,
      topK: topK,
      tokenBudget: tokenBudget,
      strategy: strategy,
      vectorWeight: vectorWeight,
      bm25Weight: bm25Weight,
    );
  }

  /// Rebuild the HNSW index after adding documents.
  ///
  /// Call this after adding one or more documents for optimal search
  /// performance. The index enables fast approximate nearest neighbor search.
  Future<void> rebuildIndex() => _ragService.rebuildIndex();

  /// Try to load a cached HNSW index from disk.
  ///
  /// Returns true if a previously built index exists.
  Future<bool> tryLoadCachedIndex() => _ragService.tryLoadCachedIndex();

  /// Save the HNSW index marker to disk.
  Future<void> saveIndex() => _ragService.saveIndex();

  /// Get statistics about stored sources and chunks.
  Future<SourceStats> getStats() => _ragService.getStats();

  /// Remove a source and all its chunks from the database.
  Future<void> removeSource(String sourceId) =>
      _ragService.removeSource(sourceId);

  /// Format search results as an LLM prompt.
  String formatPrompt(String query, RagSearchResult result) =>
      _ragService.formatPrompt(query, result);

  /// Regenerate embeddings for all existing chunks.
  ///
  /// Use this when the embedding model has been updated.
  Future<void> regenerateAllEmbeddings({
    void Function(int done, int total)? onProgress,
  }) => _ragService.regenerateAllEmbeddings(onProgress: onProgress);

  /// Access to the underlying [SourceRagService] for advanced operations.
  SourceRagService get service => _ragService;

  /// Dispose of resources.
  ///
  /// Call this when done using the engine to release resources.
  static void dispose() {
    EmbeddingService.dispose();
    closeDbPool();
  }

  /// Check if embedding model is available.
  ///
  /// Returns false if the model has not been loaded, meaning search operations
  /// will return empty results (zero embeddings produce no matches).
  static bool get isModelAvailable => EmbeddingService.isModelAvailable;
}
