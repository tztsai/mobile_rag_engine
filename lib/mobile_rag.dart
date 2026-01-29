/// Main entry point for Mobile RAG Engine.
///
/// Provides a singleton pattern for easy access throughout your app.
/// Initialize once in main(), use anywhere via [MobileRag.instance].
///
/// ## Quick Start
///
/// ```dart
/// import 'package:mobile_rag_engine/mobile_rag_engine.dart';
///
/// void main() async {
///   WidgetsFlutterBinding.ensureInitialized();
///
///   await MobileRag.initialize(
///     tokenizerAsset: 'assets/tokenizer.json',
///     modelAsset: 'assets/model.onnx',
///   );
///
///   runApp(const MyApp());
/// }
///
/// // Later, anywhere in your app:
/// final result = await MobileRag.instance.search('What is Flutter?');
/// ```
library;

import 'services/rag_config.dart';
import 'services/rag_engine.dart';
import 'services/context_builder.dart';
import 'services/source_rag_service.dart';
import 'src/rust/api/hybrid_search.dart' as hybrid;

/// Singleton facade for Mobile RAG Engine.
///
/// Use [MobileRag.initialize] to set up the engine, then access it
/// via [MobileRag.instance] anywhere in your app.
class MobileRag {
  static MobileRag? _instance;
  static RagEngine? _engine;

  MobileRag._();

  /// Initialize the RAG engine. Call once in main().
  ///
  /// This will:
  /// 1. Initialize the Rust library (FFI)
  /// 2. Load the tokenizer from assets
  /// 3. Load the ONNX embedding model
  /// 4. Initialize the SQLite database
  ///
  /// [tokenizerAsset] - Path to tokenizer.json in assets
  /// [modelAsset] - Path to ONNX model file in assets
  /// [databaseName] - Optional custom database name (default: 'rag.sqlite')
  /// [onProgress] - Optional callback for initialization progress
  ///
  /// Example:
  /// ```dart
  /// await MobileRag.initialize(
  ///   tokenizerAsset: 'assets/tokenizer.json',
  ///   modelAsset: 'assets/model.onnx',
  ///   onProgress: (status) => print(status),
  /// );
  /// ```
  static Future<void> initialize({
    String? tokenizerAsset,
    String? modelAsset,
    String? tokenizerPath,
    String? modelPath,
    String? databaseName,
    int maxChunkChars = 500,
    int overlapChars = 50,
    void Function(String status)? onProgress,
  }) async {
    if (_instance != null) {
      onProgress?.call('Already initialized');
      return;
    }

    _engine = await RagEngine.initialize(
      tokenizerAsset: tokenizerAsset,
      modelAsset: modelAsset,
      tokenizerPath: tokenizerPath,
      modelPath: modelPath,
      databaseName: databaseName,
      maxChunkChars: maxChunkChars,
      overlapChars: overlapChars,
      onProgress: onProgress,
    );
    _instance = MobileRag._();
  }

  /// Check if the engine is initialized.
  static bool get isInitialized => _instance != null;

  /// Get the singleton instance.
  ///
  /// Throws [StateError] if [initialize] has not been called.
  static MobileRag get instance {
    if (_instance == null) {
      throw StateError(
        'MobileRag not initialized. Call MobileRag.initialize() first.',
      );
    }
    return _instance!;
  }

  /// Access the underlying [RagEngine] for advanced operations.
  RagEngine get engine => _engine!;

  /// Path to the SQLite database.
  String get dbPath => _engine!.dbPath;

  /// Vocabulary size of the loaded tokenizer.
  int get vocabSize => _engine!.vocabSize;

  // ─────────────────────────────────────────────────────────────────────────
  // Convenience instance methods (delegate to RagEngine)
  // ─────────────────────────────────────────────────────────────────────────

  /// Add a document with automatic chunking and embedding.
  ///
  /// The document is split into chunks, embedded, and stored.
  /// Remember to call [rebuildIndex] after adding documents.
  Future<SourceAddResult> addDocument(
    String content, {
    String? metadata,
    String? filePath,
    ChunkingStrategy? strategy,
    void Function(int done, int total)? onProgress,
  }) => _engine!.addDocument(
    content,
    metadata: metadata,
    filePath: filePath,
    strategy: strategy,
    onProgress: onProgress,
  );

  /// Search for relevant chunks and assemble context for LLM.
  Future<RagSearchResult> search(
    String query, {
    int topK = 10,
    int tokenBudget = 2000,
    ContextStrategy strategy = ContextStrategy.relevanceFirst,
    int adjacentChunks = 0,
    bool singleSourceMode = false,
  }) => _engine!.search(
    query,
    topK: topK,
    tokenBudget: tokenBudget,
    strategy: strategy,
    adjacentChunks: adjacentChunks,
    singleSourceMode: singleSourceMode,
  );

  /// Hybrid search combining vector and keyword (BM25) search.
  Future<List<hybrid.HybridSearchResult>> searchHybrid(
    String query, {
    int topK = 10,
    double vectorWeight = 0.5,
    double bm25Weight = 0.5,
  }) => _engine!.searchHybrid(
    query,
    topK: topK,
    vectorWeight: vectorWeight,
    bm25Weight: bm25Weight,
  );

  /// Hybrid search with context assembly for LLM.
  Future<RagSearchResult> searchHybridWithContext(
    String query, {
    int topK = 10,
    int tokenBudget = 2000,
    ContextStrategy strategy = ContextStrategy.relevanceFirst,
    double vectorWeight = 0.5,
    double bm25Weight = 0.5,
  }) => _engine!.searchHybridWithContext(
    query,
    topK: topK,
    tokenBudget: tokenBudget,
    strategy: strategy,
    vectorWeight: vectorWeight,
    bm25Weight: bm25Weight,
  );

  /// Rebuild the HNSW index after adding documents.
  Future<void> rebuildIndex() => _engine!.rebuildIndex();

  /// Try to load a cached HNSW index.
  Future<bool> tryLoadCachedIndex() => _engine!.tryLoadCachedIndex();

  /// Format search results as an LLM prompt.
  String formatPrompt(String query, RagSearchResult result) =>
      _engine!.formatPrompt(query, result);

  /// Dispose of resources.
  ///
  /// Call when completely done with the engine.
  static void dispose() {
    RagEngine.dispose();
    _engine = null;
    _instance = null;
  }
}
