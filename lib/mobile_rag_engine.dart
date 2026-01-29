/// Mobile RAG Engine
///
/// A high-performance, on-device RAG (Retrieval-Augmented Generation) engine
/// for Flutter. Run semantic search completely offline on iOS and Android.
///
/// ## Quick Start (Recommended)
///
/// ```dart
/// import 'package:mobile_rag_engine/mobile_rag_engine.dart';
///
/// ```dart
/// import 'package:mobile_rag_engine/mobile_rag_engine.dart';
///
/// void main() async {
///   WidgetsFlutterBinding.ensureInitialized();
///
///   // Initialize (just 1 line!)
///   await MobileRag.initialize(
///     tokenizerAsset: 'assets/tokenizer.json',
///     modelAsset: 'assets/model.onnx',
///   );
///
///   runApp(const MyApp());
/// }
///
/// // Add documents
/// await MobileRag.instance.addDocument('Your long document text here');
/// await MobileRag.instance.rebuildIndex();
///
/// // Search with LLM context assembly
/// final result = await MobileRag.instance.search('query', tokenBudget: 2000);
/// final prompt = MobileRag.instance.formatPrompt('query', result);
/// // Send prompt to LLM
/// ```
///
/// ## Advanced Usage (Low-Level API)
///
/// For fine-grained control, use the individual services directly:
///
/// ```dart
/// await initTokenizer(tokenizerPath: 'path/to/tokenizer.json');
/// await EmbeddingService.init(modelBytes);
/// final rag = SourceRagService(dbPath: 'path/to/rag.db');
/// await rag.init();
/// ```
library;

// ✅ Primary API (Singleton Pattern)
export 'mobile_rag.dart';

// High-level services (for advanced usage)
export 'services/rag_engine.dart';
export 'services/context_builder.dart';
export 'services/source_rag_service.dart';
export 'services/embedding_service.dart';

// Document parsing (commonly used)
export 'src/rust/api/document_parser.dart';

// Note: Low-level Rust exports are hidden by default for better DX.
// If you need them, import 'package:mobile_rag_engine/src/rust/api/...' directly.
