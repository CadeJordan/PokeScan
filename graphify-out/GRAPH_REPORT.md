# Graph Report - PokeScan  (2026-10-15)

## Corpus Check
- 68 files · ~67,795 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 7 file(s) not represented in the graph (top: (none) 3, .example 2, .css 1)

## Summary
- 1936 nodes · 2555 edges · 168 communities (36 shown, 132 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 51 edges (avg confidence: 0.9)
- Token cost: 109,670 input · 0 output

## Community Hubs (Navigation)
- Cloudflare Workers Type Defs
- URL Types & Frontend Client
- CF Types: AbortSignal
- CF Types: CloseEvent
- PSA API Client
- App Settings & Entrypoints
- eBay API Client
- SQLite Cert Store
- Condition CV Subgrades
- Training Dataset & Sampler
- Grade HTTP Endpoint
- PSA Harvest Ops & Docs
- CORN Grade Model
- Card Detection & Dewarp
- eBay Harvest Pipeline
- CF Types: Console
- CF Types: CompressionStream
- TypeScript Config
- Multi-Token PSA Collection
- CF Types: URLSearchParams
- Lightning Training Modules
- Centering Subgrade
- CF Types: DurableObjectStorage
- Frontend Package Manifest
- Frontend Layout & Components
- CF Types: Element
- CF Types: Headers
- CF Types: SubtleCrypto
- CF Types: Blob
- CF Types: Body
- CF Types: FormData
- CF Types: URLPattern
- ONNX Export & Calibration
- CF Types: DurableObjectState
- Temperature Calibration
- Augmentations & Image Loading
- Image Quality Filter
- CF Types: WorkerEntrypoint
- Frontend Dev Dependencies
- ONNX Inference Service
- CF Types: R2Object
- Front/Back Side Classifier
- Training Metrics
- CF Types: ByteLengthQueuingStrategy
- CF Types: DigestStream
- CF Types: DurableObjectTransaction
- CF Types: ReadableStream
- CF Types: Socket
- CF Types: WritableStreamDefaultWriter
- Frontend npm Scripts
- PSA Token Settings
- YOLO Defect Detector
- CF Types: Container
- CF Types: DurableObject
- CF Types: R2Bucket
- CF Types: SqlStorageCursor
- CF Types: Vectorize
- ESLint Config
- Serving & Deploy Overview
- Multi-Task Grade Model
- CF Types: DurableObjectNamespace
- CF Types: ReadableStreamBYOBReader
- CF Types: VectorizeIndex
- CF Types: WorkflowInstance
- Next/OpenNext Config
- Frontend Runtime Deps
- CF Types: Ai
- CF Types: D1Database
- CF Types: D1PreparedStatement
- CF Types: KVNamespace
- CF Types: ReadableByteStreamController
- CF Types: ReadableStreamDefaultReader
- CF Types: TextDecoder
- CF Types: AiGateway
- CF Types: Comment
- CF Types: EmailMessage
- CF Types: HTMLRewriter
- CF Types: HTMLRewriterDocumentContentH
- CF Types: ReadableStreamBYOBRequest
- CF Types: ReadableStreamDefaultControl
- CF Types: SyncKvStorage
- CF Types: Table
- CF Types: Text
- CF Types: TextEncoder
- CF Types: TransformStreamDefaultContro
- Frontend Static Icons
- CF Types: AbortController
- CF Types: AutoRAG
- CF Types: Cache
- CF Types: Crypto
- CF Types: D1DatabaseSession
- CF Types: EndTag
- CF Types: HTMLRewriterElementContentHa
- CF Types: ImageTransformationResult
- CF Types: ImageTransformer
- CF Types: MediaTransformationResult
- CF Types: Module
- CF Types: R2MultipartUpload
- CF Types: WebSocketRequestResponsePair
- CF Types: Workflow
- CF Types: WorkflowEntrypoint
- CF Types: BasicImageTransformations
- CF Types: Disposable
- CF Types: DOMException
- CF Types: DurableObjectId
- CF Types: ExecutionContext
- CF Types: Global
- CF Types: HelloWorldBinding
- CF Types: ImagesBinding
- CF Types: Memory
- CF Types: Message
- CF Types: MessageBatch
- CF Types: NodeStyleServer
- CF Types: Performance
- CF Types: PipelineTransformationEntryp
- CF Types: Queue
- CF Types: SqlStorage
- CF Types: ToMarkdownService
- CF Types: WorkflowStep
- CF Types: WritableStreamDefaultControl
- CF Types: AnalyticsEngineDataset
- CF Types: CacheStorage
- CF Types: CloudflareEnv
- CF Types: CompileError
- CF Types: DispatchNamespace
- CF Types: DocumentEnd
- CF Types: EventListenerObject
- CF Types: Hyperdrive
- CF Types: IncomingRequestCfPropertiesB
- CF Types: Instance
- CF Types: JsonWebKey
- CF Types: MediaBinding
- CF Types: MediaTransformationGenerator
- CF Types: MediaTransformer
- CF Types: MessageChannel
- CF Types: Navigator
- CF Types: NonRetryableError
- CF Types: Pipeline
- CF Types: ProcessEnv
- CF Types: R2Checksums
- CF Types: RateLimit
- CF Types: ResponseFunctionToolCall
- CF Types: RpcTarget
- CF Types: RuntimeError
- CF Types: ScheduledController
- CF Types: Scheduler
- CF Types: SecretsStoreSecret
- CF Types: SendEmail
- CF Types: TraceItemFetchEventInfoReque
- CF Types: UnsafeTraceMetrics
- CF Types: WorkerLoader
- CF Types: WorkerStub
- postcss.config.mjs
- CF Types: __DURABLE_OBJECT_BRAND
- CF Types: onRequest
- CF Types: __RPC_STUB_BRAND
- CF Types: __RPC_TARGET_BRAND
- CF Types: __WORKER_ENTRYPOINT_BRAND
- CF Types: __WORKFLOW_ENTRYPOINT_BRAND
- pokescan

## God Nodes (most connected - your core abstractions)
1. `get_settings()` - 28 edges
2. `Event` - 25 edges
3. `detect_and_crop()` - 21 edges
4. `Console` - 21 edges
5. `connect()` - 21 edges
6. `GradeModel` - 18 edges
7. `compilerOptions` - 17 edges
8. `URLSearchParams` - 16 edges
9. `PSAClient` - 16 edges
10. `DurableObjectStorage` - 15 edges

## Surprising Connections (you probably didn't know these)
- `fit_temperature()` --uses--> `GradeModel`  [INFERRED]
  ml/training/calibrate.py → backend/app/ml/model.py
- `GradeLightningModule` --uses--> `GradeModel`  [INFERRED]
  ml/training/lightning_module.py → backend/app/ml/model.py
- `MultiTaskGradeLightningModule` --uses--> `MultiTaskGradeModel`  [INFERRED]
  ml/training/lightning_module.py → backend/app/ml/multitask.py
- `_run_all()` --calls--> `get_settings()`  [EXTRACTED]
  ml/data/collect_all.py → backend/app/core/config.py
- `_run()` --calls--> `get_settings()`  [EXTRACTED]
  ml/data/collect.py → backend/app/core/config.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **ConvNeXt-Tiny Training Config Family** — ml_training_configs_mvp_config, ml_training_configs_mvp_8gb_config, ml_training_configs_smoke_config, ml_training_configs_multitask_config [INFERRED 0.95]
- **Grade Inference Pipeline** — readme_card_detector, readme_convnext_corn_model, readme_centering_cv, readme_condition_cv, readme_grade_endpoint [EXTRACTED 1.00]
- **PSA Multi-Token Harvest Operations** — handoff_collect_all_multi_token, handoff_resume_strategy, handoff_non_overlapping_windows, handoff_psa_daily_quota, handoff_upsert_cert [EXTRACTED 1.00]
- **Next.js create-next-app default public assets** — frontend_public_file_file_icon, frontend_public_globe_globe_icon, frontend_public_window_window_icon, frontend_public_next_next_js_logo [INFERRED 0.85]

## Communities (168 total, 132 thin omitted)

### Community 0 - "Cloudflare Workers Type Defs"
Cohesion: 0.00
Nodes (614): RFC-2253, RFC-5246, Ai_Cf_Ai4Bharat_Indictrans2_En_Indic_1B_Input, Ai_Cf_Ai4Bharat_Indictrans2_En_Indic_1B_Output, Ai_Cf_Aisingapore_Gemma_Sea_Lion_V4_27B_It_Async_Batch, Ai_Cf_Aisingapore_Gemma_Sea_Lion_V4_27B_It_AsyncResponse, Ai_Cf_Aisingapore_Gemma_Sea_Lion_V4_27B_It_Chat_Completion_Response, Ai_Cf_Aisingapore_Gemma_Sea_Lion_V4_27B_It_Input (+606 more)

### Community 1 - "URL Types & Frontend Client"
Cohesion: 0.06
Nodes (29): URL, CenteringSubGrade, FactorSubGrade, gradeCard(), GradeResponse, HealthResponse, SubGrades, Home() (+21 more)

### Community 2 - "CF Types: AbortSignal"
Cohesion: 0.04
Nodes (7): AbortSignal, EventSource, EventTarget, MessagePort, ServiceWorkerGlobalScope, WebSocket, WorkerGlobalScope

### Community 3 - "CF Types: CloseEvent"
Cohesion: 0.04
Nodes (12): CloseEvent, CustomEvent, EmailEvent, ErrorEvent, Event, ExtendableEvent, FetchEvent, MessageEvent (+4 more)

### Community 4 - "PSA API Client"
Cohesion: 0.09
Nodes (8): _process_cert(), parse_grade(), PSAAPIError, PSACertDetails, PSACertImages, PSAClient, PSAQuotaExhaustedError, PSARateLimitError

### Community 5 - "App Settings & Entrypoints"
Cohesion: 0.10
Nodes (7): get_settings(), create_app(), _lifespan(), grade_distribution(), main(), main(), _stratified_assign()

### Community 6 - "eBay API Client"
Cohesion: 0.09
Nodes (6): EBayAPIError, EBayClient, EBayItemSummary, upscale_ebay_image_url(), _resolve_candidate_urls(), _add()

### Community 7 - "SQLite Cert Store"
Cohesion: 0.12
Nodes (11): connect(), factor_label_coverage(), _migrate(), precrop_coverage(), update_factor_subgrades(), update_precrop_paths(), main(), _resolve_image() (+3 more)

### Community 8 - "Condition CV Subgrades"
Cohesion: 0.20
Nodes (17): _analyze_side(), _corner_factor(), _corner_geometry(), _edge_factor(), _edge_roughness(), _edge_whitening(), FactorResult, _fit_line() (+9 more)

### Community 9 - "Training Dataset & Sampler"
Cohesion: 0.13
Nodes (7): _balanced_sampler(), CardRecord, load_records(), PSACardDataModule, PSACardDataset, _load_yaml(), main()

### Community 10 - "Grade HTTP Endpoint"
Cohesion: 0.12
Nodes (10): FactorSubGrade, grade_card(), GradeResponse, health(), _read_image(), SubGrades, compute_condition_subgrades(), ConditionResult (+2 more)

### Community 11 - "PSA Harvest Ops & Docs"
Cohesion: 0.11
Nodes (19): Cert Range Strategy (open question), Harvested Grade Histogram (6,352 certs, 10-heavy), PSA Daily Quota / Retry-After Exhaustion, PokeScan Session Handoff, upsert_cert (idempotent cert storage), multitask.yaml Training Config, mvp_8gb.yaml Training Config, mvp.yaml Training Config (+11 more)

### Community 12 - "CORN Grade Model"
Cohesion: 0.15
Nodes (7): corn_loss(), decode_corn_logits(), grade_to_targets(), GradeOutput, multi_task_loss(), MultiTaskOutput, confusion_matrix()

### Community 13 - "Card Detection & Dewarp"
Cohesion: 0.17
Nodes (8): CardCrop, detect_and_crop(), encode_jpeg(), _find_card_quad(), _order_quad(), _resize_pad(), _warp_to_canvas(), compress_jpeg()

### Community 14 - "eBay Harvest Pipeline"
Cohesion: 0.15
Nodes (8): cert_exists(), _classify_listing(), _harvest_grade(), _ingest_item(), ListingDecision, main(), _run(), save_compressed_jpeg()

### Community 16 - "CF Types: CompressionStream"
Cohesion: 0.10
Nodes (7): CompressionStream, DecompressionStream, FixedLengthStream, IdentityTransformStream, TextDecoderStream, TextEncoderStream, TransformStream

### Community 17 - "TypeScript Config"
Cohesion: 0.10
Nodes (19): compilerOptions, allowJs, esModuleInterop, incremental, isolatedModules, jsx, lib, module (+11 more)

### Community 18 - "Multi-Token PSA Collection"
Cohesion: 0.18
Nodes (9): _effective_token0_count(), main(), _parse_int_list(), _run_all(), main(), _run(), _drain(), pokemon_cert_count() (+1 more)

### Community 21 - "Centering Subgrade"
Cohesion: 0.19
Nodes (6): CenteringSubGrade, CenteringResult, compute_centering_subgrade(), _find_inner_edges(), _grade_from_ratio(), measure_centering()

### Community 23 - "Frontend Package Manifest"
Cohesion: 0.13
Nodes (14): name, private, version, eslint, eslint-config-next, react-dom, react-icons, tailwindcss (+6 more)

### Community 24 - "Frontend Layout & Components"
Cohesion: 0.21
Nodes (10): getHealth(), inter, metadata, RootLayout(), Navbar(), PokeballIcon(), Props, client (+2 more)

### Community 29 - "CF Types: Body"
Cohesion: 0.15
Nodes (3): Body, Request, Response

### Community 32 - "ONNX Export & Calibration"
Cohesion: 0.20
Nodes (3): GradeModel, CalibratedExportModule, main()

### Community 34 - "Temperature Calibration"
Cohesion: 0.21
Nodes (5): CalibrationResult, fit_temperature(), _closure(), _nll(), _per_class_log_probs()

### Community 36 - "Image Quality Filter"
Cohesion: 0.22
Nodes (3): assess_image(), _laplacian_variance(), QualityResult

### Community 38 - "Frontend Dev Dependencies"
Cohesion: 0.18
Nodes (11): devDependencies, eslint, eslint-config-next, @eslint/eslintrc, tailwindcss, @tailwindcss/postcss, @types/node, @types/react (+3 more)

### Community 41 - "Front/Back Side Classifier"
Cohesion: 0.33
Nodes (4): back_score_image(), _load(), pick_front_back(), SideScore

### Community 42 - "Training Metrics"
Cohesion: 0.31
Nodes (3): exact_match(), mean_absolute_error(), off_by_one()

### Community 43 - "CF Types: ByteLengthQueuingStrategy"
Cohesion: 0.22
Nodes (3): ByteLengthQueuingStrategy, CountQueuingStrategy, QueuingStrategy

### Community 49 - "Frontend npm Scripts"
Cohesion: 0.22
Nodes (9): scripts, build, cf-typegen, deploy, dev, lint, preview, start (+1 more)

### Community 57 - "ESLint Config"
Cohesion: 0.25
Nodes (5): compat, __dirname, eslintConfig, __filename, @eslint/eslintrc

### Community 58 - "Serving & Deploy Overview"
Cohesion: 0.25
Nodes (8): Cloudflare Runtime Preview/Deploy, OpenNext Starter (Next.js on Cloudflare), smoke.yaml Training Config, FastAPI Inference Service, POST /api/grade, GET /api/health, Next.js Upload UI, ONNX Model Export (grade_model.onnx)

### Community 64 - "Next/OpenNext Config"
Cohesion: 0.29
Nodes (3): nextConfig, next, @opennextjs/cloudflare

### Community 65 - "Frontend Runtime Deps"
Cohesion: 0.29
Nodes (7): dependencies, next, @opennextjs/cloudflare, react, react-dom, react-icons, @tanstack/react-query

### Community 85 - "Frontend Static Icons"
Cohesion: 0.40
Nodes (5): Favicon (Ovoid Outline Logo), File Document Icon, Globe Icon, Next.js Wordmark Logo, Browser Window Icon

### Community 101 - "CF Types: BasicImageTransformations"
Cohesion: 0.67
Nodes (3): BasicImageTransformations, RequestInitCfPropertiesImage, RequestInitCfPropertiesImageDraw

## Ambiguous Edges - Review These
- `Favicon (Ovoid Outline Logo)` → `Next.js Wordmark Logo`  [AMBIGUOUS]
  frontend/public/favicon.svg · relation: conceptually_related_to

## Knowledge Gaps
- **30 isolated node(s):** `RFC-5246`, `RFC-2253`, `react-dom`, `react-icons`, `@tailwindcss/postcss` (+25 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1440 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **132 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `URL` connect `URL Types & Frontend Client` to `Cloudflare Workers Type Defs`?**
  _High betweenness centrality (0.100) - this node is a cross-community bridge._
- **What connects `RFC-5246`, `RFC-2253`, `react-dom` to the rest of the system?**
  _30 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Cloudflare Workers Type Defs` be split into smaller, more focused modules?**
  _Cohesion score 0.003236245954692557 - nodes in this community are weakly interconnected._
- **Why does `react` connect `URL Types & Frontend Client` to `Frontend Package Manifest`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Should `URL Types & Frontend Client` be split into smaller, more focused modules?**
  _Cohesion score 0.057329462989840346 - nodes in this community are weakly interconnected._
- **Should `CF Types: AbortSignal` be split into smaller, more focused modules?**
  _Cohesion score 0.043478260869565216 - nodes in this community are weakly interconnected._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Favicon (Ovoid Outline Logo)` and `Next.js Wordmark Logo`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._