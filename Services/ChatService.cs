using LLama;
using LLama.Common;
using LLama.Sampling;
using System.Runtime.CompilerServices;

namespace offline_ai_gpt.Services
{
    public class ChatService : IDisposable
    {
        private readonly object _loadLock = new();
        private LLamaWeights? _model;
        private LLamaContext? _context;
        private readonly SemaphoreSlim _useSemaphore = new SemaphoreSlim(1, 1); // single concurrent generation

        private static string ModelsFolder => Environment.GetEnvironmentVariable("MODELS_FOLDER") ?? "Models";

        public string? LoadedModelPath { get; private set; }
        public ChatService()
        {

        }

        public async Task LoadModel(string fullPath)
        {
            if (string.IsNullOrWhiteSpace(fullPath)) throw new ArgumentException("Model path required", nameof(fullPath));
            if (!File.Exists(fullPath)) throw new FileNotFoundException("Model file not found", fullPath);

            // Load on a background thread because loading can be slow/IO heavy.
            await Task.Run(() =>
            {
                lock (_loadLock)
                {
                    // Dispose existing model/context first (if any)
                    _context?.Dispose();
                    _model?.Dispose();

                    var parameters = new ModelParams(fullPath)
                    {
                        ContextSize = 4096u,
                        GpuLayerCount = 0,
                        BatchSize = 512,
                        Threads = (int)Environment.ProcessorCount,
                        UseMemorymap = true
                    };

                    _model = LLamaWeights.LoadFromFile(parameters);
                    _context = _model.CreateContext(parameters);
                    LoadedModelPath = fullPath;
                }
            }).ConfigureAwait(false);
        }

        public IEnumerable<string> ListAvailableModels()
        {
            var folder = ModelsFolder;
            if (!Directory.Exists(folder)) return Enumerable.Empty<string>();
            return Directory.GetFiles(folder, "*.gguf", SearchOption.TopDirectoryOnly).Select(Path.GetFileName);
        }

        /// <summary>
        /// Streams generated text chunks as they become available.
        /// Throws InvalidOperationException if no model is loaded.
        /// </summary>
        public async IAsyncEnumerable<string> StreamAsync(string userMessage, [EnumeratorCancellation] CancellationToken ct = default)
        {
            if (_model == null || _context == null)
            {
                throw new InvalidOperationException("No model loaded. Call LoadModel(...) or set MODEL_PATH before start.");
            }

            // Ensure only one generation uses the model/context at a time
            await _useSemaphore.WaitAsync(ct).ConfigureAwait(false);
            try
            {
                var executor = new StatelessExecutor(_model, _context.Params);

                var prompt = $"You are a helpful AI assistant.\nUser: {userMessage}\nAssistant:";

                var inference = new InferenceParams
                {
                    MaxTokens = 512,
                    SamplingPipeline = new DefaultSamplingPipeline
                    {
                        Temperature = 0.7f, //lower = more focused, less random
                        TopP = 0.9f,//Avoids low-probability words
                        TopK = 40, //Limits to the K most likely words
                        RepeatPenalty = 1.1f //Penalizes words that have already been generated
                    },
                    AntiPrompts = new List<string>()
                };

                await foreach (var text in executor.InferAsync(prompt, inference).WithCancellation(ct))
                {
                    if (ct.IsCancellationRequested)
                    {
                        yield break;
                    }

                    var cleaned = System.Text.RegularExpressions.Regex.Replace(text, @"<\|[^|]+\|>", "");

                    if (!string.IsNullOrEmpty(text))
                    {
                        yield return text;
                    }
                }
            }
            finally
            {
                _useSemaphore.Release();
            }
        }

        public void Dispose()
        {
            lock (_loadLock)
            {
                _context?.Dispose();
                _model?.Dispose();
                _context = null;
                _model = null;
            }
            _useSemaphore?.Dispose();
        }
    }
}
