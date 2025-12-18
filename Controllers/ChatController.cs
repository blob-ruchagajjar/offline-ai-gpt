using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.AI;
using offline_ai_gpt.ApiModels;
using offline_ai_gpt.Models;
using offline_ai_gpt.Services;
using System.Text.RegularExpressions;

namespace offline_ai_gpt.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class ChatController : ControllerBase
    {
        private readonly ChatService _chatService;

        public ChatController(ChatService chatService)
        {
            _chatService = chatService;
        }

        [HttpPost("stream")]
        public async Task Stream([FromBody] ChatRequest req, CancellationToken ct)
        {
            if (req == null || string.IsNullOrWhiteSpace(req.Message))
            {
                Response.StatusCode = 400;
                await Response.WriteAsync("Message cannot be empty", ct);
                return;
            }

            // Set headers for streaming response
            Response.StatusCode = 200;
            Response.ContentType = "text/plain; charset=utf-8";
            Response.Headers.CacheControl = "no-cache, no-store, must-revalidate";
            Response.Headers.Connection = "keep-alive";
            Response.Headers.Append("X-Accel-Buffering", "no");

            try
            {
                await foreach (var chunk in _chatService.StreamAsync(req.Message, ct))
                {
                    var cleaned = System.Text.RegularExpressions.Regex.Replace(chunk, @"<\|[^|]+\|>", "");

                    if (!string.IsNullOrWhiteSpace(cleaned))
                    {
                        await Response.WriteAsync(cleaned, ct);
                        await Response.Body.FlushAsync(ct);
                    }
                }
            }
            catch (InvalidOperationException ex)
            {
                // Model not loaded
                Response.StatusCode = 409;
                await Response.WriteAsync(ex.Message, ct);
            }
            catch (OperationCanceledException)
            {
                // client disconnected or cancelled - nothing to do
            }
            catch (Exception ex)
            {
                await Response.WriteAsync($"[ERROR] {ex.Message}", ct);
            }
        }

        [HttpPost("load")]
        public async Task<IActionResult> Load([FromBody] LoadModelRequest req)
        {
            if (req == null || string.IsNullOrWhiteSpace(req.RelativePath))
                return BadRequest("relativePath is required");

            var modelsFolder = Environment.GetEnvironmentVariable("MODELS_FOLDER") ?? "Models";
            var fullPath = Path.IsPathRooted(req.RelativePath) ? req.RelativePath : Path.Combine(modelsFolder, req.RelativePath);

            if (!System.IO.File.Exists(fullPath))
                return NotFound($"Model file not found: {req.RelativePath}");

            try
            {
                await _chatService.LoadModel(fullPath);
                return Ok(new { LoadedModel = fullPath });
            }
            catch (Exception ex)
            {
                // Return a 500 with the error message (useful for debugging)
                return StatusCode(500, new { Error = ex.Message });
            }
        }

    }
}
