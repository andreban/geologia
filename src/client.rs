// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use crate::error::{Error as GeminiError, Result as GeminiResult};
use crate::network::event_source::{EventSource, ServerSentEvent};
use crate::prelude::*;

use tokio_stream::{Stream, StreamExt};
use tokio_util::codec::LinesCodecError;
use tracing::error;

/// Default base URL for the Gemini Developer API.
pub const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Async client for the Google Gemini API.
///
/// Provides methods for model discovery, content generation, streaming, token counting,
/// text embeddings, image generation, context caching, file management, batch generation
/// (Batch API), File Search store management for semantic retrieval, and structured decision
/// evaluation (Decisions API). All requests are authenticated with an API key passed at construction time.
///
/// Use [`GeminiClient::builder`] for advanced configuration, such as pointing at a mock
/// server or sharing a configured [`reqwest::Client`].
///
/// # Example
///
/// ```no_run
/// use geologia::prelude::*;
///
/// # async fn run() -> geologia::error::Result<()> {
/// let client = GeminiClient::new("YOUR_API_KEY".into());
/// let request = GenerateContentRequest::builder()
///     .contents(vec![Content::builder().add_text_part("Hi!").build()])
///     .build();
/// let response = client.generate_content(&request, "gemini-2.0-flash").await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct GeminiClient {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl GeminiClient {
    /// Creates a new [`GeminiClient`] with the given API key.
    pub fn new(api_key: String) -> Self {
        Self::builder(api_key).build()
    }

    /// Creates a builder for configuring and constructing a [`GeminiClient`].
    pub fn builder(api_key: impl Into<String>) -> GeminiClientBuilder {
        GeminiClientBuilder::new(api_key)
    }

    /// Returns the configured base URL for this client.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Returns a reference to the underlying [`reqwest::Client`].
    pub fn http_client(&self) -> &reqwest::Client {
        &self.client
    }

    /// Returns the API key used by this client.
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Returns the response unchanged on a success status, otherwise reads the response
    /// body and maps it to a structured error.
    ///
    /// Unlike [`reqwest::Response::error_for_status`], this consumes the body so the
    /// provider's error detail (e.g. the Gemini `INVALID_ARGUMENT` message naming the
    /// offending field) is surfaced to the caller rather than discarded. When the body
    /// parses as a Gemini error envelope it becomes a [`GeminiError::VertexError`];
    /// otherwise the raw body is preserved in a [`GeminiError::ApiError`].
    async fn error_for_status(resp: reqwest::Response) -> GeminiResult<reqwest::Response> {
        let status = resp.status();
        if !status.is_client_error() && !status.is_server_error() {
            return Ok(resp);
        }
        let body = resp.text().await?;
        match serde_json::from_str::<VertexApiErrorResponse>(&body) {
            Ok(envelope) => Err(GeminiError::VertexError(envelope.error)),
            Err(_) => Err(GeminiError::ApiError { status, body }),
        }
    }

    /// Sends a content generation request and returns a stream of response chunks via SSE.
    ///
    /// Each item in the stream is a [`GenerateContentResponseResult`] containing one or more
    /// candidates (or [`PromptFeedback`] when the prompt is blocked). Useful for displaying
    /// incremental output as it is generated.
    ///
    /// SSE events without a `data:` field (such as comments and keep-alives) are skipped.
    /// Mid-stream transport or SSE decoding failures are yielded as
    /// `Err(`[`Error::EventSourceError`](crate::error::Error::EventSourceError)`)` items rather
    /// than silently terminating the stream.
    ///
    /// Note that a stream ending cleanly (`None` with no preceding `Err`) only indicates that the
    /// underlying HTTP body reached EOF between SSE events; it does not by itself guarantee that
    /// the model finished generating. Callers should inspect [`Candidate::finish_reason`] on the
    /// last chunk (or [`GenerateContentResponseResult::prompt_feedback`] if no candidates were
    /// returned) to confirm the response is complete.
    pub async fn stream_generate_content(
        &self,
        request: &GenerateContentRequest,
        model: &str,
    ) -> GeminiResult<impl Stream<Item = GeminiResult<GenerateContentResponseResult>>> {
        let endpoint_url =
            self.endpoint_url(&format!("models/{model}:streamGenerateContent?alt=sse"));
        let client = self.client.clone();
        let request = request.clone();
        let resp = client
            .post(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .json(&request)
            .send()
            .await?;
        Ok(Self::response_event_stream(
            Self::error_for_status(resp).await?,
        ))
    }

    fn response_event_stream(
        resp: reqwest::Response,
    ) -> impl Stream<Item = GeminiResult<GenerateContentResponseResult>> {
        resp.event_stream().filter_map(Self::parse_event)
    }

    fn parse_event(
        event_result: std::result::Result<ServerSentEvent, LinesCodecError>,
    ) -> Option<GeminiResult<GenerateContentResponseResult>> {
        let event = match event_result {
            Ok(event) => event,
            Err(err) => return Some(Err(err.into())),
        };
        let data = event.data?;

        Some(
            serde_json::from_str::<GenerateContentResponse>(&data)
                .map_err(Into::into)
                .and_then(|resp| resp.into_result()),
        )
    }

    /// Sends a content generation request and returns the complete response.
    ///
    /// For streaming responses, use [`stream_generate_content`](Self::stream_generate_content).
    pub async fn generate_content(
        &self,
        request: &GenerateContentRequest,
        model: &str,
    ) -> GeminiResult<GenerateContentResponseResult> {
        let endpoint_url = self.endpoint_url(&format!("models/{model}:generateContent"));
        let resp = self
            .client
            .post(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .json(&request)
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;
        tracing::debug!("generate_content response: {:?}", txt_json);

        match serde_json::from_str::<GenerateContentResponse>(&txt_json) {
            Ok(response) => Ok(response.into_result()?),
            Err(e) => {
                tracing::error!("Failed to parse response: {} with error {}", txt_json, e);
                Err(e.into())
            }
        }
    }

    /// Generates an embedding vector for the given content.
    ///
    /// See <https://ai.google.dev/api/embeddings#method:-models.embedcontent>.
    pub async fn embed_content(
        &self,
        request: &EmbedContentRequest,
        model: &str,
    ) -> GeminiResult<EmbedContentResponse> {
        let model_id = model.strip_prefix("models/").unwrap_or(model);
        let endpoint_url = self.endpoint_url(&format!("models/{model_id}:embedContent"));
        let resp = self
            .client
            .post(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .json(request)
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;
        tracing::debug!("embed_content response: {:?}", txt_json);

        match serde_json::from_str::<EmbedContentResponse>(&txt_json) {
            Ok(response) => Ok(response),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Generates multiple embedding vectors in a single batch request.
    ///
    /// Sets `model` (as `models/{model}`) on each inner [`EmbedContentRequest`] in the batch
    /// as required by the API.
    ///
    /// See <https://ai.google.dev/api/embeddings#method:-models.batchembedcontents>.
    pub async fn batch_embed_contents(
        &self,
        request: &BatchEmbedContentsRequest,
        model: &str,
    ) -> GeminiResult<BatchEmbedContentsResponse> {
        let model_id = model.strip_prefix("models/").unwrap_or(model);
        let endpoint_url = self.endpoint_url(&format!("models/{model_id}:batchEmbedContents"));
        let resp = self
            .client
            .post(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .json(&request.with_model(model_id))
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;
        tracing::debug!("batch_embed_contents response: {:?}", txt_json);

        match serde_json::from_str::<BatchEmbedContentsResponse>(&txt_json) {
            Ok(response) => Ok(response),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Counts the number of tokens in the given content or `generateContent` request.
    ///
    /// When `request` carries a [`GenerateContentRequest`],
    /// its `model` is set from `model`.
    pub async fn count_tokens(
        &self,
        request: &CountTokensRequest,
        model: &str,
    ) -> GeminiResult<CountTokensResponseResult> {
        let endpoint_url = self.endpoint_url(&format!("models/{model}:countTokens"));
        let resp = self
            .client
            .post(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .json(&request.with_model(model))
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;
        tracing::debug!("count_tokens response: {:?}", txt_json);

        match serde_json::from_str::<CountTokensResponse>(&txt_json) {
            Ok(response) => Ok(response.into_result()?),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Generates images from a text prompt using an Imagen model via `models/{model}:predict`.
    ///
    /// # Availability
    ///
    /// Standard Gemini API keys do not list any `imagen-*` models or any model supporting
    /// `:predict` in `GET /v1beta/models`. For Gemini image models (such as
    /// `"gemini-2.5-flash-image"`), use [`generate_content`](Self::generate_content) with
    /// [`Modality::Image`](crate::prelude::Modality::Image) and
    /// [`ImageConfig`](crate::prelude::ImageConfig) instead.
    #[deprecated(
        note = "Standard Gemini API keys do not list Imagen `:predict` models; use `GeminiClient::generate_content` with `Modality::Image` and `ImageConfig` instead."
    )]
    #[allow(deprecated)]
    pub async fn predict_image(
        &self,
        request: &PredictImageRequest,
        model: &str,
    ) -> GeminiResult<PredictImageResponse> {
        let endpoint_url = self.endpoint_url(&format!("models/{model}:predict"));

        let resp = self
            .client
            .post(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .json(&request)
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;

        match serde_json::from_str::<PredictImageResponse>(&txt_json) {
            Ok(response) => Ok(response),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Creates a new explicit context cache resource.
    ///
    /// Context caching allows pre-computing and reusing large prefixes (e.g. documents, system
    /// instructions, or tool declarations) across requests.
    ///
    /// The `model` field on `request` is required and will be automatically prefixed with
    /// `models/` if omitted.
    ///
    /// See <https://ai.google.dev/api/caching#cachedContents.create>.
    pub async fn create_cached_content(
        &self,
        request: &CachedContent,
    ) -> GeminiResult<CachedContent> {
        let endpoint_url = self.endpoint_url("cachedContents");
        let mut body = request.clone();
        if let Some(ref model) = body.model
            && !model.starts_with("models/")
        {
            body.model = Some(format!("models/{model}"));
        }
        let resp = self
            .client
            .post(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .json(&body)
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;
        tracing::debug!("create_cached_content response: {:?}", txt_json);

        match serde_json::from_str::<CachedContent>(&txt_json) {
            Ok(content) => Ok(content),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Retrieves metadata for an existing cached content resource by name or ID.
    ///
    /// `name` can be either the full resource name (e.g. `cachedContents/abc123xyz`)
    /// or just the resource ID (e.g. `abc123xyz`).
    ///
    /// See <https://ai.google.dev/api/caching#cachedContents.get>.
    pub async fn get_cached_content(&self, name: &str) -> GeminiResult<CachedContent> {
        let endpoint_url = self.cached_content_resource_url(name);
        let resp = self
            .client
            .get(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;
        tracing::debug!("get_cached_content response: {:?}", txt_json);

        match serde_json::from_str::<CachedContent>(&txt_json) {
            Ok(content) => Ok(content),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Lists existing cached content resources, with optional pagination.
    ///
    /// Pass `page_size` (up to 50) and `page_token` to paginate through results.
    ///
    /// See <https://ai.google.dev/api/caching#cachedContents.list>.
    pub async fn list_cached_contents(
        &self,
        page_size: Option<u32>,
        page_token: Option<&str>,
    ) -> GeminiResult<ListCachedContentsResponse> {
        let mut endpoint_url = self.endpoint_url("cachedContents");
        let mut query_params = Vec::new();
        if let Some(size) = page_size {
            query_params.push(format!("pageSize={size}"));
        }
        if let Some(token) = page_token {
            query_params.push(format!("pageToken={}", encode_query_value(token)));
        }
        if !query_params.is_empty() {
            endpoint_url.push('?');
            endpoint_url.push_str(&query_params.join("&"));
        }

        let resp = self
            .client
            .get(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;
        tracing::debug!("list_cached_contents response: {:?}", txt_json);

        match serde_json::from_str::<ListCachedContentsResponse>(&txt_json) {
            Ok(response) => Ok(response),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Updates the expiration (`ttl` or `expire_time`) of an existing cached content resource.
    ///
    /// Other fields of a cached content resource are immutable.
    ///
    /// If `update_mask` is `None`, the field mask is automatically derived based on whichever
    /// of `request.ttl` and `request.expire_time` are set.
    ///
    /// See <https://ai.google.dev/api/caching#cachedContents.patch>.
    pub async fn update_cached_content(
        &self,
        name: &str,
        request: &CachedContent,
        update_mask: Option<&str>,
    ) -> GeminiResult<CachedContent> {
        let mut endpoint_url = self.cached_content_resource_url(name);
        let mask = update_mask.map(str::to_string).or_else(|| {
            let mut fields = Vec::new();
            if request.ttl.is_some() {
                fields.push("ttl");
            }
            if request.expire_time.is_some() {
                fields.push("expireTime");
            }
            if fields.is_empty() {
                None
            } else {
                Some(fields.join(","))
            }
        });
        if let Some(mask) = mask {
            endpoint_url.push_str(&format!("?updateMask={mask}"));
        }

        let resp = self
            .client
            .patch(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .json(request)
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;
        tracing::debug!("update_cached_content response: {:?}", txt_json);

        match serde_json::from_str::<CachedContent>(&txt_json) {
            Ok(content) => Ok(content),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Deletes an existing cached content resource by name or ID.
    ///
    /// `name` can be either the full resource name (e.g. `cachedContents/abc123xyz`)
    /// or just the resource ID (e.g. `abc123xyz`).
    ///
    /// See <https://ai.google.dev/api/caching#cachedContents.delete>.
    pub async fn delete_cached_content(&self, name: &str) -> GeminiResult<()> {
        let endpoint_url = self.cached_content_resource_url(name);
        let resp = self
            .client
            .delete(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await?;
        let _ = Self::error_for_status(resp).await?;
        Ok(())
    }

    /// Lists the [`Model`]s available through the Gemini API, with optional pagination.
    ///
    /// If `page_size` is `None`, the API returns 50 models per page by default (maximum 1000).
    /// Pass `next_page_token` from a previous [`ListModelsResponse`] as `page_token` to retrieve
    /// the next page.
    ///
    /// See <https://ai.google.dev/api/models#method:-models.list>.
    pub async fn list_models(
        &self,
        page_size: Option<u32>,
        page_token: Option<&str>,
    ) -> GeminiResult<ListModelsResponse> {
        let mut endpoint_url = self.endpoint_url("models");
        let mut query_params = Vec::new();
        if let Some(size) = page_size {
            query_params.push(format!("pageSize={size}"));
        }
        if let Some(token) = page_token {
            query_params.push(format!("pageToken={}", encode_query_value(token)));
        }
        if !query_params.is_empty() {
            endpoint_url.push('?');
            endpoint_url.push_str(&query_params.join("&"));
        }

        let resp = self
            .client
            .get(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;
        tracing::debug!("list_models response: {:?}", txt_json);

        match serde_json::from_str::<ListModelsResponse>(&txt_json) {
            Ok(response) => Ok(response),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Retrieves information about a specific [`Model`], such as its version, token limits,
    /// supported generation methods, and default generation parameters.
    ///
    /// `name` can be either the full resource name (e.g. `models/gemini-2.5-flash`)
    /// or just the model ID (e.g. `gemini-2.5-flash`).
    ///
    /// See <https://ai.google.dev/api/models#method:-models.get>.
    pub async fn get_model(&self, name: &str) -> GeminiResult<Model> {
        let endpoint_url = self.model_resource_url(name);
        let resp = self
            .client
            .get(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;
        tracing::debug!("get_model response: {:?}", txt_json);

        match serde_json::from_str::<Model>(&txt_json) {
            Ok(model) => Ok(model),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Uploads raw bytes to the Gemini Files API (`POST /upload/v1beta/files`) using the
    /// two-step resumable upload protocol and returns the created [`File`].
    ///
    /// Pass an optional human-readable `display_name` (up to 512 characters). To also specify a
    /// custom resource ID (`files/{id}`), use
    /// [`upload_file_with_metadata`](Self::upload_file_with_metadata).
    ///
    /// # Buffering
    ///
    /// `data` is copied into an in-memory request body and sent in a single `upload, finalize`
    /// HTTP request (without multi-chunk streaming or resume retries). Callers should ensure the
    /// payload fits comfortably in memory.
    ///
    /// # Asynchronous processing
    ///
    /// Uploaded files (especially video, audio, and large documents) are processed asynchronously
    /// after the upload finishes: the returned [`File`] may initially report
    /// [`FileState::Processing`] and later transition to [`FileState::Active`] (or
    /// [`FileState::Failed`], populating [`File::error`]). Callers **must** wait until
    /// [`File::state`] is `Some(`[`FileState::Active`]`)` (by polling [`get_file`](Self::get_file))
    /// before referencing [`File::uri`] in a `generateContent` request.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::time::Duration;
    /// use geologia::prelude::*;
    ///
    /// # async fn run() -> core::result::Result<(), Box<dyn std::error::Error>> {
    /// let client = GeminiClient::new("YOUR_API_KEY".into());
    /// let mut file = client
    ///     .upload_file(b"Hello from Files API!", "text/plain", Some("greeting.txt"))
    ///     .await?;
    /// let file_name = file.name.clone().ok_or("missing file name")?;
    ///
    /// while file.state != Some(FileState::Active) {
    ///     if file.state == Some(FileState::Failed) {
    ///         return Err(format!("file processing failed: {:?}", file.error).into());
    ///     }
    ///     tokio::time::sleep(Duration::from_secs(2)).await;
    ///     file = client.get_file(&file_name).await?;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// See <https://ai.google.dev/api/files#method:-media.upload>.
    pub async fn upload_file(
        &self,
        data: &[u8],
        mime_type: &str,
        display_name: Option<&str>,
    ) -> GeminiResult<File> {
        let metadata = display_name.map(|name| File::builder().display_name(name).build());
        self.upload_file_internal(data, mime_type, metadata).await
    }

    /// Uploads raw bytes to the Gemini Files API (`POST /upload/v1beta/files`) with explicit
    /// [`File`] metadata (such as a custom [`File::name`] and/or [`File::display_name`]).
    ///
    /// Only the writable create-time fields ([`File::name`] and [`File::display_name`]) are
    /// forwarded in the initiation request; output-only fields on `file` are ignored. If
    /// `file.name` is set without the `"files/"` prefix (e.g. `"my-doc-123"`), `"files/"` is
    /// prepended automatically. An empty `file.name` (`""`) is treated as unset.
    ///
    /// `data` is copied into an in-memory request body and sent in a single `upload, finalize`
    /// HTTP request.
    ///
    /// Uploaded files are processed asynchronously (`Processing` -> `Active` or `Failed`).
    /// Callers **must** wait until [`File::state`] is `Some(`[`FileState::Active`]`)` (by polling
    /// [`get_file`](Self::get_file)) before referencing [`File::uri`] in a `generateContent`
    /// request.
    ///
    /// See <https://ai.google.dev/api/files#method:-media.upload>.
    pub async fn upload_file_with_metadata(
        &self,
        data: &[u8],
        mime_type: &str,
        file: &File,
    ) -> GeminiResult<File> {
        let normalized_name = file.name.as_deref().and_then(|name| {
            if name.is_empty() {
                None
            } else if name.starts_with("files/") {
                Some(name.to_string())
            } else {
                Some(format!("files/{name}"))
            }
        });
        let normalized = File {
            name: normalized_name,
            display_name: file.display_name.clone(),
            ..Default::default()
        };
        self.upload_file_internal(data, mime_type, Some(normalized))
            .await
    }

    async fn upload_file_internal(
        &self,
        data: &[u8],
        mime_type: &str,
        file: Option<File>,
    ) -> GeminiResult<File> {
        let start_url = self.upload_endpoint_url("files");
        let num_bytes = data.len();
        let create_request = CreateFileRequest { file };

        // Step 1: Initiate a resumable upload session and read `x-goog-upload-url`.
        let start_resp = self
            .client
            .post(&start_url)
            .header("x-goog-api-key", &self.api_key)
            .header("X-Goog-Upload-Protocol", "resumable")
            .header("X-Goog-Upload-Command", "start")
            .header("X-Goog-Upload-Header-Content-Length", num_bytes.to_string())
            .header("X-Goog-Upload-Header-Content-Type", mime_type)
            .json(&create_request)
            .send()
            .await?;
        let start_resp = Self::error_for_status(start_resp).await?;

        let upload_url = start_resp
            .headers()
            .get("x-goog-upload-url")
            .and_then(|value| value.to_str().ok())
            .map(str::trim)
            .filter(|url| !url.is_empty())
            .map(str::to_owned);

        let upload_url = match upload_url {
            Some(url) => url,
            None => {
                let body = start_resp.text().await.unwrap_or_default();
                let detail = if body.is_empty() {
                    "Resumable upload initiation response missing x-goog-upload-url header"
                        .to_string()
                } else {
                    format!(
                        "Resumable upload initiation response missing x-goog-upload-url header: {body}"
                    )
                };
                return Err(GeminiError::UnexpectedResponse { message: detail });
            }
        };

        // Step 2: Upload the raw file bytes and finalize the pre-authenticated upload session.
        let finalize_resp = self
            .client
            .post(&upload_url)
            .header(reqwest::header::CONTENT_LENGTH, num_bytes)
            .header("X-Goog-Upload-Offset", "0")
            .header("X-Goog-Upload-Command", "upload, finalize")
            .body(data.to_vec())
            .send()
            .await?;
        let finalize_resp = Self::error_for_status(finalize_resp).await?;

        let txt_json = finalize_resp.text().await?;
        tracing::debug!("upload_file response: {:?}", txt_json);

        match serde_json::from_str::<CreateFileResponse>(&txt_json) {
            Ok(response) => Ok(response.file),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Retrieves metadata for an uploaded or generated [`File`] by name or ID.
    ///
    /// `name` can be either the full resource name (e.g. `files/abc-123`)
    /// or just the file ID (e.g. `abc-123`).
    ///
    /// Newly uploaded files are processed asynchronously (`Processing` -> `Active` or `Failed`).
    /// Poll `get_file` until [`File::state`] is `Some(`[`FileState::Active`]`)` before referencing
    /// [`File::uri`] in a `generateContent` request.
    ///
    /// See <https://ai.google.dev/api/files#method:-files.get>.
    pub async fn get_file(&self, name: &str) -> GeminiResult<File> {
        let endpoint_url = self.file_resource_url(name);
        let resp = self
            .client
            .get(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;
        tracing::debug!("get_file response: {:?}", txt_json);

        match serde_json::from_str::<File>(&txt_json) {
            Ok(file) => Ok(file),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Lists [`File`]s owned by the requesting project, with optional pagination.
    ///
    /// If `page_size` is `None`, the API returns 10 files per page by default (maximum 100).
    /// Pass `next_page_token` from a previous [`ListFilesResponse`] as `page_token` to retrieve
    /// the next page.
    ///
    /// See <https://ai.google.dev/api/files#method:-files.list>.
    pub async fn list_files(
        &self,
        page_size: Option<u32>,
        page_token: Option<&str>,
    ) -> GeminiResult<ListFilesResponse> {
        let mut endpoint_url = self.endpoint_url("files");
        let mut query_params = Vec::new();
        if let Some(size) = page_size {
            query_params.push(format!("pageSize={size}"));
        }
        if let Some(token) = page_token {
            query_params.push(format!("pageToken={}", encode_query_value(token)));
        }
        if !query_params.is_empty() {
            endpoint_url.push('?');
            endpoint_url.push_str(&query_params.join("&"));
        }

        let resp = self
            .client
            .get(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;
        tracing::debug!("list_files response: {:?}", txt_json);

        match serde_json::from_str::<ListFilesResponse>(&txt_json) {
            Ok(response) => Ok(response),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Deletes an existing [`File`] resource by name or ID.
    ///
    /// `name` can be either the full resource name (e.g. `files/abc-123`)
    /// or just the file ID (e.g. `abc-123`).
    ///
    /// See <https://ai.google.dev/api/files#method:-files.delete>.
    pub async fn delete_file(&self, name: &str) -> GeminiResult<()> {
        let endpoint_url = self.file_resource_url(name);
        let resp = self
            .client
            .delete(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await?;
        let _ = Self::error_for_status(resp).await?;
        Ok(())
    }

    /// Enqueues a batch of `generateContent` requests for asynchronous processing via
    /// `POST v1beta/models/{model}:batchGenerateContent`.
    ///
    /// The batch can supply input requests inline via [`InputConfig::requests`] or reference a
    /// `.jsonl` file previously uploaded through the Files API via [`InputConfig::file_name`].
    ///
    /// Batch generation runs asynchronously at reduced cost. The returned [`BatchOperation`]
    /// provides the server-assigned operation name (`batches/{id}`) and initial state
    /// ([`BatchState::Pending`]). Callers should poll [`get_batch`](Self::get_batch) until
    /// [`BatchOperation::is_done`] is `true` before reading the output.
    ///
    /// `model` can be either the full resource name (e.g. `models/gemini-2.5-flash`)
    /// or just the model ID (e.g. `gemini-2.5-flash`).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use geologia::prelude::*;
    ///
    /// # async fn run() -> geologia::error::Result<()> {
    /// let client = GeminiClient::new("YOUR_API_KEY".into());
    /// let request = BatchGenerateContentRequest::new(
    ///     GenerateContentBatch::builder()
    ///         .display_name("summaries-batch")
    ///         .input_config(InputConfig::from_inline(vec![
    ///             GenerateContentRequest::builder()
    ///                 .contents(vec![Content::builder().add_text_part("Say hi!").build()])
    ///                 .build(),
    ///         ]))
    ///         .build(),
    /// );
    ///
    /// let operation = client.batch_generate_content(&request, "gemini-2.5-flash").await?;
    /// println!("Created batch: {:?}", operation.name);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// See <https://ai.google.dev/gemini-api/docs/batch-mode>.
    pub async fn batch_generate_content(
        &self,
        request: &BatchGenerateContentRequest,
        model: &str,
    ) -> GeminiResult<BatchOperation> {
        let model_id = model.strip_prefix("models/").unwrap_or(model);
        let endpoint_url = self.endpoint_url(&format!("models/{model_id}:batchGenerateContent"));
        let resp = self
            .client
            .post(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .json(request)
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;
        tracing::debug!("batch_generate_content response: {:?}", txt_json);

        match serde_json::from_str::<BatchOperation>(&txt_json) {
            Ok(operation) => Ok(operation),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Retrieves the latest status, metadata, and results of a batch job by name or ID.
    ///
    /// `name` can be either the full resource name (e.g. `batches/abc-123`)
    /// or just the batch ID (e.g. `abc-123`).
    ///
    /// Use this method to poll the progress of an asynchronous batch job until
    /// [`BatchOperation::is_done`] returns `true` or [`BatchOperation::state`] reaches
    /// [`BatchState::Succeeded`], [`BatchState::Failed`], or [`BatchState::Cancelled`].
    ///
    /// See <https://ai.google.dev/gemini-api/docs/batch-mode>.
    pub async fn get_batch(&self, name: &str) -> GeminiResult<BatchOperation> {
        let endpoint_url = self.batch_resource_url(name);
        let resp = self
            .client
            .get(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;
        tracing::debug!("get_batch response: {:?}", txt_json);

        match serde_json::from_str::<BatchOperation>(&txt_json) {
            Ok(operation) => Ok(operation),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Lists batch operations owned by the requesting project, with optional pagination.
    ///
    /// If `page_size` is `None`, the API returns 10 batches per page by default (maximum 100).
    /// Pass `next_page_token` from a previous [`ListBatchesResponse`] as `page_token` to
    /// retrieve the next page.
    ///
    /// See <https://ai.google.dev/api/batch#methods>.
    pub async fn list_batches(
        &self,
        page_size: Option<u32>,
        page_token: Option<&str>,
    ) -> GeminiResult<ListBatchesResponse> {
        let mut endpoint_url = self.endpoint_url("batches");
        let mut query_params = Vec::new();
        if let Some(size) = page_size {
            query_params.push(format!("pageSize={size}"));
        }
        if let Some(token) = page_token {
            query_params.push(format!("pageToken={}", encode_query_value(token)));
        }
        if !query_params.is_empty() {
            endpoint_url.push('?');
            endpoint_url.push_str(&query_params.join("&"));
        }

        let resp = self
            .client
            .get(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;
        tracing::debug!("list_batches response: {:?}", txt_json);

        match serde_json::from_str::<ListBatchesResponse>(&txt_json) {
            Ok(response) => Ok(response),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Cancels an in-progress batch generation job by name or ID via
    /// `POST v1beta/batches/{name}:cancel`.
    ///
    /// `name` can be either the full resource name (e.g. `batches/abc-123`)
    /// or just the batch ID (e.g. `abc-123`).
    ///
    /// When cancelled, the batch stops processing new requests and its state transitions
    /// to [`BatchState::Cancelled`].
    ///
    /// See <https://ai.google.dev/api/batch#methods>.
    pub async fn cancel_batch(&self, name: &str) -> GeminiResult<()> {
        let base_url = self.batch_resource_url(name);
        let endpoint_url = format!("{base_url}:cancel");
        let resp = self
            .client
            .post(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .json(&serde_json::json!({}))
            .send()
            .await?;
        let _ = Self::error_for_status(resp).await?;
        Ok(())
    }

    /// Deletes an existing batch resource by name or ID via
    /// `DELETE v1beta/batches/{name}`.
    ///
    /// `name` can be either the full resource name (e.g. `batches/abc-123`)
    /// or just the batch ID (e.g. `abc-123`).
    ///
    /// Deleting a batch indicates the client is no longer interested in the results and
    /// cleans up metadata. It does not cancel an in-progress batch; call [`cancel_batch`](Self::cancel_batch)
    /// first if the batch is still running.
    ///
    /// See <https://ai.google.dev/api/batch#methods>.
    pub async fn delete_batch(&self, name: &str) -> GeminiResult<()> {
        let endpoint_url = self.batch_resource_url(name);
        let resp = self
            .client
            .delete(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await?;
        let _ = Self::error_for_status(resp).await?;
        Ok(())
    }

    /// Creates a new [`FileSearchStore`] for indexing documents.
    ///
    /// Only `display_name` and `embedding_model` are sent; the server assigns the resource name.
    ///
    /// If `store.embedding_model` is specified without a `models/` prefix, `models/` is prepended
    /// automatically.
    ///
    /// See <https://ai.google.dev/api/file-search/file-search-stores#method:-filesearchstores.create>.
    pub async fn create_file_search_store(
        &self,
        store: &FileSearchStore,
    ) -> GeminiResult<FileSearchStore> {
        let endpoint_url = self.endpoint_url("fileSearchStores");
        let mut body = FileSearchStore {
            display_name: store.display_name.clone(),
            embedding_model: store.embedding_model.clone(),
            ..Default::default()
        };
        if let Some(ref model) = body.embedding_model
            && !model.starts_with("models/")
        {
            body.embedding_model = Some(format!("models/{model}"));
        }
        let resp = self
            .client
            .post(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .json(&body)
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;
        tracing::debug!("create_file_search_store response: {:?}", txt_json);

        match serde_json::from_str::<FileSearchStore>(&txt_json) {
            Ok(created) => Ok(created),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Retrieves metadata for an existing [`FileSearchStore`] by name or ID.
    ///
    /// `name` can be either the full resource name (e.g. `fileSearchStores/my-store-123`)
    /// or just the store ID (e.g. `my-store-123`).
    ///
    /// See <https://ai.google.dev/api/file-search/file-search-stores#method:-filesearchstores.get>.
    pub async fn get_file_search_store(&self, name: &str) -> GeminiResult<FileSearchStore> {
        let endpoint_url = self.file_search_store_resource_url(name);
        let resp = self
            .client
            .get(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;
        tracing::debug!("get_file_search_store response: {:?}", txt_json);

        match serde_json::from_str::<FileSearchStore>(&txt_json) {
            Ok(store) => Ok(store),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Lists [`FileSearchStore`]s owned by the requesting project, with optional pagination.
    ///
    /// Pass `page_size` and `page_token` to paginate through results.
    ///
    /// See <https://ai.google.dev/api/file-search/file-search-stores#method:-filesearchstores.list>.
    pub async fn list_file_search_stores(
        &self,
        page_size: Option<u32>,
        page_token: Option<&str>,
    ) -> GeminiResult<ListFileSearchStoresResponse> {
        let mut endpoint_url = self.endpoint_url("fileSearchStores");
        let mut query_params = Vec::new();
        if let Some(size) = page_size {
            query_params.push(format!("pageSize={size}"));
        }
        if let Some(token) = page_token {
            query_params.push(format!("pageToken={}", encode_query_value(token)));
        }
        if !query_params.is_empty() {
            endpoint_url.push('?');
            endpoint_url.push_str(&query_params.join("&"));
        }

        let resp = self
            .client
            .get(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;
        tracing::debug!("list_file_search_stores response: {:?}", txt_json);

        match serde_json::from_str::<ListFileSearchStoresResponse>(&txt_json) {
            Ok(response) => Ok(response),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Deletes an existing [`FileSearchStore`] resource by name or ID.
    ///
    /// `name` can be either the full resource name (e.g. `fileSearchStores/my-store-123`)
    /// or just the store ID (e.g. `my-store-123`).
    ///
    /// When `force` is `true`, any documents contained in the store are deleted as well.
    /// When `force` is `false`, attempting to delete a non-empty store fails with an error.
    ///
    /// See <https://ai.google.dev/api/file-search/file-search-stores#method:-filesearchstores.delete>.
    pub async fn delete_file_search_store(&self, name: &str, force: bool) -> GeminiResult<()> {
        let mut endpoint_url = self.file_search_store_resource_url(name);
        if force {
            endpoint_url.push_str("?force=true");
        }
        let resp = self
            .client
            .delete(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await?;
        let _ = Self::error_for_status(resp).await?;
        Ok(())
    }

    /// Directly uploads raw bytes to a [`FileSearchStore`] via
    /// `POST /upload/v1beta/{store}:uploadToFileSearchStore`.
    ///
    /// This initiates a two-step resumable upload session and returns a [`FileSearchOperation`].
    /// Processing (parsing, chunking, and vector embedding) happens asynchronously. Callers should
    /// poll [`get_file_search_operation`](Self::get_file_search_operation) until
    /// [`FileSearchOperation::is_done`] is `true` before querying the store with the [`FileSearch`] tool.
    ///
    /// `store` can be either the full resource name (e.g. `fileSearchStores/my-store-123`)
    /// or just the store ID (e.g. `my-store-123`).
    ///
    /// See <https://ai.google.dev/api/file-search/file-search-stores#method:-media.uploadtofilesearchstore>.
    pub async fn upload_to_file_search_store(
        &self,
        store: &str,
        data: &[u8],
        mime_type: Option<&str>,
        display_name: Option<&str>,
    ) -> GeminiResult<FileSearchOperation> {
        let request = UploadToFileSearchStoreRequest {
            display_name: display_name.map(str::to_string),
            mime_type: mime_type.map(str::to_string),
            custom_metadata: None,
            chunking_config: None,
        };
        self.upload_to_file_search_store_with_request(store, data, &request)
            .await
    }

    /// Directly uploads raw bytes to a [`FileSearchStore`] with a custom [`UploadToFileSearchStoreRequest`]
    /// (such as custom chunking parameters or custom metadata).
    ///
    /// See <https://ai.google.dev/api/file-search/file-search-stores#method:-media.uploadtofilesearchstore>.
    pub async fn upload_to_file_search_store_with_request(
        &self,
        store: &str,
        data: &[u8],
        request: &UploadToFileSearchStoreRequest,
    ) -> GeminiResult<FileSearchOperation> {
        let normalized_store = if store.starts_with("fileSearchStores/") {
            store.to_string()
        } else {
            format!("fileSearchStores/{store}")
        };
        let start_url =
            self.upload_endpoint_url(&format!("{normalized_store}:uploadToFileSearchStore"));
        let num_bytes = data.len();
        let mime = request
            .mime_type
            .as_deref()
            .unwrap_or("application/octet-stream");

        // Step 1: Initiate a resumable upload session and read `x-goog-upload-url`.
        let start_resp = self
            .client
            .post(&start_url)
            .header("x-goog-api-key", &self.api_key)
            .header("X-Goog-Upload-Protocol", "resumable")
            .header("X-Goog-Upload-Command", "start")
            .header("X-Goog-Upload-Header-Content-Length", num_bytes.to_string())
            .header("X-Goog-Upload-Header-Content-Type", mime)
            .json(request)
            .send()
            .await?;
        let start_resp = Self::error_for_status(start_resp).await?;

        let upload_url = start_resp
            .headers()
            .get("x-goog-upload-url")
            .and_then(|value| value.to_str().ok())
            .map(str::trim)
            .filter(|url| !url.is_empty())
            .map(str::to_owned);

        let upload_url = match upload_url {
            Some(url) => url,
            None => {
                let body = start_resp.text().await.unwrap_or_default();
                let detail = if body.is_empty() {
                    "Resumable upload initiation response missing x-goog-upload-url header"
                        .to_string()
                } else {
                    format!(
                        "Resumable upload initiation response missing x-goog-upload-url header: {body}"
                    )
                };
                return Err(GeminiError::UnexpectedResponse { message: detail });
            }
        };

        // Step 2: Upload the raw file bytes and finalize the pre-authenticated upload session.
        let finalize_resp = self
            .client
            .post(&upload_url)
            .header(reqwest::header::CONTENT_LENGTH, num_bytes)
            .header("X-Goog-Upload-Offset", "0")
            .header("X-Goog-Upload-Command", "upload, finalize")
            .body(data.to_vec())
            .send()
            .await?;
        let finalize_resp = Self::error_for_status(finalize_resp).await?;

        let txt_json = finalize_resp.text().await?;
        tracing::debug!("upload_to_file_search_store response: {:?}", txt_json);

        match serde_json::from_str::<FileSearchOperation>(&txt_json) {
            Ok(op) => Ok(op),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Imports a file previously uploaded via the Files API into a [`FileSearchStore`] via
    /// `POST v1beta/{store}:importFile`.
    ///
    /// Returns a [`FileSearchOperation`]. Callers should poll
    /// [`get_file_search_operation`](Self::get_file_search_operation) until
    /// [`FileSearchOperation::is_done`] is `true`.
    ///
    /// `store` can be either the full resource name (e.g. `fileSearchStores/my-store-123`)
    /// or just the store ID (e.g. `my-store-123`).
    ///
    /// See <https://ai.google.dev/api/file-search/file-search-stores#method:-filesearchstores.importfile>.
    pub async fn import_file_to_file_search_store(
        &self,
        store: &str,
        request: &ImportFileRequest,
    ) -> GeminiResult<FileSearchOperation> {
        let normalized_store = if store.starts_with("fileSearchStores/") {
            store.to_string()
        } else {
            format!("fileSearchStores/{store}")
        };
        let endpoint_url = self.endpoint_url(&format!("{normalized_store}:importFile"));
        let resp = self
            .client
            .post(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .json(request)
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;
        tracing::debug!("import_file_to_file_search_store response: {:?}", txt_json);

        match serde_json::from_str::<FileSearchOperation>(&txt_json) {
            Ok(op) => Ok(op),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Retrieves the status of a long-running [`FileSearchOperation`] by name.
    ///
    /// `name` can be either the full resource name (e.g.
    /// `fileSearchStores/{store_id}/upload/operations/{op_id}` for uploads,
    /// `fileSearchStores/{store_id}/operations/{op_id}` for imports, or
    /// `operations/{op_id}`).
    ///
    /// See <https://ai.google.dev/api/file-search/file-search-stores#Operation>.
    pub async fn get_file_search_operation(&self, name: &str) -> GeminiResult<FileSearchOperation> {
        let endpoint_url = self.endpoint_url(name);
        let resp = self
            .client
            .get(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;
        tracing::debug!("get_file_search_operation response: {:?}", txt_json);

        match serde_json::from_str::<FileSearchOperation>(&txt_json) {
            Ok(op) => Ok(op),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Retrieves metadata for a [`Document`] in a [`FileSearchStore`].
    ///
    /// `name` is the full resource name of the document, formatted as
    /// `fileSearchStores/{store_id}/documents/{doc_id}`.
    ///
    /// Poll `get_document` (or the parent [`FileSearchOperation`]) until [`Document::state`] is
    /// `Some(`[`DocumentState::Active`]`)` before querying the store.
    ///
    /// See <https://ai.google.dev/api/file-search/file-search-stores#method:-filesearchstores.documents.get>.
    pub async fn get_document(&self, name: &str) -> GeminiResult<Document> {
        let endpoint_url = self.endpoint_url(name);
        let resp = self
            .client
            .get(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;
        tracing::debug!("get_document response: {:?}", txt_json);

        match serde_json::from_str::<Document>(&txt_json) {
            Ok(doc) => Ok(doc),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Evaluates one or more structured decision questions (`noul`, `choice`, or `score`)
    /// against a shared text or multimodal input context via `POST v1beta/decisions`.
    ///
    /// Returns a [`CreateDecisionResponse`] containing calibrated probabilities and
    /// closed-world confidence scores keyed by question ID.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use geologia::prelude::*;
    ///
    /// # async fn run() -> geologia::error::Result<()> {
    /// let client = GeminiClient::new("YOUR_API_KEY".into());
    /// let request = CreateDecisionRequest::builder()
    ///     .model("gemini-decision-1")
    ///     .text_input("Order #42 was delivered 3 days late and the outer box was crushed.")
    ///     .add_noul_question(
    ///         "needs_refund_review",
    ///         NoulQuestion::new("Should this order be routed to the refund specialist team?"),
    ///     )
    ///     .add_choice_question(
    ///         "priority",
    ///         ChoiceQuestion::new(
    ///             "What support priority should be assigned?",
    ///             [
    ///                 ("low", "General inquiry with no delivery or damage issue"),
    ///                 ("medium", "Minor delay without package damage"),
    ///                 ("high", "Late delivery combined with physical package damage"),
    ///             ],
    ///         ),
    ///     )
    ///     .build()?;
    ///
    /// let response = client.create_decision(&request).await?;
    /// if let Some(noul) = response.get_noul("needs_refund_review") {
    ///     println!("P(true) = {}, confidence = {}", noul.noul, noul.confidence);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// See <https://aistudio.google.com/docs/decisions-api>.
    pub async fn create_decision(
        &self,
        request: &CreateDecisionRequest,
    ) -> GeminiResult<CreateDecisionResponse> {
        request.validate()?;
        let endpoint_url = self.endpoint_url("decisions");
        let resp = self
            .client
            .post(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .json(request)
            .send()
            .await?;
        let resp = Self::error_for_status(resp).await?;

        let txt_json = resp.text().await?;
        tracing::debug!("create_decision response: {:?}", txt_json);

        match serde_json::from_str::<CreateDecisionResponse>(&txt_json) {
            Ok(response) => Ok(response),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    fn batch_resource_url(&self, name: &str) -> String {
        let path = if name.starts_with("batches/") {
            name.to_string()
        } else {
            format!("batches/{name}")
        };
        self.endpoint_url(&path)
    }

    fn file_resource_url(&self, name: &str) -> String {
        let path = if name.starts_with("files/") {
            name.to_string()
        } else {
            format!("files/{name}")
        };
        self.endpoint_url(&path)
    }

    fn model_resource_url(&self, name: &str) -> String {
        let path = if name.starts_with("models/") {
            name.to_string()
        } else {
            format!("models/{name}")
        };
        self.endpoint_url(&path)
    }

    fn cached_content_resource_url(&self, name: &str) -> String {
        let path = if name.starts_with("cachedContents/") {
            name.to_string()
        } else {
            format!("cachedContents/{name}")
        };
        self.endpoint_url(&path)
    }

    fn file_search_store_resource_url(&self, name: &str) -> String {
        let path = if name.starts_with("fileSearchStores/") {
            name.to_string()
        } else {
            format!("fileSearchStores/{name}")
        };
        self.endpoint_url(&path)
    }

    fn endpoint_url(&self, path: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        let path = path.trim_start_matches('/');
        format!("{base}/{path}")
    }

    /// Derives a media upload endpoint URL from the configured `base_url` by inserting `/upload`
    /// before the API version segment when present (e.g.
    /// `https://generativelanguage.googleapis.com/v1beta` ->
    /// `https://generativelanguage.googleapis.com/upload/v1beta/{path}`), or appending `/upload`
    /// to an unversioned base URL (e.g. `http://127.0.0.1:8080` ->
    /// `http://127.0.0.1:8080/upload/{path}`).
    fn upload_endpoint_url(&self, path: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        let path = path.trim_start_matches('/');

        let (origin, base_path) = match base.find("://") {
            Some(scheme_sep) => {
                let after_scheme = scheme_sep + 3;
                match base[after_scheme..].find('/') {
                    Some(rel_slash) => {
                        let slash_idx = after_scheme + rel_slash;
                        (&base[..slash_idx], &base[slash_idx..])
                    }
                    None => (base, ""),
                }
            }
            None => ("", base),
        };

        if let Some((prefix, last_segment)) = base_path.rsplit_once('/')
            && is_api_version_segment(last_segment)
        {
            format!("{origin}{prefix}/upload/{last_segment}/{path}")
        } else {
            format!("{origin}{base_path}/upload/{path}")
        }
    }
}

fn is_api_version_segment(segment: &str) -> bool {
    let mut chars = segment.chars();
    matches!(chars.next(), Some('v'))
        && matches!(chars.next(), Some(c) if c.is_ascii_digit())
        && chars.all(|c| c.is_ascii_alphanumeric())
}

fn encode_query_value(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            _ => {
                use std::fmt::Write as _;
                let _ = write!(encoded, "%{byte:02X}");
            }
        }
    }
    encoded
}

/// Builder for constructing a [`GeminiClient`] with custom configuration.
///
/// Created via [`GeminiClient::builder`].
///
/// # Example
///
/// ```no_run
/// use geologia::prelude::*;
///
/// let client = GeminiClient::builder("YOUR_API_KEY")
///     .base_url("https://my-proxy.example.com/v1beta")
///     .build();
/// ```
#[derive(Clone, Debug)]
pub struct GeminiClientBuilder {
    api_key: String,
    base_url: Option<String>,
    client: Option<reqwest::Client>,
}

impl GeminiClientBuilder {
    /// Creates a new builder for the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: None,
            client: None,
        }
    }

    /// Sets the base URL for API requests.
    ///
    /// Defaults to [`DEFAULT_BASE_URL`].
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Sets a custom [`reqwest::Client`] for making HTTP requests.
    ///
    /// When omitted, a default client is created via [`reqwest::Client::new`].
    pub fn http_client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Consumes the builder and returns the configured [`GeminiClient`].
    pub fn build(self) -> GeminiClient {
        GeminiClient {
            client: self.client.unwrap_or_default(),
            api_key: self.api_key,
            base_url: self
                .base_url
                .unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
        }
    }
}

#[cfg(test)]
#[path = "client_tests.rs"]
mod tests;
