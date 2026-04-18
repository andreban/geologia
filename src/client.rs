// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use crate::error::{Error as GeminiError, Result as GeminiResult};
use crate::network::event_source::{EventSource, ServerSentEvent};
use crate::prelude::*;

use tokio_stream::{Stream, StreamExt};
use tokio_util::codec::LinesCodecError;
use tracing::error;

/// Async client for the Google Gemini API.
///
/// Provides methods for content generation, streaming, token counting, text embeddings,
/// and image prediction. All requests are authenticated with an API key passed at
/// construction time.
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
}

impl GeminiClient {
    /// Creates a new [`GeminiClient`] with the given API key.
    pub fn new(api_key: String) -> Self {
        GeminiClient {
            client: reqwest::Client::new(),
            api_key,
        }
    }

    /// Sends a content generation request and returns a stream of response chunks via SSE.
    ///
    /// Each item in the stream is a [`GenerateContentResponseResult`] containing one or more
    /// candidates. Useful for displaying incremental output as it is generated.
    pub async fn stream_generate_content(
        &self,
        request: &GenerateContentRequest,
        model: &str,
    ) -> GeminiResult<impl Stream<Item = GeminiResult<GenerateContentResponseResult>>> {
        let endpoint_url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"
        );
        let client = self.client.clone();
        let request = request.clone();
        Ok(client
            .post(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .json(&request)
            .send()
            .await?
            .event_stream()
            .filter_map(Self::parse_event))
    }

    fn parse_event(
        event_result: std::result::Result<ServerSentEvent, LinesCodecError>,
    ) -> Option<GeminiResult<GenerateContentResponseResult>> {
        let data = event_result.map_err(Into::<GeminiError>::into).ok()?.data?;

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
        let endpoint_url: String = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        );
        let resp = self
            .client
            .post(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .json(&request)
            .send()
            .await?;

        let status = resp.status();
        let txt_json = resp.text().await?;
        tracing::debug!("generate_content response: {:?}", txt_json);

        if !status.is_success() {
            if let Ok(gemini_error) =
                serde_json::from_str::<crate::types::GeminiApiError>(&txt_json)
            {
                return Err(GeminiError::GeminiError(gemini_error));
            }
            // Fallback if parsing fails, though it should ideally match GeminiApiError
            return Err(GeminiError::GenericApiError {
                status: status.as_u16(),
                body: txt_json,
            });
        }

        match serde_json::from_str::<GenerateContentResponse>(&txt_json) {
            Ok(response) => Ok(response.into_result()?),
            Err(e) => {
                tracing::error!("Failed to parse response: {} with error {}", txt_json, e);
                Err(e.into())
            }
        }
    }

    /// Generates text embeddings for the given input.
    pub async fn text_embeddings(
        &self,
        request: &TextEmbeddingRequest,
        model: &str,
    ) -> GeminiResult<TextEmbeddingResponseOk> {
        let endpoint_url =
            format!("https://generativelanguage.googleapis.com/v1beta/models/{model}:predict");
        let resp = self
            .client
            .post(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .json(&request)
            .send()
            .await?;

        let status = resp.status();
        let txt_json = resp.text().await?;
        tracing::debug!("text_embeddings response: {:?}", txt_json);

        if !status.is_success() {
            if let Ok(gemini_error) =
                serde_json::from_str::<crate::types::GeminiApiError>(&txt_json)
            {
                return Err(GeminiError::GeminiError(gemini_error));
            }
            return Err(GeminiError::GenericApiError {
                status: status.as_u16(),
                body: txt_json,
            });
        }

        match serde_json::from_str::<TextEmbeddingResponse>(&txt_json) {
            Ok(response) => Ok(response.into_result()?),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Counts the number of tokens in the given content.
    pub async fn count_tokens(
        &self,
        request: &CountTokensRequest,
        model: &str,
    ) -> GeminiResult<CountTokensResponseResult> {
        let endpoint_url =
            format!("https://generativelanguage.googleapis.com/v1beta/models/{model}:countTokens");
        let resp = self
            .client
            .post(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .json(&request)
            .send()
            .await?;

        let status = resp.status();
        let txt_json = resp.text().await?;
        tracing::debug!("count_tokens response: {:?}", txt_json);

        if !status.is_success() {
            if let Ok(gemini_error) =
                serde_json::from_str::<crate::types::GeminiApiError>(&txt_json)
            {
                return Err(GeminiError::GeminiError(gemini_error));
            }
            return Err(GeminiError::GenericApiError {
                status: status.as_u16(),
                body: txt_json,
            });
        }

        match serde_json::from_str::<CountTokensResponse>(&txt_json) {
            Ok(response) => Ok(response.into_result()?),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }

    /// Generates images from a text prompt using an Imagen model.
    pub async fn predict_image(
        &self,
        request: &PredictImageRequest,
        model: &str,
    ) -> GeminiResult<PredictImageResponse> {
        let endpoint_url =
            format!("https://generativelanguage.googleapis.com/v1beta/models/{model}:predict");

        let resp = self
            .client
            .post(&endpoint_url)
            .header("x-goog-api-key", &self.api_key)
            .json(&request)
            .send()
            .await?;

        let status = resp.status();
        let txt_json = resp.text().await?;

        if !status.is_success() {
            if let Ok(gemini_error) =
                serde_json::from_str::<crate::types::GeminiApiError>(&txt_json)
            {
                return Err(GeminiError::GeminiError(gemini_error));
            }
            return Err(GeminiError::GenericApiError {
                status: status.as_u16(),
                body: txt_json,
            });
        }

        match serde_json::from_str::<PredictImageResponse>(&txt_json) {
            Ok(response) => Ok(response),
            Err(e) => {
                error!(response = txt_json, error = ?e, "Failed to parse response");
                Err(e.into())
            }
        }
    }
}
