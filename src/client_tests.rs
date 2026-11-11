// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::network::event_source::ServerSentEvent;
use wiremock::matchers::{body_bytes, body_json, header, headers, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

type Result<T> = core::result::Result<T, Box<dyn std::error::Error>>;

#[test]
fn stream_parses_blocked_prompt_response() -> Result<()> {
    let event = Ok(ServerSentEvent {
        data: Some(
            r#"{"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":5}}"#
                .to_string(),
        ),
        ..Default::default()
    });

    let result = GeminiClient::parse_event(event).ok_or("SSE event should produce a result")?;
    let response = result?;
    assert!(response.candidates.is_empty());
    assert_eq!(
        response
            .prompt_feedback
            .ok_or("missing prompt_feedback")?
            .block_reason,
        Some(crate::prelude::BlockReason::Safety)
    );
    Ok(())
}

#[test]
fn resource_url_handles_prefixed_and_unprefixed_names() {
    let client = GeminiClient::new("test-api-key".into());
    assert_eq!(
        client.cached_content_resource_url("cachedContents/abc123xyz"),
        "https://generativelanguage.googleapis.com/v1beta/cachedContents/abc123xyz"
    );
    assert_eq!(
        client.cached_content_resource_url("abc123xyz"),
        "https://generativelanguage.googleapis.com/v1beta/cachedContents/abc123xyz"
    );
    assert_eq!(
        client.model_resource_url("models/gemini-2.5-flash"),
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash"
    );
    assert_eq!(
        client.model_resource_url("gemini-2.5-flash"),
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash"
    );
    assert_eq!(
        client.file_resource_url("files/clip-123abc"),
        "https://generativelanguage.googleapis.com/v1beta/files/clip-123abc"
    );
    assert_eq!(
        client.file_resource_url("clip-123abc"),
        "https://generativelanguage.googleapis.com/v1beta/files/clip-123abc"
    );
    assert_eq!(
        client.batch_resource_url("batches/batch-123"),
        "https://generativelanguage.googleapis.com/v1beta/batches/batch-123"
    );
    assert_eq!(
        client.batch_resource_url("batch-123"),
        "https://generativelanguage.googleapis.com/v1beta/batches/batch-123"
    );
    assert_eq!(
        client.file_search_store_resource_url("fileSearchStores/my-store-123"),
        "https://generativelanguage.googleapis.com/v1beta/fileSearchStores/my-store-123"
    );
    assert_eq!(
        client.file_search_store_resource_url("my-store-123"),
        "https://generativelanguage.googleapis.com/v1beta/fileSearchStores/my-store-123"
    );
}

#[test]
fn all_endpoint_urls_build_from_configured_base_url() {
    let client = GeminiClient::builder("test-api-key")
        .base_url("https://custom.endpoint.example/v1")
        .build();

    assert_eq!(
        client.endpoint_url("models/gemini-2.0-flash:streamGenerateContent?alt=sse"),
        "https://custom.endpoint.example/v1/models/gemini-2.0-flash:streamGenerateContent?alt=sse"
    );
    assert_eq!(
        client.endpoint_url("models/gemini-2.0-flash:generateContent"),
        "https://custom.endpoint.example/v1/models/gemini-2.0-flash:generateContent"
    );
    assert_eq!(
        client.endpoint_url("models/gemini-embedding-001:embedContent"),
        "https://custom.endpoint.example/v1/models/gemini-embedding-001:embedContent"
    );
    assert_eq!(
        client.endpoint_url("models/gemini-embedding-001:batchEmbedContents"),
        "https://custom.endpoint.example/v1/models/gemini-embedding-001:batchEmbedContents"
    );
    assert_eq!(
        client.endpoint_url("models/gemini-2.0-flash:countTokens"),
        "https://custom.endpoint.example/v1/models/gemini-2.0-flash:countTokens"
    );
    assert_eq!(
        client.endpoint_url("models/imagen-3.0-generate-002:predict"),
        "https://custom.endpoint.example/v1/models/imagen-3.0-generate-002:predict"
    );
    assert_eq!(
        client.endpoint_url("cachedContents"),
        "https://custom.endpoint.example/v1/cachedContents"
    );
    assert_eq!(
        client.cached_content_resource_url("cachedContents/abc123xyz"),
        "https://custom.endpoint.example/v1/cachedContents/abc123xyz"
    );
    assert_eq!(
        client.cached_content_resource_url("abc123xyz"),
        "https://custom.endpoint.example/v1/cachedContents/abc123xyz"
    );
    assert_eq!(
        client.endpoint_url("models"),
        "https://custom.endpoint.example/v1/models"
    );
    assert_eq!(
        client.model_resource_url("models/gemini-2.5-flash"),
        "https://custom.endpoint.example/v1/models/gemini-2.5-flash"
    );
    assert_eq!(
        client.model_resource_url("gemini-2.5-flash"),
        "https://custom.endpoint.example/v1/models/gemini-2.5-flash"
    );
    assert_eq!(
        client.endpoint_url("files"),
        "https://custom.endpoint.example/v1/files"
    );
    assert_eq!(
        client.file_resource_url("files/clip-123abc"),
        "https://custom.endpoint.example/v1/files/clip-123abc"
    );
    assert_eq!(
        client.file_resource_url("clip-123abc"),
        "https://custom.endpoint.example/v1/files/clip-123abc"
    );
    assert_eq!(
        client.upload_endpoint_url("files"),
        "https://custom.endpoint.example/upload/v1/files"
    );
    assert_eq!(
        client.endpoint_url("fileSearchStores"),
        "https://custom.endpoint.example/v1/fileSearchStores"
    );
    assert_eq!(
        client.file_search_store_resource_url("fileSearchStores/my-store-123"),
        "https://custom.endpoint.example/v1/fileSearchStores/my-store-123"
    );
    assert_eq!(
        client.file_search_store_resource_url("my-store-123"),
        "https://custom.endpoint.example/v1/fileSearchStores/my-store-123"
    );
}

#[test]
fn upload_endpoint_url_derives_upload_prefix_from_configured_base_url() {
    let default_client = GeminiClient::new("test-api-key".into());
    assert_eq!(
        default_client.upload_endpoint_url("files"),
        "https://generativelanguage.googleapis.com/upload/v1beta/files"
    );
    assert_eq!(
        default_client.upload_endpoint_url("/files"),
        "https://generativelanguage.googleapis.com/upload/v1beta/files"
    );

    let trailing_slash = GeminiClient::builder("test-api-key")
        .base_url("https://generativelanguage.googleapis.com/v1beta/")
        .build();
    assert_eq!(
        trailing_slash.upload_endpoint_url("files"),
        "https://generativelanguage.googleapis.com/upload/v1beta/files"
    );

    let prefixed_proxy = GeminiClient::builder("test-api-key")
        .base_url("https://proxy.example.com/gemini/v1beta")
        .build();
    assert_eq!(
        prefixed_proxy.upload_endpoint_url("files"),
        "https://proxy.example.com/gemini/upload/v1beta/files"
    );

    let unversioned_root = GeminiClient::builder("test-api-key")
        .base_url("http://127.0.0.1:8080")
        .build();
    assert_eq!(
        unversioned_root.upload_endpoint_url("files"),
        "http://127.0.0.1:8080/upload/files"
    );

    let unversioned_prefix = GeminiClient::builder("test-api-key")
        .base_url("https://proxy.example.com/custom-api")
        .build();
    assert_eq!(
        unversioned_prefix.upload_endpoint_url("files"),
        "https://proxy.example.com/custom-api/upload/files"
    );
}

#[test]
fn endpoint_url_normalizes_slashes() {
    let client = GeminiClient::builder("test-api-key")
        .base_url("https://custom.endpoint.example/v1/")
        .build();

    assert_eq!(
        client.endpoint_url("/models/gemini-2.0-flash:generateContent"),
        "https://custom.endpoint.example/v1/models/gemini-2.0-flash:generateContent"
    );
    assert_eq!(
        client.endpoint_url("models/gemini-2.0-flash:generateContent"),
        "https://custom.endpoint.example/v1/models/gemini-2.0-flash:generateContent"
    );
}

#[test]
fn default_client_uses_default_base_url() {
    let client = GeminiClient::new("test-api-key".into());
    assert_eq!(client.base_url(), DEFAULT_BASE_URL);
    assert_eq!(client.api_key(), "test-api-key");
    assert_eq!(
        client.endpoint_url("models/gemini-2.0-flash:generateContent"),
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    );
}

#[tokio::test]
async fn generate_content_against_mock_server_checks_path_and_api_key() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:generateContent"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "candidates": [{
                "index": 0,
                "content": {
                    "parts": [{"text": "Hello from mock server!"}],
                    "role": "model"
                }
            }]
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let request = GenerateContentRequest::builder()
        .contents(vec![Content::builder().add_text_part("Hello").build()])
        .build();

    let response = client
        .generate_content(&request, "gemini-2.0-flash")
        .await?;
    let candidate = response.candidates.first().ok_or("expected candidate")?;
    assert_eq!(
        candidate.get_text().as_deref(),
        Some("Hello from mock server!")
    );
    Ok(())
}

#[tokio::test]
async fn http_client_is_used_when_provided() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:generateContent"))
        .and(header("x-goog-api-key", "test-api-key"))
        .and(header("x-custom-test-header", "custom-client-val"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "candidates": [{
                "index": 0,
                "content": {
                    "parts": [{"text": "Custom HTTP client used!"}],
                    "role": "model"
                }
            }]
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        "x-custom-test-header",
        reqwest::header::HeaderValue::from_static("custom-client-val"),
    );
    let custom_http_client = reqwest::Client::builder()
        .default_headers(headers)
        .build()?;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .http_client(custom_http_client)
        .build();

    let request = GenerateContentRequest::builder()
        .contents(vec![Content::builder().add_text_part("Test").build()])
        .build();

    let response = client
        .generate_content(&request, "gemini-2.0-flash")
        .await?;
    let candidate = response.candidates.first().ok_or("expected candidate")?;
    assert_eq!(
        candidate.get_text().as_deref(),
        Some("Custom HTTP client used!")
    );
    Ok(())
}

#[tokio::test]
async fn count_tokens_against_mock_server_checks_path_and_api_key() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:countTokens"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "totalTokens": 25
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let request = CountTokensRequest::builder()
        .contents(vec![Content::builder().add_text_part("Count me").build()])
        .build();

    let response = client.count_tokens(&request, "gemini-2.0-flash").await?;
    assert_eq!(response.total_tokens, 25);
    Ok(())
}

#[tokio::test]
async fn embed_content_against_mock_server_checks_path_body_and_response() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/models/gemini-embedding-001:embedContent"))
        .and(header("x-goog-api-key", "test-api-key"))
        .and(body_json(serde_json::json!({
            "content": {
                "parts": [{"text": "Embed this query"}]
            },
            "taskType": "RETRIEVAL_QUERY",
            "outputDimensionality": 3
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "embedding": {
                "values": [0.1, 0.2, 0.3]
            },
            "usageMetadata": {
                "promptTokenCount": 3
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let request = EmbedContentRequest::builder()
        .text("Embed this query")
        .task_type(TaskType::RetrievalQuery)
        .output_dimensionality(3)
        .build();

    let response = client
        .embed_content(&request, "gemini-embedding-001")
        .await?;
    assert_eq!(response.embedding.values, vec![0.1, 0.2, 0.3]);
    assert_eq!(
        response
            .usage_metadata
            .ok_or("expected usage_metadata")?
            .prompt_token_count,
        Some(3)
    );
    Ok(())
}

#[tokio::test]
async fn batch_embed_contents_against_mock_server_sets_inner_models_and_parses_response()
-> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/models/gemini-embedding-001:batchEmbedContents"))
        .and(header("x-goog-api-key", "test-api-key"))
        .and(body_json(serde_json::json!({
            "requests": [
                {
                    "model": "models/gemini-embedding-001",
                    "content": {
                        "parts": [{"text": "First document"}]
                    },
                    "taskType": "RETRIEVAL_DOCUMENT",
                    "title": "Doc 1"
                },
                {
                    "model": "models/gemini-embedding-001",
                    "content": {
                        "parts": [{"text": "Second document"}]
                    }
                }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "embeddings": [
                {"values": [0.5, -0.5]},
                {"values": [0.25, 0.75]}
            ],
            "usageMetadata": {
                "promptTokenCount": 5
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let batch = BatchEmbedContentsRequest::builder()
        .add_request(
            EmbedContentRequest::builder()
                .text("First document")
                .task_type(TaskType::RetrievalDocument)
                .title("Doc 1")
                .build(),
        )
        .add_request(EmbedContentRequest::from_text("Second document"))
        .build();

    // Passing `models/gemini-embedding-001` also exercises prefix normalization.
    let response = client
        .batch_embed_contents(&batch, "models/gemini-embedding-001")
        .await?;
    assert_eq!(response.embeddings.len(), 2);
    assert_eq!(response.embeddings[0].values, vec![0.5, -0.5]);
    assert_eq!(response.embeddings[1].values, vec![0.25, 0.75]);
    assert_eq!(
        response
            .usage_metadata
            .ok_or("expected usage_metadata")?
            .prompt_token_count,
        Some(5)
    );
    Ok(())
}

#[tokio::test]
async fn error_for_status_maps_structured_gemini_error() -> Result<()> {
    let raw_response = http::Response::builder()
        .status(400)
        .body(reqwest::Body::from(
            r#"{"error":{"code":400,"message":"Invalid TTL duration","status":"INVALID_ARGUMENT"}}"#,
        ))?;
    let response: reqwest::Response = raw_response.into();

    let err = GeminiClient::error_for_status(response)
        .await
        .err()
        .ok_or("expected error")?;

    match err {
        GeminiError::VertexError(vertex_err) => {
            assert_eq!(vertex_err.code, 400);
            assert_eq!(vertex_err.message, "Invalid TTL duration");
            assert_eq!(vertex_err.status, "INVALID_ARGUMENT");
        }
        other => panic!("expected VertexError, got {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn error_for_status_preserves_raw_body_on_unstructured_error() -> Result<()> {
    let raw_response = http::Response::builder()
        .status(502)
        .body(reqwest::Body::from("Bad Gateway"))?;
    let response: reqwest::Response = raw_response.into();

    let err = GeminiClient::error_for_status(response)
        .await
        .err()
        .ok_or("expected error")?;

    match err {
        GeminiError::ApiError { status, body } => {
            assert_eq!(status, reqwest::StatusCode::BAD_GATEWAY);
            assert_eq!(body, "Bad Gateway");
        }
        other => panic!("expected ApiError, got {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn error_for_status_passes_through_success() -> Result<()> {
    let raw_response = http::Response::builder()
        .status(200)
        .body(reqwest::Body::from("{}"))?;
    let response: reqwest::Response = raw_response.into();

    let result = GeminiClient::error_for_status(response).await;
    assert!(result.is_ok());
    Ok(())
}

#[tokio::test]
async fn stream_yields_ok_then_err_on_mid_stream_io_error() -> Result<()> {
    use tokio_util::bytes::Bytes;

    let chunks: Vec<std::result::Result<Bytes, std::io::Error>> = vec![
        Ok(Bytes::from_static(
            b"data: {\"candidates\":[{\"index\":0,\"content\":{\"parts\":[{\"text\":\"Hello\"}],\"role\":\"model\"}}]}\n\n",
        )),
        Err(std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "connection reset mid-stream",
        )),
    ];
    let body = reqwest::Body::wrap_stream(tokio_stream::iter(chunks));
    let response: reqwest::Response = http::Response::builder().status(200).body(body)?.into();

    let mut stream = std::pin::pin!(GeminiClient::response_event_stream(response));

    let first = stream
        .next()
        .await
        .ok_or("expected first Ok item from stream")??;
    assert_eq!(
        first
            .candidates
            .first()
            .and_then(|c| c.get_text())
            .as_deref(),
        Some("Hello")
    );

    let second = stream
        .next()
        .await
        .ok_or("expected Err item after mid-stream I/O error")?;
    match second {
        Err(GeminiError::EventSourceError(LinesCodecError::Io(ref io_err))) => {
            let mut has_root_cause = false;
            let mut current = std::error::Error::source(io_err);
            while let Some(cause) = current {
                if cause.to_string().contains("connection reset mid-stream") {
                    has_root_cause = true;
                    break;
                }
                current = cause.source();
            }
            assert!(
                has_root_cause,
                "expected I/O error source chain to contain root cause, got {io_err:?}"
            );
        }
        other => panic!("expected Err(EventSourceError(LinesCodecError::Io(_))), got {other:?}"),
    }

    assert!(stream.next().await.is_none());
    Ok(())
}

#[tokio::test]
async fn stream_yields_err_when_io_error_interrupts_partial_event() -> Result<()> {
    use tokio_util::bytes::Bytes;

    let chunks: Vec<std::result::Result<Bytes, std::io::Error>> = vec![
        Ok(Bytes::from_static(
            b"data: {\"candidates\":[{\"index\":0,\"content\":{\"parts\":[{\"text\":\"Part 1\"}],\"role\":\"model\"}}]}\n\n",
        )),
        Ok(Bytes::from_static(b"data: {\"candidates\":[{\"index\":0")),
        Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "stream truncated mid-event",
        )),
    ];
    let body = reqwest::Body::wrap_stream(tokio_stream::iter(chunks));
    let response: reqwest::Response = http::Response::builder().status(200).body(body)?.into();

    let mut stream = std::pin::pin!(GeminiClient::response_event_stream(response));

    let first = stream
        .next()
        .await
        .ok_or("expected first Ok item from stream")??;
    assert_eq!(
        first
            .candidates
            .first()
            .and_then(|c| c.get_text())
            .as_deref(),
        Some("Part 1")
    );

    let second = stream
        .next()
        .await
        .ok_or("expected Err item after truncated partial event")?;
    assert!(matches!(
        second,
        Err(GeminiError::EventSourceError(LinesCodecError::Io(_)))
    ));
    Ok(())
}

#[tokio::test]
async fn stream_skips_events_without_data_line_without_error() -> Result<()> {
    use tokio_util::bytes::Bytes;

    assert!(
        GeminiClient::parse_event(Ok(ServerSentEvent {
            event: Some("ping".into()),
            data: None,
            id: Some("1".into()),
            retry: Some(1000),
        }))
        .is_none()
    );

    let chunks: Vec<std::result::Result<Bytes, std::io::Error>> = vec![Ok(Bytes::from_static(
        b": keep-alive\n\nevent: ping\nid: 1\nretry: 1000\n\n\ndata: {\"candidates\":[{\"index\":0,\"content\":{\"parts\":[{\"text\":\"Done\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}]}\n\n",
    ))];
    let body = reqwest::Body::wrap_stream(tokio_stream::iter(chunks));
    let response: reqwest::Response = http::Response::builder().status(200).body(body)?.into();

    let mut stream = std::pin::pin!(GeminiClient::response_event_stream(response));

    let item = stream
        .next()
        .await
        .ok_or("expected Ok item after skipped no-data events")??;
    let candidate = item.candidates.first().ok_or("expected candidate")?;
    assert_eq!(candidate.get_text().as_deref(), Some("Done"));
    assert_eq!(candidate.finish_reason.as_deref(), Some("STOP"));

    assert!(stream.next().await.is_none());
    Ok(())
}

#[test]
fn parse_event_surfaces_codec_max_line_length_error() {
    let result = GeminiClient::parse_event(Err(LinesCodecError::MaxLineLengthExceeded));
    assert!(matches!(
        result,
        Some(Err(GeminiError::EventSourceError(
            LinesCodecError::MaxLineLengthExceeded
        )))
    ));
}

#[tokio::test]
async fn stream_generate_content_against_mock_server_skips_keepalives_and_yields_chunks()
-> Result<()> {
    let mock_server = MockServer::start().await;

    let sse_body = concat!(
        ": keep-alive\n\n",
        "data: {\"candidates\":[{\"index\":0,\"content\":{\"parts\":[{\"text\":\"Hello, \"}],\"role\":\"model\"}}]}\n\n",
        "event: ping\n\n",
        "data: {\"candidates\":[{\"index\":0,\"content\":{\"parts\":[{\"text\":\"world!\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}]}\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:streamGenerateContent"))
        .and(query_param("alt", "sse"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(sse_body, "text/event-stream; charset=utf-8"),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let request = GenerateContentRequest::builder()
        .contents(vec![Content::builder().add_text_part("Greet me").build()])
        .build();

    let stream = client
        .stream_generate_content(&request, "gemini-2.0-flash")
        .await?;
    let mut stream = std::pin::pin!(stream);

    let first = stream.next().await.ok_or("expected first chunk")??;
    assert_eq!(
        first
            .candidates
            .first()
            .and_then(|c| c.get_text())
            .as_deref(),
        Some("Hello, ")
    );
    assert_eq!(
        first
            .candidates
            .first()
            .and_then(|c| c.finish_reason.as_deref()),
        None
    );

    let second = stream.next().await.ok_or("expected second chunk")??;
    assert_eq!(
        second
            .candidates
            .first()
            .and_then(|c| c.get_text())
            .as_deref(),
        Some("world!")
    );
    assert_eq!(
        second
            .candidates
            .first()
            .and_then(|c| c.finish_reason.as_deref()),
        Some("STOP")
    );

    assert!(stream.next().await.is_none());
    Ok(())
}

#[test]
fn encode_query_value_percent_encodes_reserved_characters() {
    assert_eq!(encode_query_value("plain-token_1.0~"), "plain-token_1.0~");
    assert_eq!(
        encode_query_value("ChFtb2RlbHM+1/2==&x=y"),
        "ChFtb2RlbHM%2B1%2F2%3D%3D%26x%3Dy"
    );
}

#[tokio::test]
async fn list_models_against_mock_server_without_pagination_params() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/models"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "models": [
                {
                    "name": "models/gemini-2.5-flash",
                    "version": "001",
                    "displayName": "Gemini 2.5 Flash",
                    "inputTokenLimit": 1_048_576,
                    "outputTokenLimit": 65_536,
                    "supportedGenerationMethods": ["generateContent", "countTokens"],
                    "thinking": true
                }
            ]
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let response = client.list_models(None, None).await?;
    assert_eq!(response.models.len(), 1);
    assert_eq!(response.models[0].name, "models/gemini-2.5-flash");
    assert_eq!(response.models[0].input_token_limit, Some(1_048_576));
    assert_eq!(response.models[0].output_token_limit, Some(65_536));
    assert_eq!(response.models[0].thinking, Some(true));
    assert!(response.next_page_token.is_none());
    Ok(())
}

#[tokio::test]
async fn list_models_against_mock_server_with_pagination_params() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/models"))
        .and(query_param("pageSize", "2"))
        .and(query_param("pageToken", "tok+1/2=="))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "models": [
                {
                    "name": "models/gemini-2.5-pro",
                    "displayName": "Gemini 2.5 Pro",
                    "supportedGenerationMethods": ["generateContent"]
                },
                {
                    "name": "models/gemini-embedding-001",
                    "displayName": "Gemini Embedding 001",
                    "supportedGenerationMethods": ["embedContent"]
                }
            ],
            "nextPageToken": "next-tok=="
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let response = client.list_models(Some(2), Some("tok+1/2==")).await?;
    assert_eq!(response.models.len(), 2);
    assert_eq!(response.models[0].name, "models/gemini-2.5-pro");
    assert_eq!(response.models[1].name, "models/gemini-embedding-001");
    assert_eq!(response.next_page_token.as_deref(), Some("next-tok=="));
    Ok(())
}

#[tokio::test]
async fn get_model_against_mock_server_handles_prefixed_and_unprefixed_name() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/models/gemini-2.5-flash"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "name": "models/gemini-2.5-flash",
            "baseModelId": "gemini-2.5-flash",
            "version": "001",
            "displayName": "Gemini 2.5 Flash",
            "description": "Fast multimodal model.",
            "inputTokenLimit": 1_048_576,
            "outputTokenLimit": 65_536,
            "supportedGenerationMethods": ["generateContent", "countTokens"],
            "temperature": 1.0,
            "maxTemperature": 2.0,
            "topP": 0.95,
            "topK": 64,
            "thinking": true
        })))
        .expect(2)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let unprefixed = client.get_model("gemini-2.5-flash").await?;
    let prefixed = client.get_model("models/gemini-2.5-flash").await?;

    assert_eq!(unprefixed, prefixed);
    assert_eq!(unprefixed.name, "models/gemini-2.5-flash");
    assert_eq!(
        unprefixed.base_model_id.as_deref(),
        Some("gemini-2.5-flash")
    );
    assert_eq!(unprefixed.version.as_deref(), Some("001"));
    assert_eq!(unprefixed.display_name.as_deref(), Some("Gemini 2.5 Flash"));
    assert_eq!(
        unprefixed.description.as_deref(),
        Some("Fast multimodal model.")
    );
    assert_eq!(unprefixed.input_token_limit, Some(1_048_576));
    assert_eq!(unprefixed.output_token_limit, Some(65_536));
    assert_eq!(
        unprefixed.supported_generation_methods,
        vec!["generateContent", "countTokens"]
    );
    assert_eq!(unprefixed.temperature, Some(1.0));
    assert_eq!(unprefixed.max_temperature, Some(2.0));
    assert_eq!(unprefixed.top_p, Some(0.95));
    assert_eq!(unprefixed.top_k, Some(64));
    assert_eq!(unprefixed.thinking, Some(true));
    Ok(())
}

#[tokio::test]
async fn get_model_maps_not_found_error() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/models/does-not-exist"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
            "error": {
                "code": 404,
                "message": "models/does-not-exist is not found for API version v1beta",
                "status": "NOT_FOUND"
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let err = client
        .get_model("does-not-exist")
        .await
        .err()
        .ok_or("expected 404 error")?;

    match err {
        GeminiError::VertexError(vertex_err) => {
            assert_eq!(vertex_err.code, 404);
            assert_eq!(vertex_err.status, "NOT_FOUND");
        }
        other => panic!("expected VertexError, got {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn list_cached_contents_against_mock_server_with_pagination_params() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/cachedContents"))
        .and(query_param("pageSize", "5"))
        .and(query_param("pageToken", "tok+1/2=="))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "cachedContents": [
                {
                    "name": "cachedContents/abc123xyz",
                    "model": "models/gemini-2.5-flash",
                    "displayName": "Test Cache"
                }
            ],
            "nextPageToken": "next-cache-tok=="
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let response = client
        .list_cached_contents(Some(5), Some("tok+1/2=="))
        .await?;
    assert_eq!(response.cached_contents.len(), 1);
    assert_eq!(
        response.cached_contents[0].name.as_deref(),
        Some("cachedContents/abc123xyz")
    );
    assert_eq!(
        response.next_page_token.as_deref(),
        Some("next-cache-tok==")
    );
    Ok(())
}

#[tokio::test]
async fn get_file_against_mock_server_handles_prefixed_and_unprefixed_name() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/files/clip-123abc"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "name": "files/clip-123abc",
            "displayName": "Sample Bunny Clip",
            "mimeType": "video/mp4",
            "sizeBytes": "10485760",
            "createTime": "2026-10-30T15:00:00Z",
            "updateTime": "2026-10-30T15:00:05Z",
            "expirationTime": "2026-11-01T15:00:00Z",
            "sha256Hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "uri": "https://generativelanguage.googleapis.com/v1beta/files/clip-123abc",
            "state": "ACTIVE",
            "source": "UPLOADED",
            "videoMetadata": {
                "videoDuration": "32.5s"
            }
        })))
        .expect(2)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let unprefixed = client.get_file("clip-123abc").await?;
    let prefixed = client.get_file("files/clip-123abc").await?;

    assert_eq!(unprefixed, prefixed);
    assert_eq!(unprefixed.name.as_deref(), Some("files/clip-123abc"));
    assert_eq!(
        unprefixed.display_name.as_deref(),
        Some("Sample Bunny Clip")
    );
    assert_eq!(unprefixed.mime_type.as_deref(), Some("video/mp4"));
    assert_eq!(unprefixed.size_bytes, Some(10_485_760));
    assert_eq!(unprefixed.state, Some(FileState::Active));
    assert_eq!(unprefixed.source, Some(FileSource::Uploaded));
    assert_eq!(
        unprefixed.uri.as_deref(),
        Some("https://generativelanguage.googleapis.com/v1beta/files/clip-123abc")
    );
    assert_eq!(
        unprefixed.video_metadata,
        Some(VideoFileMetadata {
            video_duration: Some("32.5s".to_string()),
        })
    );
    Ok(())
}

#[tokio::test]
async fn get_file_maps_not_found_error() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/files/missing-file"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
            "error": {
                "code": 404,
                "message": "File files/missing-file does not exist.",
                "status": "NOT_FOUND"
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let err = client
        .get_file("missing-file")
        .await
        .err()
        .ok_or("expected 404 error")?;

    match err {
        GeminiError::VertexError(vertex_err) => {
            assert_eq!(vertex_err.code, 404);
            assert_eq!(vertex_err.status, "NOT_FOUND");
        }
        other => panic!("expected VertexError, got {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn list_files_against_mock_server_without_pagination_params() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/files"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "files": [
                {
                    "name": "files/doc-1",
                    "displayName": "Notes.txt",
                    "mimeType": "text/plain",
                    "sizeBytes": "512",
                    "state": "ACTIVE",
                    "source": "UPLOADED"
                }
            ]
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let response = client.list_files(None, None).await?;
    assert_eq!(response.files.len(), 1);
    assert_eq!(response.files[0].name.as_deref(), Some("files/doc-1"));
    assert_eq!(response.files[0].size_bytes, Some(512));
    assert_eq!(response.files[0].state, Some(FileState::Active));
    assert!(response.next_page_token.is_none());
    Ok(())
}

#[tokio::test]
async fn list_files_against_mock_server_with_pagination_params() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/files"))
        .and(query_param("pageSize", "10"))
        .and(query_param("pageToken", "tok+1/2=="))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "files": [
                {
                    "name": "files/doc-1",
                    "state": "ACTIVE"
                },
                {
                    "name": "files/vid-2",
                    "state": "PROCESSING"
                }
            ],
            "nextPageToken": "next-files-tok=="
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let response = client.list_files(Some(10), Some("tok+1/2==")).await?;
    assert_eq!(response.files.len(), 2);
    assert_eq!(response.files[0].name.as_deref(), Some("files/doc-1"));
    assert_eq!(response.files[0].state, Some(FileState::Active));
    assert_eq!(response.files[1].name.as_deref(), Some("files/vid-2"));
    assert_eq!(response.files[1].state, Some(FileState::Processing));
    assert_eq!(
        response.next_page_token.as_deref(),
        Some("next-files-tok==")
    );
    Ok(())
}

#[tokio::test]
async fn delete_file_against_mock_server_handles_prefixed_and_unprefixed_name() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("DELETE"))
        .and(path("/files/clip-123abc"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
        .expect(2)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    client.delete_file("clip-123abc").await?;
    client.delete_file("files/clip-123abc").await?;
    Ok(())
}

#[tokio::test]
async fn delete_file_maps_permission_denied_error() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("DELETE"))
        .and(path("/files/other-project-file"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(403).set_body_json(serde_json::json!({
            "error": {
                "code": 403,
                "message": "You do not have permission to access the File other-project-file or it may not exist.",
                "status": "PERMISSION_DENIED"
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let err = client
        .delete_file("files/other-project-file")
        .await
        .err()
        .ok_or("expected 403 error")?;

    match err {
        GeminiError::VertexError(vertex_err) => {
            assert_eq!(vertex_err.code, 403);
            assert_eq!(vertex_err.status, "PERMISSION_DENIED");
        }
        other => panic!("expected VertexError, got {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn upload_file_executes_two_step_resumable_protocol_with_display_name() -> Result<()> {
    let mock_server = MockServer::start().await;
    let payload: &[u8] = b"\x00\x01Hello resumable binary upload!\xff";
    let session_url = format!(
        "{}/upload/v1beta/files?upload_id=session-xyz-123",
        mock_server.uri()
    );

    // Step 1: Start resumable upload against `/upload/files` (unversioned mock root).
    Mock::given(method("POST"))
        .and(path("/upload/files"))
        .and(header("x-goog-api-key", "test-api-key"))
        .and(header("x-goog-upload-protocol", "resumable"))
        .and(header("x-goog-upload-command", "start"))
        .and(header(
            "x-goog-upload-header-content-length",
            payload.len().to_string().as_str(),
        ))
        .and(header(
            "x-goog-upload-header-content-type",
            "application/octet-stream",
        ))
        .and(body_json(serde_json::json!({
            "file": {
                "displayName": "sample.bin"
            }
        })))
        .respond_with(
            ResponseTemplate::new(200).insert_header("x-goog-upload-url", session_url.as_str()),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    // Step 2: Upload raw bytes and finalize against the pre-authenticated `x-goog-upload-url`.
    Mock::given(method("POST"))
        .and(path("/upload/v1beta/files"))
        .and(query_param("upload_id", "session-xyz-123"))
        .and(header("content-length", payload.len().to_string().as_str()))
        .and(header("x-goog-upload-offset", "0"))
        .and(headers("x-goog-upload-command", vec!["upload", "finalize"]))
        .and(body_bytes(payload.to_vec()))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "file": {
                "name": "files/bin-123",
                "displayName": "sample.bin",
                "mimeType": "application/octet-stream",
                "sizeBytes": payload.len().to_string(),
                "state": "PROCESSING",
                "source": "UPLOADED",
                "uri": "https://generativelanguage.googleapis.com/v1beta/files/bin-123"
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let file = client
        .upload_file(payload, "application/octet-stream", Some("sample.bin"))
        .await?;

    assert_eq!(file.name.as_deref(), Some("files/bin-123"));
    assert_eq!(file.display_name.as_deref(), Some("sample.bin"));
    assert_eq!(file.mime_type.as_deref(), Some("application/octet-stream"));
    assert_eq!(file.size_bytes, Some(payload.len() as i64));
    assert_eq!(file.state, Some(FileState::Processing));
    assert_eq!(file.source, Some(FileSource::Uploaded));
    assert_eq!(
        file.uri.as_deref(),
        Some("https://generativelanguage.googleapis.com/v1beta/files/bin-123")
    );

    // Ensure Step 2 sent the exact `upload, finalize` command header and did NOT send
    // `x-goog-api-key` to the pre-authenticated session URL.
    let received = mock_server
        .received_requests()
        .await
        .ok_or("request recording should be enabled")?;
    assert_eq!(received.len(), 2);
    assert!(received[0].headers.contains_key("x-goog-api-key"));
    assert!(!received[1].headers.contains_key("x-goog-api-key"));
    assert_eq!(
        received[1]
            .headers
            .get("x-goog-upload-command")
            .and_then(|v| v.to_str().ok()),
        Some("upload, finalize")
    );
    Ok(())
}

#[tokio::test]
async fn upload_file_without_display_name_sends_empty_metadata_object() -> Result<()> {
    let mock_server = MockServer::start().await;
    let payload: &[u8] = b"Poem without display name";
    let session_url = format!("{}/upload-session/1", mock_server.uri());

    Mock::given(method("POST"))
        .and(path("/upload/files"))
        .and(header("x-goog-api-key", "test-api-key"))
        .and(header("x-goog-upload-protocol", "resumable"))
        .and(header("x-goog-upload-command", "start"))
        .and(header("x-goog-upload-header-content-type", "text/plain"))
        .and(body_json(serde_json::json!({})))
        .respond_with(
            ResponseTemplate::new(200).insert_header("X-Goog-Upload-URL", session_url.as_str()),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    Mock::given(method("POST"))
        .and(path("/upload-session/1"))
        .and(header("x-goog-upload-offset", "0"))
        .and(headers("x-goog-upload-command", vec!["upload", "finalize"]))
        .and(body_bytes(payload.to_vec()))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "file": {
                "name": "files/generated-id",
                "mimeType": "text/plain",
                "state": "ACTIVE"
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let file = client.upload_file(payload, "text/plain", None).await?;
    assert_eq!(file.name.as_deref(), Some("files/generated-id"));
    assert_eq!(file.state, Some(FileState::Active));
    Ok(())
}

#[tokio::test]
async fn upload_file_with_metadata_normalizes_unprefixed_and_preserves_prefixed_name() -> Result<()>
{
    let mock_server = MockServer::start().await;
    let payload: &[u8] = b"# Markdown Spec";
    let session_url = format!(
        "{}/upload/v1beta/files?upload_id=custom-meta-1",
        mock_server.uri()
    );

    Mock::given(method("POST"))
        .and(path("/upload/v1beta/files"))
        .and(header("x-goog-api-key", "test-api-key"))
        .and(header("x-goog-upload-protocol", "resumable"))
        .and(header("x-goog-upload-command", "start"))
        .and(header("x-goog-upload-header-content-type", "text/markdown"))
        .and(body_json(serde_json::json!({
            "file": {
                "name": "files/custom-spec-id",
                "displayName": "Spec.md"
            }
        })))
        .respond_with(
            ResponseTemplate::new(200).insert_header("x-goog-upload-url", session_url.as_str()),
        )
        .expect(2)
        .mount(&mock_server)
        .await;

    Mock::given(method("POST"))
        .and(path("/upload/v1beta/files"))
        .and(query_param("upload_id", "custom-meta-1"))
        .and(header("x-goog-upload-offset", "0"))
        .and(headers("x-goog-upload-command", vec!["upload", "finalize"]))
        .and(body_bytes(payload.to_vec()))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "file": {
                "name": "files/custom-spec-id",
                "displayName": "Spec.md",
                "mimeType": "text/markdown",
                "state": "ACTIVE"
            }
        })))
        .expect(2)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(format!("{}/v1beta", mock_server.uri()))
        .build();

    let unprefixed = File::builder()
        .name("custom-spec-id")
        .display_name("Spec.md")
        .build();
    let file_from_unprefixed = client
        .upload_file_with_metadata(payload, "text/markdown", &unprefixed)
        .await?;

    let already_prefixed = File::builder()
        .name("files/custom-spec-id")
        .display_name("Spec.md")
        .build();
    let file_from_prefixed = client
        .upload_file_with_metadata(payload, "text/markdown", &already_prefixed)
        .await?;

    assert_eq!(file_from_unprefixed, file_from_prefixed);
    assert_eq!(
        file_from_unprefixed.name.as_deref(),
        Some("files/custom-spec-id")
    );
    assert_eq!(
        file_from_unprefixed.display_name.as_deref(),
        Some("Spec.md")
    );
    assert_eq!(file_from_unprefixed.state, Some(FileState::Active));
    Ok(())
}

#[tokio::test]
async fn upload_file_with_metadata_ignores_empty_name_and_strips_output_only_fields() -> Result<()>
{
    let mock_server = MockServer::start().await;
    let payload: &[u8] = b"video-bytes";
    let session_url = format!("{}/upload-session/stripped", mock_server.uri());

    Mock::given(method("POST"))
        .and(path("/upload/files"))
        .and(header("x-goog-api-key", "test-api-key"))
        .and(header("x-goog-upload-protocol", "resumable"))
        .and(header("x-goog-upload-command", "start"))
        .and(body_json(serde_json::json!({
            "file": {
                "displayName": "Reuploaded Clip.mp4"
            }
        })))
        .respond_with(
            ResponseTemplate::new(200).insert_header("x-goog-upload-url", session_url.as_str()),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    Mock::given(method("POST"))
        .and(path("/upload-session/stripped"))
        .and(header("x-goog-upload-offset", "0"))
        .and(headers("x-goog-upload-command", vec!["upload", "finalize"]))
        .and(body_bytes(payload.to_vec()))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "file": {
                "name": "files/new-server-id",
                "displayName": "Reuploaded Clip.mp4",
                "mimeType": "video/mp4",
                "state": "PROCESSING"
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let fetched_like_file = File::builder()
        .name("")
        .display_name("Reuploaded Clip.mp4")
        .mime_type("video/mp4")
        .size_bytes(999_999)
        .create_time("2026-10-30T00:00:00Z")
        .update_time("2026-10-30T00:01:00Z")
        .expiration_time("2026-11-01T00:00:00Z")
        .sha256_hash("deadbeef")
        .uri("https://generativelanguage.googleapis.com/v1beta/files/old-id")
        .download_uri("https://generativelanguage.googleapis.com/v1beta/files/old-id:download")
        .state(FileState::Active)
        .source(FileSource::Uploaded)
        .video_metadata(VideoFileMetadata {
            video_duration: Some("12s".to_string()),
        })
        .build();

    let uploaded = client
        .upload_file_with_metadata(payload, "video/mp4", &fetched_like_file)
        .await?;
    assert_eq!(uploaded.name.as_deref(), Some("files/new-server-id"));
    assert_eq!(
        uploaded.display_name.as_deref(),
        Some("Reuploaded Clip.mp4")
    );
    Ok(())
}

#[tokio::test]
async fn upload_file_returns_unexpected_response_when_upload_url_header_is_missing() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/upload/files"))
        .respond_with(ResponseTemplate::new(200).set_body_string("missing header"))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let err = client
        .upload_file(b"abc", "text/plain", None)
        .await
        .err()
        .ok_or("expected missing upload header error")?;

    match &err {
        GeminiError::UnexpectedResponse { message } => {
            assert!(message.contains("x-goog-upload-url"));
        }
        other => panic!("expected UnexpectedResponse, got {other:?}"),
    }
    assert!(err.to_string().starts_with("Unexpected API response:"));
    Ok(())
}

#[tokio::test]
async fn upload_file_maps_structured_gemini_error_on_initiation() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/upload/files"))
        .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
            "error": {
                "code": 400,
                "message": "Invalid file resource name.",
                "status": "INVALID_ARGUMENT"
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let metadata = File::builder().name("files/-invalid-").build();
    let err = client
        .upload_file_with_metadata(b"abc", "text/plain", &metadata)
        .await
        .err()
        .ok_or("expected 400 error")?;

    match err {
        GeminiError::VertexError(vertex_err) => {
            assert_eq!(vertex_err.code, 400);
            assert_eq!(vertex_err.status, "INVALID_ARGUMENT");
        }
        other => panic!("expected VertexError, got {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn files_end_to_end_upload_poll_generate_and_delete_lifecycle() -> Result<()> {
    let mock_server = MockServer::start().await;
    let payload: &[u8] = b"Pedra da Mina: 2,798m";
    let session_url = format!("{}/upload/v1beta/files?upload_id=e2e-1", mock_server.uri());
    let file_uri = "https://generativelanguage.googleapis.com/v1beta/files/notes-e2e";

    // 1a. Initiate resumable upload.
    Mock::given(method("POST"))
        .and(path("/upload/v1beta/files"))
        .and(header("x-goog-api-key", "test-api-key"))
        .and(header("x-goog-upload-protocol", "resumable"))
        .and(header("x-goog-upload-command", "start"))
        .respond_with(
            ResponseTemplate::new(200).insert_header("x-goog-upload-url", session_url.as_str()),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    // 1b. Finalize upload -> initially `PROCESSING`.
    Mock::given(method("POST"))
        .and(path("/upload/v1beta/files"))
        .and(query_param("upload_id", "e2e-1"))
        .and(headers("x-goog-upload-command", vec!["upload", "finalize"]))
        .and(body_bytes(payload.to_vec()))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "file": {
                "name": "files/notes-e2e",
                "displayName": "notes.txt",
                "mimeType": "text/plain",
                "sizeBytes": payload.len().to_string(),
                "state": "PROCESSING",
                "uri": file_uri
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    // 2. Poll `get_file` -> transitions to `ACTIVE`.
    Mock::given(method("GET"))
        .and(path("/v1beta/files/notes-e2e"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "name": "files/notes-e2e",
            "displayName": "notes.txt",
            "mimeType": "text/plain",
            "sizeBytes": payload.len().to_string(),
            "state": "ACTIVE",
            "uri": file_uri
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    // 3. Reference `fileData` in `generate_content`.
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.5-flash:generateContent"))
        .and(header("x-goog-api-key", "test-api-key"))
        .and(body_json(serde_json::json!({
            "contents": [{
                "role": "user",
                "parts": [
                    {
                        "fileData": {
                            "fileUri": file_uri,
                            "mimeType": "text/plain"
                        }
                    },
                    {
                        "text": "What is the elevation?"
                    }
                ]
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "candidates": [{
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [{"text": "2,798 meters."}]
                }
            }]
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    // 4. Delete the file.
    Mock::given(method("DELETE"))
        .and(path("/v1beta/files/notes-e2e"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(format!("{}/v1beta", mock_server.uri()))
        .build();

    let mut file = client
        .upload_file(payload, "text/plain", Some("notes.txt"))
        .await?;
    assert_eq!(file.state, Some(FileState::Processing));
    let file_name = file.name.clone().ok_or("expected file name")?;

    while file.state != Some(FileState::Active) {
        file = client.get_file(&file_name).await?;
    }

    let file_part = Part {
        thought: None,
        thought_signature: None,
        part_metadata: None,
        media_resolution: None,
        video_metadata: None,
        speech_metadata: None,
        data: PartData::FileData(serde_json::json!({
            "fileUri": file.uri.as_deref().ok_or("expected uri")?,
            "mimeType": file.mime_type.as_deref().ok_or("expected mimeType")?
        })),
    };
    let request = GenerateContentRequest::builder()
        .contents(vec![
            Content::builder()
                .role(Role::User)
                .add_part(file_part)
                .add_text_part("What is the elevation?")
                .build(),
        ])
        .build();

    let response = client
        .generate_content(&request, "gemini-2.5-flash")
        .await?;
    assert_eq!(
        response
            .candidates
            .first()
            .and_then(|c| c.get_text())
            .as_deref(),
        Some("2,798 meters.")
    );

    client.delete_file(&file_name).await?;
    Ok(())
}

#[tokio::test]
async fn batch_generate_content_against_mock_server_checks_path_key_body_and_response() -> Result<()>
{
    let mock_server = MockServer::start().await;

    let request_body = serde_json::json!({
        "batch": {
            "displayName": "test-batch",
            "inputConfig": {
                "requests": {
                    "requests": [
                        {
                            "request": {
                                "contents": [
                                    {
                                        "role": "user",
                                        "parts": [{ "text": "Say hi!" }]
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.5-flash:batchGenerateContent"))
        .and(header("x-goog-api-key", "test-api-key"))
        .and(body_json(request_body.clone()))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "name": "batches/op-batch-123",
            "metadata": {
                "name": "batches/op-batch-123",
                "model": "models/gemini-2.5-flash",
                "displayName": "test-batch",
                "state": "BATCH_STATE_PENDING",
                "createTime": "2026-10-30T19:00:00Z",
                "updateTime": "2026-10-30T19:00:00Z",
                "batchStats": {
                    "requestCount": "1",
                    "pendingRequestCount": "1"
                }
            }
        })))
        .expect(2)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(format!("{}/v1beta", mock_server.uri()))
        .build();

    let request: BatchGenerateContentRequest = serde_json::from_value(request_body)?;

    // 1. Unprefixed model ID
    let op = client
        .batch_generate_content(&request, "gemini-2.5-flash")
        .await?;

    assert_eq!(op.name.as_deref(), Some("batches/op-batch-123"));
    assert!(!op.is_done());
    assert_eq!(op.state(), Some(BatchState::Pending));
    let batch = op.batch().expect("batch metadata");
    assert_eq!(batch.display_name.as_deref(), Some("test-batch"));
    assert_eq!(
        batch.batch_stats.as_ref().and_then(|s| s.request_count),
        Some(1)
    );

    // 2. Prefixed model ID ("models/gemini-2.5-flash")
    let op_prefixed = client
        .batch_generate_content(&request, "models/gemini-2.5-flash")
        .await?;
    assert_eq!(op_prefixed.name.as_deref(), Some("batches/op-batch-123"));
    assert_eq!(op_prefixed.state(), Some(BatchState::Pending));
    Ok(())
}

#[tokio::test]
async fn get_batch_against_mock_server_handles_prefixed_and_unprefixed_name() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1beta/batches/batch-456"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "name": "batches/batch-456",
            "metadata": {
                "name": "batches/batch-456",
                "model": "models/gemini-2.5-flash",
                "displayName": "completed-batch",
                "state": "BATCH_STATE_SUCCEEDED",
                "output": {
                    "responsesFile": "files/output-jsonl-123"
                }
            },
            "done": true,
            "response": {
                "responsesFile": "files/output-jsonl-123"
            }
        })))
        .expect(2)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(format!("{}/v1beta", mock_server.uri()))
        .build();

    // 1. With prefix
    let op1 = client.get_batch("batches/batch-456").await?;
    assert!(op1.is_done());
    assert_eq!(op1.state(), Some(BatchState::Succeeded));
    assert_eq!(
        op1.output().and_then(|o| o.responses_file.as_deref()),
        Some("files/output-jsonl-123")
    );

    // 2. Without prefix
    let op2 = client.get_batch("batch-456").await?;
    assert!(op2.is_done());
    assert_eq!(op2.state(), Some(BatchState::Succeeded));
    Ok(())
}

#[tokio::test]
async fn batch_generate_content_maps_structured_gemini_error() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.5-flash:batchGenerateContent"))
        .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
            "error": {
                "code": 400,
                "message": "Invalid inputConfig: neither requests nor fileName specified",
                "status": "INVALID_ARGUMENT"
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(format!("{}/v1beta", mock_server.uri()))
        .build();

    let request = BatchGenerateContentRequest::new(GenerateContentBatch::default());
    let err = client
        .batch_generate_content(&request, "gemini-2.5-flash")
        .await
        .unwrap_err();

    match err {
        crate::error::Error::VertexError(vertex_err) => {
            assert_eq!(vertex_err.code, 400);
            assert_eq!(vertex_err.status, "INVALID_ARGUMENT");
            assert!(vertex_err.message.contains("Invalid inputConfig"));
        }
        other => panic!("expected VertexError, got: {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn get_batch_maps_not_found_error() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1beta/batches/nonexistent"))
        .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
            "error": {
                "code": 404,
                "message": "Batch not found: batches/nonexistent",
                "status": "NOT_FOUND"
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(format!("{}/v1beta", mock_server.uri()))
        .build();

    let err = client.get_batch("nonexistent").await.unwrap_err();

    match err {
        crate::error::Error::VertexError(vertex_err) => {
            assert_eq!(vertex_err.code, 404);
            assert_eq!(vertex_err.status, "NOT_FOUND");
        }
        other => panic!("expected VertexError, got: {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn list_batches_against_mock_server_without_pagination_params() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1beta/batches"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "operations": [
                {
                    "name": "batches/batch-1",
                    "metadata": {
                        "name": "batches/batch-1",
                        "model": "models/gemini-2.5-flash",
                        "displayName": "batch-one",
                        "state": "BATCH_STATE_SUCCEEDED"
                    },
                    "done": true
                },
                {
                    "name": "batches/batch-2",
                    "metadata": {
                        "name": "batches/batch-2",
                        "model": "models/gemini-2.5-flash",
                        "displayName": "batch-two",
                        "state": "BATCH_STATE_RUNNING"
                    },
                    "done": false
                }
            ],
            "nextPageToken": "token-page-2"
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(format!("{}/v1beta", mock_server.uri()))
        .build();

    let resp = client.list_batches(None, None).await?;
    assert_eq!(resp.operations.len(), 2);
    assert_eq!(resp.next_page_token.as_deref(), Some("token-page-2"));

    let display_names: Vec<&str> = resp
        .batches()
        .filter_map(|b| b.display_name.as_deref())
        .collect();
    assert_eq!(display_names, vec!["batch-one", "batch-two"]);
    Ok(())
}

#[tokio::test]
async fn list_batches_against_mock_server_with_pagination_params() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1beta/batches"))
        .and(query_param("pageSize", "10"))
        .and(query_param("pageToken", "page/cursor 123"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "operations": [],
            "nextPageToken": null
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(format!("{}/v1beta", mock_server.uri()))
        .build();

    let resp = client
        .list_batches(Some(10), Some("page/cursor 123"))
        .await?;
    assert!(resp.operations.is_empty());
    assert!(resp.next_page_token.is_none());
    Ok(())
}

#[tokio::test]
async fn cancel_batch_against_mock_server_handles_prefixed_and_unprefixed_name() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1beta/batches/batch-cancel-1:cancel"))
        .and(header("x-goog-api-key", "test-api-key"))
        .and(body_json(serde_json::json!({})))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
        .expect(2)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(format!("{}/v1beta", mock_server.uri()))
        .build();

    // 1. Prefixed
    client.cancel_batch("batches/batch-cancel-1").await?;

    // 2. Unprefixed
    client.cancel_batch("batch-cancel-1").await?;
    Ok(())
}

#[tokio::test]
async fn delete_batch_against_mock_server_handles_prefixed_and_unprefixed_name() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("DELETE"))
        .and(path("/v1beta/batches/batch-del-1"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
        .expect(2)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(format!("{}/v1beta", mock_server.uri()))
        .build();

    // 1. Prefixed
    client.delete_batch("batches/batch-del-1").await?;

    // 2. Unprefixed
    client.delete_batch("batch-del-1").await?;
    Ok(())
}

#[tokio::test]
async fn cancel_batch_maps_structured_gemini_error() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1beta/batches/batch-done:cancel"))
        .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
            "error": {
                "code": 400,
                "message": "Cannot cancel completed batch",
                "status": "FAILED_PRECONDITION"
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(format!("{}/v1beta", mock_server.uri()))
        .build();

    let err = client.cancel_batch("batch-done").await.unwrap_err();
    match err {
        crate::error::Error::VertexError(vertex_err) => {
            assert_eq!(vertex_err.code, 400);
            assert_eq!(vertex_err.status, "FAILED_PRECONDITION");
            assert!(vertex_err.message.contains("Cannot cancel completed batch"));
        }
        other => panic!("expected VertexError, got: {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn delete_batch_maps_structured_gemini_error() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("DELETE"))
        .and(path("/v1beta/batches/nonexistent"))
        .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
            "error": {
                "code": 404,
                "message": "Batch not found: batches/nonexistent",
                "status": "NOT_FOUND"
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(format!("{}/v1beta", mock_server.uri()))
        .build();

    let err = client.delete_batch("nonexistent").await.unwrap_err();
    match err {
        crate::error::Error::VertexError(vertex_err) => {
            assert_eq!(vertex_err.code, 404);
            assert_eq!(vertex_err.status, "NOT_FOUND");
        }
        other => panic!("expected VertexError, got: {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn batch_end_to_end_create_poll_cancel_delete_lifecycle() -> Result<()> {
    let mock_server = MockServer::start().await;

    // 1. Create batch
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-2.5-flash:batchGenerateContent"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "name": "batches/lifecycle-123",
            "metadata": {
                "name": "batches/lifecycle-123",
                "state": "BATCH_STATE_PENDING"
            },
            "done": false
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    // 2. Poll: running
    Mock::given(method("GET"))
        .and(path("/v1beta/batches/lifecycle-123"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "name": "batches/lifecycle-123",
            "metadata": {
                "name": "batches/lifecycle-123",
                "state": "BATCH_STATE_RUNNING"
            },
            "done": false
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    // 3. Cancel
    Mock::given(method("POST"))
        .and(path("/v1beta/batches/lifecycle-123:cancel"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
        .expect(1)
        .mount(&mock_server)
        .await;

    // 4. Delete
    Mock::given(method("DELETE"))
        .and(path("/v1beta/batches/lifecycle-123"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(format!("{}/v1beta", mock_server.uri()))
        .build();

    let request = BatchGenerateContentRequest::new(GenerateContentBatch::default());
    let op = client
        .batch_generate_content(&request, "gemini-2.5-flash")
        .await?;
    let batch_name = op.name.as_deref().expect("name");
    assert_eq!(op.state(), Some(BatchState::Pending));

    let polled = client.get_batch(batch_name).await?;
    assert_eq!(polled.state(), Some(BatchState::Running));

    client.cancel_batch(batch_name).await?;
    client.delete_batch(batch_name).await?;
    Ok(())
}

#[tokio::test]
async fn create_file_search_store_against_mock_server_normalizes_model_prefix() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/fileSearchStores"))
        .and(header("x-goog-api-key", "test-api-key"))
        .and(body_json(serde_json::json!({
            "displayName": "My Store",
            "embeddingModel": "models/gemini-embedding-2"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "name": "fileSearchStores/my-store-123",
            "displayName": "My Store",
            "embeddingModel": "models/gemini-embedding-2",
            "activeDocumentsCount": "0",
            "createTime": "2026-03-30T10:00:00Z"
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let store_req = FileSearchStore::builder()
        .name("fileSearchStores/ignored-id")
        .display_name("My Store")
        .embedding_model("gemini-embedding-2") // should be normalized to models/gemini-embedding-2
        .active_documents_count(10)
        .size_bytes(2048)
        .build();

    let created = client.create_file_search_store(&store_req).await?;
    assert_eq!(
        created.name.as_deref(),
        Some("fileSearchStores/my-store-123")
    );
    assert_eq!(created.display_name.as_deref(), Some("My Store"));
    assert_eq!(
        created.embedding_model.as_deref(),
        Some("models/gemini-embedding-2")
    );
    assert_eq!(created.active_documents_count, Some(0));
    Ok(())
}

#[tokio::test]
async fn get_file_search_store_against_mock_server_handles_prefixed_and_unprefixed_name()
-> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/fileSearchStores/store-xyz"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "name": "fileSearchStores/store-xyz",
            "displayName": "Found Store",
            "activeDocumentsCount": "10",
            "sizeBytes": "4096"
        })))
        .expect(2)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let store1 = client.get_file_search_store("store-xyz").await?;
    let store2 = client
        .get_file_search_store("fileSearchStores/store-xyz")
        .await?;
    assert_eq!(store1, store2);
    assert_eq!(store1.name.as_deref(), Some("fileSearchStores/store-xyz"));
    assert_eq!(store1.display_name.as_deref(), Some("Found Store"));
    assert_eq!(store1.active_documents_count, Some(10));
    assert_eq!(store1.size_bytes, Some(4096));
    Ok(())
}

#[tokio::test]
async fn get_file_search_store_maps_not_found_error() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/fileSearchStores/missing-store"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
            "error": {
                "code": 404,
                "message": "FileSearchStore missing-store not found",
                "status": "NOT_FOUND"
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let err = client
        .get_file_search_store("missing-store")
        .await
        .err()
        .ok_or("expected 404 error")?;

    match err {
        GeminiError::VertexError(vertex_err) => {
            assert_eq!(vertex_err.code, 404);
            assert_eq!(vertex_err.status, "NOT_FOUND");
        }
        other => panic!("expected VertexError, got {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn list_file_search_stores_against_mock_server_without_pagination_params() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/fileSearchStores"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "fileSearchStores": [
                {
                    "name": "fileSearchStores/store-1",
                    "displayName": "First Store"
                }
            ]
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let response = client.list_file_search_stores(None, None).await?;
    assert_eq!(response.file_search_stores.len(), 1);
    assert_eq!(
        response.file_search_stores[0].name.as_deref(),
        Some("fileSearchStores/store-1")
    );
    assert_eq!(response.next_page_token, None);
    Ok(())
}

#[tokio::test]
async fn list_file_search_stores_against_mock_server_with_pagination_params() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/fileSearchStores"))
        .and(query_param("pageSize", "10"))
        .and(query_param("pageToken", "next-tok=="))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "fileSearchStores": [
                {
                    "name": "fileSearchStores/store-2",
                    "displayName": "Second Store"
                }
            ],
            "nextPageToken": "page-2-tok=="
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let response = client
        .list_file_search_stores(Some(10), Some("next-tok=="))
        .await?;
    assert_eq!(response.file_search_stores.len(), 1);
    assert_eq!(
        response.file_search_stores[0].name.as_deref(),
        Some("fileSearchStores/store-2")
    );
    assert_eq!(response.next_page_token.as_deref(), Some("page-2-tok=="));
    Ok(())
}

#[tokio::test]
async fn delete_file_search_store_against_mock_server_handles_prefixed_and_unprefixed_name()
-> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("DELETE"))
        .and(path("/fileSearchStores/del-store-1"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
        .expect(2)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    client
        .delete_file_search_store("del-store-1", false)
        .await?;
    client
        .delete_file_search_store("fileSearchStores/del-store-1", false)
        .await?;
    Ok(())
}

#[tokio::test]
async fn delete_file_search_store_against_mock_server_with_force_true() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("DELETE"))
        .and(path("/fileSearchStores/force-del-store"))
        .and(query_param("force", "true"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    client
        .delete_file_search_store("force-del-store", true)
        .await?;
    Ok(())
}

#[tokio::test]
async fn delete_file_search_store_maps_structured_gemini_error() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("DELETE"))
        .and(path("/fileSearchStores/in-use-store"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
            "error": {
                "code": 400,
                "message": "FileSearchStore is not empty. Set force=true to delete.",
                "status": "FAILED_PRECONDITION"
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let err = client
        .delete_file_search_store("in-use-store", false)
        .await
        .err()
        .ok_or("expected 400 error")?;

    match err {
        GeminiError::VertexError(vertex_err) => {
            assert_eq!(vertex_err.code, 400);
            assert_eq!(vertex_err.status, "FAILED_PRECONDITION");
        }
        other => panic!("expected VertexError, got {other:?}"),
    }
    Ok(())
}

#[tokio::test]
async fn upload_to_file_search_store_executes_two_step_resumable_protocol() -> Result<()> {
    let mock_server = MockServer::start().await;
    let payload: &[u8] = b"Project Geologia Field Notes: Nepheline syenite formations.";
    let session_url = format!("{}/upload-session/store-doc-1", mock_server.uri());

    // Step 1: Start resumable upload against `/upload/fileSearchStores/my-store:uploadToFileSearchStore`
    Mock::given(method("POST"))
        .and(path(
            "/upload/fileSearchStores/my-store:uploadToFileSearchStore",
        ))
        .and(header("x-goog-api-key", "test-api-key"))
        .and(header("x-goog-upload-protocol", "resumable"))
        .and(header("x-goog-upload-command", "start"))
        .and(header(
            "x-goog-upload-header-content-length",
            payload.len().to_string().as_str(),
        ))
        .and(header("x-goog-upload-header-content-type", "text/plain"))
        .and(body_json(serde_json::json!({
            "displayName": "notes.txt",
            "mimeType": "text/plain"
        })))
        .respond_with(
            ResponseTemplate::new(200).insert_header("x-goog-upload-url", session_url.as_str()),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    // Step 2: Upload raw bytes and finalize
    Mock::given(method("POST"))
        .and(path("/upload-session/store-doc-1"))
        .and(header("content-length", payload.len().to_string().as_str()))
        .and(header("x-goog-upload-offset", "0"))
        .and(headers("x-goog-upload-command", vec!["upload", "finalize"]))
        .and(body_bytes(payload.to_vec()))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "name": "fileSearchStores/my-store/operations/upload-op-1",
            "done": false
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let op = client
        .upload_to_file_search_store("my-store", payload, Some("text/plain"), Some("notes.txt"))
        .await?;

    assert_eq!(
        op.name.as_deref(),
        Some("fileSearchStores/my-store/operations/upload-op-1")
    );
    assert!(!op.is_done());
    assert!(op.response.is_none());
    Ok(())
}

#[tokio::test]
async fn upload_to_file_search_store_with_request_supports_custom_config() -> Result<()> {
    let mock_server = MockServer::start().await;
    let payload: &[u8] = b"Custom chunking payload";
    let session_url = format!("{}/upload-session/custom-1", mock_server.uri());

    Mock::given(method("POST"))
        .and(path(
            "/upload/v1beta/fileSearchStores/store-99:uploadToFileSearchStore",
        ))
        .and(header("x-goog-api-key", "test-api-key"))
        .and(header("x-goog-upload-protocol", "resumable"))
        .and(header("x-goog-upload-command", "start"))
        .and(body_json(serde_json::json!({
            "displayName": "custom.txt",
            "mimeType": "text/plain",
            "customMetadata": [{ "key": "author", "stringValue": "Alice" }],
            "chunkingConfig": {
                "whiteSpaceConfig": {
                    "maxTokensPerChunk": 150,
                    "maxOverlapTokens": 15
                }
            }
        })))
        .respond_with(
            ResponseTemplate::new(200).insert_header("x-goog-upload-url", session_url.as_str()),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    Mock::given(method("POST"))
        .and(path("/upload-session/custom-1"))
        .and(headers("x-goog-upload-command", vec!["upload", "finalize"]))
        .and(body_bytes(payload.to_vec()))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "name": "fileSearchStores/store-99/operations/op-custom",
            "done": true,
            "response": {
                "parent": "fileSearchStores/store-99",
                "documentName": "fileSearchStores/store-99/documents/doc-custom"
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(format!("{}/v1beta", mock_server.uri()))
        .build();

    let request = UploadToFileSearchStoreRequest::builder()
        .display_name("custom.txt")
        .mime_type("text/plain")
        .custom_metadata(vec![CustomMetadata::string("author", "Alice")])
        .chunking_config(ChunkingConfig::white_space(150, 15))
        .build();

    let op = client
        .upload_to_file_search_store_with_request("fileSearchStores/store-99", payload, &request)
        .await?;

    assert!(op.is_done());
    assert_eq!(
        op.document_name().as_deref(),
        Some("fileSearchStores/store-99/documents/doc-custom")
    );
    let resp = op.response.as_ref().expect("response present");
    assert_eq!(resp.parent.as_deref(), Some("fileSearchStores/store-99"));
    assert_eq!(
        resp.document_name.as_deref(),
        Some("fileSearchStores/store-99/documents/doc-custom")
    );
    Ok(())
}

#[tokio::test]
async fn import_file_to_file_search_store_against_mock_server() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/fileSearchStores/target-store:importFile"))
        .and(header("x-goog-api-key", "test-api-key"))
        .and(body_json(serde_json::json!({
            "fileName": "files/source-file-123",
            "customMetadata": [{ "key": "source", "stringValue": "Files API" }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "name": "fileSearchStores/target-store/operations/import-op-1",
            "done": false
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let request = ImportFileRequest::new("source-file-123")
        .with_custom_metadata(vec![CustomMetadata::string("source", "Files API")]);

    let op = client
        .import_file_to_file_search_store("target-store", &request)
        .await?;

    assert_eq!(
        op.name.as_deref(),
        Some("fileSearchStores/target-store/operations/import-op-1")
    );
    assert!(!op.is_done());
    Ok(())
}

#[tokio::test]
async fn get_file_search_operation_against_mock_server() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/fileSearchStores/store-1/operations/op-1"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "name": "fileSearchStores/store-1/operations/op-1",
            "done": true,
            "response": {
                "parent": "fileSearchStores/store-1",
                "documentName": "fileSearchStores/store-1/documents/doc-1"
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let op = client
        .get_file_search_operation("fileSearchStores/store-1/operations/op-1")
        .await?;

    assert!(op.is_done());
    assert_eq!(
        op.document_name().as_deref(),
        Some("fileSearchStores/store-1/documents/doc-1")
    );
    let resp = op.response.as_ref().expect("response present");
    assert_eq!(resp.parent.as_deref(), Some("fileSearchStores/store-1"));
    assert_eq!(
        resp.document_name.as_deref(),
        Some("fileSearchStores/store-1/documents/doc-1")
    );
    Ok(())
}

#[tokio::test]
async fn get_document_against_mock_server() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/fileSearchStores/store-1/documents/doc-1"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "name": "fileSearchStores/store-1/documents/doc-1",
            "displayName": "queried-doc.txt",
            "mimeType": "text/plain",
            "sizeBytes": "512",
            "state": "STATE_ACTIVE"
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let doc = client
        .get_document("fileSearchStores/store-1/documents/doc-1")
        .await?;

    assert_eq!(
        doc.name.as_deref(),
        Some("fileSearchStores/store-1/documents/doc-1")
    );
    assert_eq!(doc.display_name.as_deref(), Some("queried-doc.txt"));
    assert_eq!(doc.state, Some(DocumentState::Active));
    assert_eq!(doc.size_bytes, Some(512));
    Ok(())
}

#[tokio::test]
async fn generate_content_image_generation_sends_modalities_and_decodes_inline_data() -> Result<()>
{
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/models/gemini-2.5-flash-image:generateContent"))
        .and(header("x-goog-api-key", "test-api-key"))
        .and(body_json(serde_json::json!({
            "contents": [{
                "role": "user",
                "parts": [{"text": "A tuxedo cat riding a rocket"}]
            }],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {
                    "aspectRatio": "1:1"
                }
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "candidates": [{
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [{
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": "aGVsbG8="
                        }
                    }]
                },
                "finishReason": "STOP"
            }]
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let request = GenerateContentRequest::builder()
        .contents(vec![
            Content::builder()
                .role(Role::User)
                .add_text_part("A tuxedo cat riding a rocket")
                .build(),
        ])
        .generation_config(
            GenerationConfig::builder()
                .response_modalities([Modality::Image])
                .image_config(ImageConfig::builder().aspect_ratio("1:1").build())
                .build(),
        )
        .build();

    let response = client
        .generate_content(&request, "gemini-2.5-flash-image")
        .await?;
    let candidate = response.candidates.first().ok_or("expected candidate")?;
    let (mime_type, bytes) = candidate
        .get_inline_data()
        .ok_or("expected inline image data")??;
    assert_eq!(mime_type, "image/png");
    assert_eq!(bytes, b"hello");
    Ok(())
}

#[tokio::test]
async fn create_decision_against_mock_server_evaluates_noul_choice_and_score_questions()
-> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/decisions"))
        .and(header("x-goog-api-key", "test-api-key"))
        .and(body_json(serde_json::json!({
            "model": "gemini-decision-1",
            "input": "Customer ticket: My headphones arrived with a cracked headband and the left speaker does not produce sound.",
            "questions": {
                "is_defective": {
                    "type": "noul",
                    "instructions": "Does the customer report a physical or functional product defect?",
                    "criteria": {
                        "true_description": "Customer reports physical damage or non-working hardware",
                        "false_description": "Customer inquiry is unrelated to hardware defects"
                    }
                },
                "department": {
                    "type": "choice",
                    "instructions": "Which support queue should handle this ticket?",
                    "criteria": {
                        "billing": "Payment, invoice, or subscription charges",
                        "warranty": "Damaged or defective hardware replacement",
                        "general": "General product questions"
                    }
                },
                "severity": {
                    "type": "score",
                    "instructions": "Rate the severity of the product issue.",
                    "criteria": [
                        "No functional impact",
                        "Minor cosmetic blemish",
                        "Product is unusable due to damage or failure"
                    ]
                }
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "answers": {
                "is_defective": {
                    "type": "noul",
                    "noul": 0.99,
                    "confidence": 0.99
                },
                "department": {
                    "type": "choice",
                    "choice": "warranty",
                    "confidence": 0.94,
                    "probabilities": {
                        "billing": 0.02,
                        "warranty": 0.94,
                        "general": 0.04
                    }
                },
                "severity": {
                    "type": "score",
                    "score": 1.92,
                    "confidence": 0.93,
                    "probabilities": {
                        "0": 0.01,
                        "1": 0.06,
                        "2": 0.93
                    },
                    "legend": {
                        "0": "No functional impact",
                        "1": "Minor cosmetic blemish",
                        "2": "Product is unusable due to damage or failure"
                    }
                }
            },
            "usage": {
                "initial_input_tokens": 24,
                "total_input_tokens": 115,
                "total_output_tokens": 12,
                "total_tokens": 127,
                "input_tokens_by_modality": [
                    {
                        "modality": "text",
                        "tokens": 115
                    }
                ]
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let request = CreateDecisionRequest::builder()
        .text_input("Customer ticket: My headphones arrived with a cracked headband and the left speaker does not produce sound.")
        .add_noul_question(
            "is_defective",
            NoulQuestion::new("Does the customer report a physical or functional product defect?")
                .true_description("Customer reports physical damage or non-working hardware")
                .false_description("Customer inquiry is unrelated to hardware defects"),
        )
        .add_choice_question(
            "department",
            ChoiceQuestion::new(
                "Which support queue should handle this ticket?",
                [
                    ("billing", "Payment, invoice, or subscription charges"),
                    ("warranty", "Damaged or defective hardware replacement"),
                    ("general", "General product questions"),
                ],
            ),
        )
        .add_score_question(
            "severity",
            ScoreQuestion::new(
                "Rate the severity of the product issue.",
                [
                    "No functional impact",
                    "Minor cosmetic blemish",
                    "Product is unusable due to damage or failure",
                ],
            ),
        )
        .build();

    let response = client.create_decision(&request).await?;

    let is_defective = response
        .get_noul("is_defective")
        .ok_or("expected is_defective noul answer")?;
    assert_eq!(is_defective.noul, 0.99);
    assert_eq!(is_defective.confidence, 0.99);

    let department = response
        .get_choice("department")
        .ok_or("expected department choice answer")?;
    assert_eq!(department.choice, "warranty");
    assert_eq!(department.confidence, 0.94);
    assert_eq!(department.probabilities.get("warranty"), Some(&0.94));

    let severity = response
        .get_score("severity")
        .ok_or("expected severity score answer")?;
    assert_eq!(severity.score, 1.92);
    assert_eq!(severity.confidence, 0.93);
    assert_eq!(
        severity.legend.get("2").map(String::as_str),
        Some("Product is unusable due to damage or failure")
    );

    let usage = response.usage.ok_or("expected usage")?;
    assert_eq!(usage.initial_input_tokens, Some(24));
    assert_eq!(usage.total_input_tokens, Some(115));
    assert_eq!(usage.total_output_tokens, Some(12));
    assert_eq!(usage.total_tokens, Some(127));
    Ok(())
}

#[tokio::test]
async fn create_decision_against_mock_server_with_multimodal_input() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/decisions"))
        .and(header("x-goog-api-key", "test-api-key"))
        .and(body_json(serde_json::json!({
            "model": "gemini-decision-1",
            "input": [
                {
                    "type": "text",
                    "text": "Review the storefront photo."
                },
                {
                    "type": "image",
                    "uri": "https://generativelanguage.googleapis.com/v1beta/files/store-photo-1",
                    "mime_type": "image/jpeg"
                }
            ],
            "questions": {
                "open_sign_lit": {
                    "type": "noul",
                    "instructions": "Is the neon OPEN sign illuminated?"
                }
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "answers": {
                "open_sign_lit": {
                    "type": "noul",
                    "noul": 0.91,
                    "confidence": 0.91
                }
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let request = CreateDecisionRequest::builder()
        .add_input_content(DecisionInputContent::text("Review the storefront photo."))
        .add_input_content(DecisionInputContent::image_uri(
            "https://generativelanguage.googleapis.com/v1beta/files/store-photo-1",
            "image/jpeg",
        ))
        .add_noul_question(
            "open_sign_lit",
            NoulQuestion::new("Is the neon OPEN sign illuminated?"),
        )
        .build();

    let response = client.create_decision(&request).await?;
    let ans = response
        .get_noul("open_sign_lit")
        .ok_or("expected open_sign_lit answer")?;
    assert_eq!(ans.noul, 0.91);
    assert_eq!(ans.confidence, 0.91);
    Ok(())
}

#[tokio::test]
async fn create_decision_against_mock_server_maps_structured_api_error() -> Result<()> {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/decisions"))
        .and(header("x-goog-api-key", "test-api-key"))
        .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
            "error": {
                "code": 400,
                "message": "questions map must contain at least one question",
                "status": "INVALID_ARGUMENT"
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = GeminiClient::builder("test-api-key")
        .base_url(mock_server.uri())
        .build();

    let request = CreateDecisionRequest::builder()
        .text_input("Empty questions test")
        .build();

    let err = client
        .create_decision(&request)
        .await
        .err()
        .ok_or("expected 400 error")?;

    match err {
        GeminiError::VertexError(vertex_err) => {
            assert_eq!(vertex_err.code, 400);
            assert_eq!(vertex_err.status, "INVALID_ARGUMENT");
            assert_eq!(
                vertex_err.message,
                "questions map must contain at least one question"
            );
        }
        other => panic!("expected VertexError, got {other:?}"),
    }
    Ok(())
}
