// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Async Rust client for the Google Gemini API.
//!
//! This crate provides a high-level async client for interacting with Google's Gemini
//! generative AI models. It supports model discovery, content generation (including streaming
//! via SSE), token counting, text embeddings, image generation, context caching, media file
//! management (Files API), batch generation (Batch API), File Search stores for semantic
//! retrieval and grounding, and structured decision evaluation (Decisions API).
//!
//! # Usage
//!
//! ```no_run
//! use geologia::prelude::*;
//!
//! # async fn run() -> geologia::error::Result<()> {
//! let client = GeminiClient::new("YOUR_API_KEY".into());
//!
//! let request = GenerateContentRequest::builder()
//!     .contents(vec![
//!         Content::builder().add_text_part("Hello, Gemini!").build()
//!     ])
//!     .build();
//!
//! let response = client.generate_content(&request, "gemini-2.0-flash").await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Advanced configuration
//!
//! Use [`GeminiClient::builder`](crate::prelude::GeminiClient::builder) to configure a custom base URL (e.g. for proxies,
//! local mock servers, or offline testing) or to supply a custom [`reqwest::Client`]:
//!
//! ```no_run
//! use geologia::prelude::*;
//!
//! let client = GeminiClient::builder("YOUR_API_KEY")
//!     .base_url("https://my-proxy.example.com/v1beta")
//!     .build();
//! ```

mod client;
pub mod error;
pub mod network;
mod types;

/// Convenience re-exports of the most commonly used types.
///
/// Importing `use geologia::prelude::*` brings [`GeminiClient`](crate::prelude::GeminiClient),
/// [`GeminiClientBuilder`](crate::prelude::GeminiClientBuilder), and all request/response types into scope.
pub mod prelude {
    pub use crate::client::*;
    #[allow(deprecated)]
    pub use crate::types::*;
}
