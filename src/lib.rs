//! Async Rust client for the Google Gemini API.
//!
//! This crate provides a high-level async client for interacting with Google's Gemini
//! generative AI models. It supports content generation (including streaming via SSE),
//! token counting, text embeddings, and image generation.
//!
//! # Usage
//!
//! ```no_run
//! use google_genai::prelude::*;
//!
//! # async fn run() -> google_genai::error::Result<()> {
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

mod client;
pub mod error;
pub mod network;
mod types;

/// Convenience re-exports of the most commonly used types.
///
/// Importing `use google_genai::prelude::*` brings [`GeminiClient`](crate::prelude::GeminiClient)
/// and all request/response types into scope.
pub mod prelude {
    pub use crate::client::*;
    pub use crate::types::*;
}
