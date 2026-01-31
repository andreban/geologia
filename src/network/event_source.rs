//! Server-Sent Events (SSE) decoder for streaming HTTP responses.
//!
//! Implements a [`tokio_util::codec::Decoder`] that parses an SSE byte stream into
//! [`ServerSentEvent`] values. Used internally by [`GeminiClient::stream_generate_content`]
//! to process chunked model responses.
//!
//! [`GeminiClient::stream_generate_content`]: crate::prelude::GeminiClient::stream_generate_content

use reqwest::Response;
use std::mem;
use tokio_stream::{Stream, StreamExt};
use tokio_util::{
    codec::{Decoder, FramedRead, LinesCodec, LinesCodecError},
    io::StreamReader,
};
use tracing::warn;

static EVENT: &str = "event: ";
static DATA: &str = "data: ";
static ID: &str = "id: ";
static RETRY: &str = "retry: ";

/// Extension trait for converting an HTTP response into a stream of [`ServerSentEvent`]s.
pub trait EventSource {
    /// Consumes the response and returns a stream of parsed SSE events.
    fn event_stream(self) -> impl Stream<Item = Result<ServerSentEvent, LinesCodecError>>;
}

impl EventSource for Response {
    fn event_stream(self) -> impl Stream<Item = Result<ServerSentEvent, LinesCodecError>> {
        stream_response(self)
    }
}

/// A parsed Server-Sent Event.
///
/// Fields correspond to the standard SSE fields: `event`, `data`, `id`, and `retry`.
/// Multiple `data:` lines within a single event are concatenated with newline separators.
#[derive(Debug, Default, Clone)]
pub struct ServerSentEvent {
    /// The event type (from the `event:` field).
    pub event: Option<String>,
    /// The event payload (from one or more `data:` fields, joined by `\n`).
    pub data: Option<String>,
    /// The event ID (from the `id:` field).
    pub id: Option<String>,
    /// The reconnection time in milliseconds (from the `retry:` field).
    pub retry: Option<usize>,
}

/// A [`Decoder`] that parses a byte stream of SSE-formatted data into [`ServerSentEvent`]s.
///
/// Wraps a [`LinesCodec`] and accumulates fields until an empty line signals the end of an event.
pub struct ServerSentEventsCodec {
    lines_code: LinesCodec,
    next: ServerSentEvent,
}

impl Default for ServerSentEventsCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl ServerSentEventsCodec {
    /// Creates a new SSE codec.
    pub fn new() -> Self {
        Self {
            lines_code: LinesCodec::new(),
            next: Default::default(),
        }
    }
}

impl Decoder for ServerSentEventsCodec {
    type Item = ServerSentEvent;
    type Error = LinesCodecError;
    fn decode(
        &mut self,
        src: &mut tokio_util::bytes::BytesMut,
    ) -> Result<Option<Self::Item>, Self::Error> {
        let res = self.lines_code.decode(src)?;

        let Some(mut line) = res else {
            return Ok(None);
        };

        if line.is_empty() {
            let result = mem::take(&mut self.next);
            return Ok(Some(result));
        }

        if line.starts_with(EVENT) {
            line.drain(..EVENT.len());
            self.next.event = Some(line);
        } else if line.starts_with(DATA) {
            line.drain(..DATA.len());
            if let Some(ref mut existing) = self.next.data {
                existing.push('\n');
                existing.push_str(&line);
            } else {
                self.next.data = Some(line);
            }
        } else if line.starts_with(ID) {
            line.drain(..ID.len());
            self.next.id = Some(line);
        } else if line.starts_with(RETRY) {
            line.drain(..RETRY.len());
            let Ok(retry) = line.parse() else {
                warn!(line, "Received invalid retry value");
                return Ok(None);
            };
            self.next.retry = Some(retry);
        }

        Ok(None)
    }
}

/// Converts a [`Response`] into a stream of [`ServerSentEvent`]s.
///
/// The response body is read as a byte stream and decoded using [`ServerSentEventsCodec`].
pub fn stream_response(
    response: Response,
) -> impl Stream<Item = Result<ServerSentEvent, LinesCodecError>> {
    let bytes_stream = response.bytes_stream();
    let body_reader = StreamReader::new(bytes_stream.map(|res| res.map_err(std::io::Error::other)));
    FramedRead::new(body_reader, ServerSentEventsCodec::new())
}
