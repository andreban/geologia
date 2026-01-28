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

pub trait EventSource {
    fn event_stream(self) -> impl Stream<Item = Result<ServerSentEvent, LinesCodecError>>;
}

impl EventSource for Response {
    fn event_stream(self) -> impl Stream<Item = Result<ServerSentEvent, LinesCodecError>> {
        stream_response(self)
    }
}

#[derive(Debug, Default, Clone)]
pub struct ServerSentEvent {
    pub event: Option<String>,
    pub data: Option<String>,
    pub id: Option<String>,
    pub retry: Option<usize>,
}

pub struct ServerSentEventsCodec {
    lines_code: LinesCodec,
    next: ServerSentEvent,
}

impl ServerSentEventsCodec {
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
            self.next.data = Some(line)
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

pub fn stream_response(
    response: Response,
) -> impl Stream<Item = Result<ServerSentEvent, LinesCodecError>> {
    let bytes_stream = response.bytes_stream();
    let body_reader = StreamReader::new(bytes_stream.map(|res| res.map_err(std::io::Error::other)));
    FramedRead::new(body_reader, ServerSentEventsCodec::new())
}
