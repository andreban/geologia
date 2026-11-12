// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Multimodal input content, media processing, and text annotation types for the Decisions API.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::types::common::encode_base64;

/// Shared input context for a [`CreateDecisionRequest`](super::CreateDecisionRequest).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum DecisionInput {
    /// A plain text input string.
    Text(String),
    /// A single multimodal content block.
    Content(DecisionInputContent),
    /// A list of multimodal content blocks.
    Contents(Vec<DecisionInputContent>),
}

impl DecisionInput {
    /// Validates that the decision input is non-empty and all multimodal blocks are well-formed.
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Text(text) => {
                if text.trim().is_empty() {
                    return Err(Error::InvalidArgument {
                        message: "decision request text `input` must not be empty".into(),
                    });
                }
                Ok(())
            }
            Self::Content(block) => block.validate(),
            Self::Contents(blocks) => {
                if blocks.is_empty() {
                    return Err(Error::InvalidArgument {
                        message: "decision request `input` content list must not be empty".into(),
                    });
                }
                for block in blocks {
                    block.validate()?;
                }
                Ok(())
            }
        }
    }
}

impl From<String> for DecisionInput {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for DecisionInput {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<DecisionInputContent> for DecisionInput {
    fn from(value: DecisionInputContent) -> Self {
        Self::Content(value)
    }
}

impl From<Vec<DecisionInputContent>> for DecisionInput {
    fn from(value: Vec<DecisionInputContent>) -> Self {
        Self::Contents(value)
    }
}

/// A content block within [`DecisionInput`].
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionInputContent {
    /// A text content block (`"type": "text"`).
    Text(DecisionTextContent),
    /// An image content block (`"type": "image"`).
    Image(DecisionImageContent),
    /// An audio content block (`"type": "audio"`).
    Audio(DecisionAudioContent),
    /// A video content block (`"type": "video"`).
    Video(DecisionVideoContent),
    /// A document content block (`"type": "document"`).
    Document(DecisionDocumentContent),
}

impl DecisionInputContent {
    /// Creates a text content block.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(DecisionTextContent::new(text))
    }

    /// Creates an image content block referencing a URI (such as a Files API URI).
    pub fn image_uri(uri: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self::Image(DecisionImageContent::from_uri(uri, mime_type))
    }

    /// Creates an inline base64-encoded image content block from raw bytes.
    pub fn image_bytes(bytes: impl AsRef<[u8]>, mime_type: impl Into<String>) -> Self {
        Self::Image(DecisionImageContent::from_bytes(bytes, mime_type))
    }

    /// Creates an audio content block referencing a URI.
    pub fn audio_uri(uri: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self::Audio(DecisionAudioContent::from_uri(uri, mime_type))
    }

    /// Creates an inline base64-encoded audio content block from raw bytes.
    pub fn audio_bytes(bytes: impl AsRef<[u8]>, mime_type: impl Into<String>) -> Self {
        Self::Audio(DecisionAudioContent::from_bytes(bytes, mime_type))
    }

    /// Creates a video content block referencing a URI.
    pub fn video_uri(uri: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self::Video(DecisionVideoContent::from_uri(uri, mime_type))
    }

    /// Creates an inline base64-encoded video content block from raw bytes.
    pub fn video_bytes(bytes: impl AsRef<[u8]>, mime_type: impl Into<String>) -> Self {
        Self::Video(DecisionVideoContent::from_bytes(bytes, mime_type))
    }

    /// Creates a document content block referencing a URI.
    pub fn document_uri(uri: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self::Document(DecisionDocumentContent::from_uri(uri, mime_type))
    }

    /// Creates an inline base64-encoded document content block from raw bytes.
    pub fn document_bytes(bytes: impl AsRef<[u8]>, mime_type: impl Into<String>) -> Self {
        Self::Document(DecisionDocumentContent::from_bytes(bytes, mime_type))
    }

    /// Validates that the content block is non-empty and media blocks specify exactly one
    /// source (`data` or `uri`) and a non-empty `mime_type`.
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Text(text) => {
                if text.text.trim().is_empty() {
                    return Err(Error::InvalidArgument {
                        message: "decision text content block must not have empty `text`".into(),
                    });
                }
                Ok(())
            }
            Self::Image(img) => validate_media_source(
                "image",
                img.data.as_deref(),
                img.uri.as_deref(),
                img.mime_type.as_deref(),
            ),
            Self::Audio(aud) => validate_media_source(
                "audio",
                aud.data.as_deref(),
                aud.uri.as_deref(),
                aud.mime_type.as_deref(),
            ),
            Self::Video(vid) => validate_media_source(
                "video",
                vid.data.as_deref(),
                vid.uri.as_deref(),
                vid.mime_type.as_deref(),
            ),
            Self::Document(doc) => validate_media_source(
                "document",
                doc.data.as_deref(),
                doc.uri.as_deref(),
                doc.mime_type.as_deref(),
            ),
        }
    }
}

fn validate_media_source(
    kind: &str,
    data: Option<&str>,
    uri: Option<&str>,
    mime_type: Option<&str>,
) -> Result<()> {
    let has_data = data.is_some_and(|d| !d.trim().is_empty());
    let has_uri = uri.is_some_and(|u| !u.trim().is_empty());
    match (has_data, has_uri) {
        (true, false) | (false, true) => {}
        (false, false) => {
            return Err(Error::InvalidArgument {
                message: format!(
                    "decision {kind} content block must specify either `data` or `uri`"
                ),
            });
        }
        (true, true) => {
            return Err(Error::InvalidArgument {
                message: format!(
                    "decision {kind} content block cannot specify both `data` and `uri`"
                ),
            });
        }
    }
    if !mime_type.is_some_and(|m| !m.trim().is_empty()) {
        return Err(Error::InvalidArgument {
            message: format!("decision {kind} content block requires a non-empty `mime_type`"),
        });
    }
    Ok(())
}

/// A text content block.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DecisionTextContent {
    /// The text content.
    pub text: String,
    /// Optional citation annotations for the text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Vec<DecisionAnnotation>>,
}

impl DecisionTextContent {
    /// Creates a new [`DecisionTextContent`] with the given text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            annotations: None,
        }
    }

    /// Attaches annotations to this text content block.
    pub fn with_annotations(mut self, annotations: Vec<DecisionAnnotation>) -> Self {
        self.annotations = Some(annotations);
        self
    }
}

/// An image content block.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DecisionImageContent {
    /// Base64-encoded image bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    /// The URI of the image.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    /// The MIME type of the image (for example, `"image/png"`, `"image/jpeg"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// The media resolution to use when processing the image.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<DecisionMediaResolution>,
}

impl DecisionImageContent {
    /// Creates an image content block referencing a URI.
    pub fn from_uri(uri: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self {
            data: None,
            uri: Some(uri.into()),
            mime_type: Some(mime_type.into()),
            resolution: None,
        }
    }

    /// Creates an inline base64-encoded image content block from raw bytes.
    pub fn from_bytes(bytes: impl AsRef<[u8]>, mime_type: impl Into<String>) -> Self {
        Self {
            data: Some(encode_base64(bytes.as_ref())),
            uri: None,
            mime_type: Some(mime_type.into()),
            resolution: None,
        }
    }

    /// Sets the media resolution for this image.
    pub fn with_resolution(mut self, resolution: DecisionMediaResolution) -> Self {
        self.resolution = Some(resolution);
        self
    }
}

/// An audio content block.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DecisionAudioContent {
    /// Base64-encoded audio bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    /// The URI of the audio.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    /// The MIME type of the audio (for example, `"audio/wav"`, `"audio/mp3"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// The number of audio channels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channels: Option<i32>,
    /// The sample rate of the audio in Hz.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<i32>,
}

impl DecisionAudioContent {
    /// Creates an audio content block referencing a URI.
    pub fn from_uri(uri: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self {
            data: None,
            uri: Some(uri.into()),
            mime_type: Some(mime_type.into()),
            channels: None,
            sample_rate: None,
        }
    }

    /// Creates an inline base64-encoded audio content block from raw bytes.
    pub fn from_bytes(bytes: impl AsRef<[u8]>, mime_type: impl Into<String>) -> Self {
        Self {
            data: Some(encode_base64(bytes.as_ref())),
            uri: None,
            mime_type: Some(mime_type.into()),
            channels: None,
            sample_rate: None,
        }
    }

    /// Sets the number of audio channels.
    pub fn with_channels(mut self, channels: i32) -> Self {
        self.channels = Some(channels);
        self
    }

    /// Sets the audio sample rate in Hz.
    pub fn with_sample_rate(mut self, sample_rate: i32) -> Self {
        self.sample_rate = Some(sample_rate);
        self
    }
}

/// A video content block.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DecisionVideoContent {
    /// Base64-encoded video bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    /// The URI of the video.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    /// The MIME type of the video (for example, `"video/mp4"`, `"video/webm"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Optional user-defined name for this video content block.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// How the model processes this video.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processing: Option<DecisionVideoProcessing>,
    /// The media resolution to use when processing the video.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<DecisionMediaResolution>,
}

impl DecisionVideoContent {
    /// Creates a video content block referencing a URI.
    pub fn from_uri(uri: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self {
            data: None,
            uri: Some(uri.into()),
            mime_type: Some(mime_type.into()),
            name: None,
            processing: None,
            resolution: None,
        }
    }

    /// Creates an inline base64-encoded video content block from raw bytes.
    pub fn from_bytes(bytes: impl AsRef<[u8]>, mime_type: impl Into<String>) -> Self {
        Self {
            data: Some(encode_base64(bytes.as_ref())),
            uri: None,
            mime_type: Some(mime_type.into()),
            name: None,
            processing: None,
            resolution: None,
        }
    }

    /// Sets a user-defined name for this video content block.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the video processing configuration.
    pub fn with_processing(mut self, processing: DecisionVideoProcessing) -> Self {
        self.processing = Some(processing);
        self
    }

    /// Sets the media resolution for this video.
    pub fn with_resolution(mut self, resolution: DecisionMediaResolution) -> Self {
        self.resolution = Some(resolution);
        self
    }
}

/// A document content block.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DecisionDocumentContent {
    /// Base64-encoded document bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    /// The URI of the document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    /// The MIME type of the document (for example, `"application/pdf"`, `"text/csv"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

impl DecisionDocumentContent {
    /// Creates a document content block referencing a URI.
    pub fn from_uri(uri: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self {
            data: None,
            uri: Some(uri.into()),
            mime_type: Some(mime_type.into()),
        }
    }

    /// Creates an inline base64-encoded document content block from raw bytes.
    pub fn from_bytes(bytes: impl AsRef<[u8]>, mime_type: impl Into<String>) -> Self {
        Self {
            data: Some(encode_base64(bytes.as_ref())),
            uri: None,
            mime_type: Some(mime_type.into()),
        }
    }
}

/// Media resolution level for image or video content blocks in the Decisions API.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecisionMediaResolution {
    /// Low resolution (`"low"`).
    Low,
    /// Medium resolution (`"medium"`).
    Medium,
    /// High resolution (`"high"`).
    High,
    /// Ultra-high resolution (`"ultra_high"`).
    UltraHigh,
}

/// Media processing mode for video content blocks.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecisionMediaProcessingMode {
    /// Process media statically before the model runs (`"static"`).
    Static,
    /// Let the model inspect media agentically during execution (`"agentic"`).
    Agentic,
}

/// Video processing configuration for [`DecisionVideoContent`].
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum DecisionVideoProcessing {
    /// Processing mode specified as a string (`"static"` or `"agentic"`).
    Mode(DecisionMediaProcessingMode),
    /// Detailed static media processing configuration (`{"type": "static", ...}`).
    StaticConfig(StaticMediaProcessing),
}

/// Discriminator tag for [`StaticMediaProcessing`].
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StaticMediaProcessingType {
    /// The `"static"` media processing type.
    #[default]
    #[serde(rename = "static")]
    Static,
}

/// Configuration for processing video statically before the model runs.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct StaticMediaProcessing {
    /// Discriminator tag (always `"static"`).
    #[serde(rename = "type")]
    pub kind: StaticMediaProcessingType,
    /// Optional start offset for clipping the video (for example, `"1.5s"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_offset: Option<String>,
    /// Optional end offset for clipping the video (for example, `"10s"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_offset: Option<String>,
    /// Optional frame sampling rate in frames per second.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<f64>,
}

impl StaticMediaProcessing {
    /// Creates a new [`StaticMediaProcessing`] configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the clip start offset (for example, `"1.5s"`).
    pub fn with_start_offset(mut self, start_offset: impl Into<String>) -> Self {
        self.start_offset = Some(start_offset.into());
        self
    }

    /// Sets the clip end offset (for example, `"10s"`).
    pub fn with_end_offset(mut self, end_offset: impl Into<String>) -> Self {
        self.end_offset = Some(end_offset.into());
        self
    }

    /// Sets the frame sampling rate in frames per second.
    pub fn with_fps(mut self, fps: f64) -> Self {
        self.fps = Some(fps);
        self
    }
}

/// Annotation attached to a [`DecisionTextContent`] block.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionAnnotation {
    /// File citation annotation (`"type": "file_citation"`).
    FileCitation(DecisionFileCitation),
    /// Place citation annotation (`"type": "place_citation"`).
    PlaceCitation(DecisionPlaceCitation),
    /// Speech metadata annotation (`"type": "speech_metadata"`).
    SpeechMetadata(DecisionSpeechAnnotation),
    /// URL citation annotation (`"type": "url_citation"`).
    UrlCitation(DecisionUrlCitation),
    /// Word timing annotation (`"type": "word_info"`).
    WordInfo(DecisionWordInfo),
}

/// File citation metadata for a segment of text.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct DecisionFileCitation {
    /// Start of the attributed segment in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_index: Option<i32>,
    /// End of the attributed segment in bytes (exclusive).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_index: Option<i32>,
    /// Name of the cited file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_name: Option<String>,
    /// Source of the cited file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Document URI of the cited file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
}

/// Google Maps place citation metadata for a segment of text.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct DecisionPlaceCitation {
    /// Start of the attributed segment in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_index: Option<i32>,
    /// End of the attributed segment in bytes (exclusive).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_index: Option<i32>,
    /// Title or name of the place.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The ID of the place in Google Maps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub place_id: Option<String>,
    /// Snippets of reviews that were used to generate the ground truth.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub review_snippets: Option<Vec<DecisionReviewSnippet>>,
    /// URI that resolves to the place on Google Maps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// Review snippet associated with a [`DecisionPlaceCitation`].
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct DecisionReviewSnippet {
    /// The ID of the review.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub review_id: Option<String>,
    /// Title of the review.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// URL to the user review on Google Maps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// Speech metadata annotation for a segment of text.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct DecisionSpeechAnnotation {
    /// Start of the attributed segment in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_index: Option<i32>,
    /// End of the attributed segment in bytes (exclusive).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_index: Option<i32>,
    /// Language code (BCP-47) of the speech (for example, `"en-US"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Speaker identifier or name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    /// Word-level timing details for the speech segment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<DecisionWordInfo>>,
}

/// URL citation metadata for a segment of text.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct DecisionUrlCitation {
    /// Start of the attributed segment in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_index: Option<i32>,
    /// End of the attributed segment in bytes (exclusive).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_index: Option<i32>,
    /// Title of the cited web page.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// URL of the cited web page.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// Word-level timing metadata for speech.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct DecisionWordInfo {
    /// Start of the attributed segment in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_index: Option<i32>,
    /// End of the attributed segment in bytes (exclusive).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_index: Option<i32>,
    /// Start time offset of the word in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_offset: Option<f64>,
    /// End time offset of the word in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_offset: Option<f64>,
    /// The transcribed word text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub word: Option<String>,
}
