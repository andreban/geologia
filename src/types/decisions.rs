// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Types for the Gemini Decisions API (`POST /v1beta/decisions`).
//!
//! The Decisions API evaluates one or more structured decision questions (`noul`,
//! `choice`, or `score`) against a shared text or multimodal input context using
//! a dedicated decision model such as [`DEFAULT_DECISION_MODEL`] (`"gemini-decision-1"`),
//! returning calibrated probabilities and closed-world confidence scores.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::common::encode_base64;

/// The default model identifier for the Gemini Decisions API.
pub const DEFAULT_DECISION_MODEL: &str = "gemini-decision-1";

// ============================================================================
// Request Types
// ============================================================================

/// Request body for `POST /v1beta/decisions` (`decisions.create`).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CreateDecisionRequest {
    /// The name of the model to use for evaluating the decisions
    /// (for example, `"gemini-decision-1"`).
    pub model: String,
    /// Shared input prefix evaluated once across all decision questions.
    pub input: DecisionInput,
    /// The decision questions to evaluate, keyed by caller-defined unique
    /// question identifiers (between 1 and 1,000 questions).
    pub questions: HashMap<String, Question>,
}

impl CreateDecisionRequest {
    /// Creates a new [`CreateDecisionRequestBuilder`].
    pub fn builder() -> CreateDecisionRequestBuilder {
        CreateDecisionRequestBuilder::default()
    }
}

/// Builder for [`CreateDecisionRequest`].
#[derive(Default, Debug, Clone)]
pub struct CreateDecisionRequestBuilder {
    model: Option<String>,
    input: Option<DecisionInput>,
    questions: HashMap<String, Question>,
}

impl CreateDecisionRequestBuilder {
    /// Sets the decision model name (defaults to [`DEFAULT_DECISION_MODEL`] if omitted).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the shared input context from any type convertible into [`DecisionInput`].
    pub fn input(mut self, input: impl Into<DecisionInput>) -> Self {
        self.input = Some(input.into());
        self
    }

    /// Sets the shared input context to a plain text string.
    pub fn text_input(mut self, text: impl Into<String>) -> Self {
        self.input = Some(DecisionInput::Text(text.into()));
        self
    }

    /// Appends a multimodal [`DecisionInputContent`] block to the request input.
    ///
    /// If the builder currently holds a plain text input or a single content block,
    /// it is converted into a sequence of content blocks (`DecisionInput::Contents`).
    pub fn add_input_content(mut self, content: DecisionInputContent) -> Self {
        self.input = Some(match self.input.take() {
            None => DecisionInput::Contents(vec![content]),
            Some(DecisionInput::Text(text)) => {
                DecisionInput::Contents(vec![DecisionInputContent::text(text), content])
            }
            Some(DecisionInput::Content(existing)) => {
                DecisionInput::Contents(vec![existing, content])
            }
            Some(DecisionInput::Contents(mut items)) => {
                items.push(content);
                DecisionInput::Contents(items)
            }
        });
        self
    }

    /// Replaces the entire questions map.
    pub fn questions<I, K>(mut self, questions: I) -> Self
    where
        I: IntoIterator<Item = (K, Question)>,
        K: Into<String>,
    {
        self.questions = questions.into_iter().map(|(k, v)| (k.into(), v)).collect();
        self
    }

    /// Inserts a single decision question under the given identifier.
    pub fn add_question(mut self, id: impl Into<String>, question: impl Into<Question>) -> Self {
        self.questions.insert(id.into(), question.into());
        self
    }

    /// Inserts a binary [`NoulQuestion`] under the given identifier.
    pub fn add_noul_question(mut self, id: impl Into<String>, question: NoulQuestion) -> Self {
        self.questions.insert(id.into(), Question::Noul(question));
        self
    }

    /// Inserts a categorical [`ChoiceQuestion`] under the given identifier.
    pub fn add_choice_question(mut self, id: impl Into<String>, question: ChoiceQuestion) -> Self {
        self.questions.insert(id.into(), Question::Choice(question));
        self
    }

    /// Inserts a quantitative [`ScoreQuestion`] under the given identifier.
    pub fn add_score_question(mut self, id: impl Into<String>, question: ScoreQuestion) -> Self {
        self.questions.insert(id.into(), Question::Score(question));
        self
    }

    /// Builds the [`CreateDecisionRequest`].
    ///
    /// Defaults `model` to [`DEFAULT_DECISION_MODEL`] (`"gemini-decision-1"`) and
    /// `input` to an empty text string if not explicitly configured.
    pub fn build(self) -> CreateDecisionRequest {
        CreateDecisionRequest {
            model: self
                .model
                .unwrap_or_else(|| DEFAULT_DECISION_MODEL.to_string()),
            input: self
                .input
                .unwrap_or_else(|| DecisionInput::Text(String::new())),
            questions: self.questions,
        }
    }
}

// ============================================================================
// Input Content & Multimodal Types
// ============================================================================

/// Shared input context for a [`CreateDecisionRequest`].
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
        Self::Image(DecisionImageContent {
            data: None,
            uri: Some(uri.into()),
            mime_type: Some(mime_type.into()),
            resolution: None,
        })
    }

    /// Creates an inline base64-encoded image content block from raw bytes.
    pub fn image_bytes(bytes: impl AsRef<[u8]>, mime_type: impl Into<String>) -> Self {
        Self::Image(DecisionImageContent {
            data: Some(encode_base64(bytes.as_ref())),
            uri: None,
            mime_type: Some(mime_type.into()),
            resolution: None,
        })
    }

    /// Creates an audio content block referencing a URI.
    pub fn audio_uri(uri: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self::Audio(DecisionAudioContent {
            data: None,
            uri: Some(uri.into()),
            mime_type: Some(mime_type.into()),
            channels: None,
            sample_rate: None,
        })
    }

    /// Creates an inline base64-encoded audio content block from raw bytes.
    pub fn audio_bytes(bytes: impl AsRef<[u8]>, mime_type: impl Into<String>) -> Self {
        Self::Audio(DecisionAudioContent {
            data: Some(encode_base64(bytes.as_ref())),
            uri: None,
            mime_type: Some(mime_type.into()),
            channels: None,
            sample_rate: None,
        })
    }

    /// Creates a video content block referencing a URI.
    pub fn video_uri(uri: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self::Video(DecisionVideoContent {
            data: None,
            uri: Some(uri.into()),
            mime_type: Some(mime_type.into()),
            name: None,
            processing: None,
            resolution: None,
        })
    }

    /// Creates an inline base64-encoded video content block from raw bytes.
    pub fn video_bytes(bytes: impl AsRef<[u8]>, mime_type: impl Into<String>) -> Self {
        Self::Video(DecisionVideoContent {
            data: Some(encode_base64(bytes.as_ref())),
            uri: None,
            mime_type: Some(mime_type.into()),
            name: None,
            processing: None,
            resolution: None,
        })
    }

    /// Creates a document content block referencing a URI.
    pub fn document_uri(uri: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self::Document(DecisionDocumentContent {
            data: None,
            uri: Some(uri.into()),
            mime_type: Some(mime_type.into()),
        })
    }

    /// Creates an inline base64-encoded document content block from raw bytes.
    pub fn document_bytes(bytes: impl AsRef<[u8]>, mime_type: impl Into<String>) -> Self {
        Self::Document(DecisionDocumentContent {
            data: Some(encode_base64(bytes.as_ref())),
            uri: None,
            mime_type: Some(mime_type.into()),
        })
    }
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
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
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

/// An audio content block.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
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

/// A video content block.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
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

/// A document content block.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
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

// ============================================================================
// Question & Instruction Types
// ============================================================================

/// A text content block used inside [`DecisionInstructions`].
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionInstructionContent {
    /// A text instruction block (`"type": "text"`).
    Text(DecisionTextContent),
}

impl DecisionInstructionContent {
    /// Creates a new text instruction block.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(DecisionTextContent::new(text))
    }
}

/// Instructions for a decision question (`NoulQuestion`, `ChoiceQuestion`, or `ScoreQuestion`).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum DecisionInstructions {
    /// Plain text instructions string.
    Text(String),
    /// A single structured text instruction block.
    Content(DecisionInstructionContent),
    /// A sequence of structured text instruction blocks.
    Contents(Vec<DecisionInstructionContent>),
}

impl From<String> for DecisionInstructions {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for DecisionInstructions {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<DecisionInstructionContent> for DecisionInstructions {
    fn from(value: DecisionInstructionContent) -> Self {
        Self::Content(value)
    }
}

impl From<Vec<DecisionInstructionContent>> for DecisionInstructions {
    fn from(value: Vec<DecisionInstructionContent>) -> Self {
        Self::Contents(value)
    }
}

/// A decision question to evaluate against the shared request input.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Question {
    /// A binary (`true` / `false`) decision question (`"type": "noul"`).
    Noul(NoulQuestion),
    /// A single-select categorical decision question (`"type": "choice"`).
    Choice(ChoiceQuestion),
    /// A quantitative rubric scoring question (`"type": "score"`).
    Score(ScoreQuestion),
}

impl Question {
    /// Creates a [`Question::Noul`] with the given instructions.
    pub fn noul(instructions: impl Into<DecisionInstructions>) -> Self {
        Self::Noul(NoulQuestion::new(instructions))
    }

    /// Creates a [`Question::Choice`] with the given instructions and option criteria.
    pub fn choice<I, K, V>(instructions: impl Into<DecisionInstructions>, criteria: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        Self::Choice(ChoiceQuestion::new(instructions, criteria))
    }

    /// Creates a [`Question::Score`] with the given instructions and rubric level descriptions.
    pub fn score<I, S>(instructions: impl Into<DecisionInstructions>, criteria: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::Score(ScoreQuestion::new(instructions, criteria))
    }
}

impl From<NoulQuestion> for Question {
    fn from(value: NoulQuestion) -> Self {
        Self::Noul(value)
    }
}

impl From<ChoiceQuestion> for Question {
    fn from(value: ChoiceQuestion) -> Self {
        Self::Choice(value)
    }
}

impl From<ScoreQuestion> for Question {
    fn from(value: ScoreQuestion) -> Self {
        Self::Score(value)
    }
}

/// A binary (`true` / `false`) decision question (`"type": "noul"`).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NoulQuestion {
    /// The question or evaluation instructions.
    pub instructions: DecisionInstructions,
    /// Optional descriptions or rubric criteria for the `true` and `false` outcomes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub criteria: Option<NoulCriteria>,
}

impl NoulQuestion {
    /// Creates a new [`NoulQuestion`] with the given instructions and no explicit outcome criteria.
    pub fn new(instructions: impl Into<DecisionInstructions>) -> Self {
        Self {
            instructions: instructions.into(),
            criteria: None,
        }
    }

    /// Creates a new [`NoulQuestion`] with explicit [`NoulCriteria`].
    pub fn with_criteria(
        instructions: impl Into<DecisionInstructions>,
        criteria: NoulCriteria,
    ) -> Self {
        Self {
            instructions: instructions.into(),
            criteria: Some(criteria),
        }
    }

    /// Sets the description for the `true` outcome.
    pub fn true_description(mut self, description: impl Into<String>) -> Self {
        self.criteria
            .get_or_insert_with(NoulCriteria::default)
            .true_description = Some(description.into());
        self
    }

    /// Sets the description for the `false` outcome.
    pub fn false_description(mut self, description: impl Into<String>) -> Self {
        self.criteria
            .get_or_insert_with(NoulCriteria::default)
            .false_description = Some(description.into());
        self
    }
}

/// Optional criteria describing the `true` and `false` outcomes of a [`NoulQuestion`].
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct NoulCriteria {
    /// Criteria or description for the `true` outcome.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub true_description: Option<String>,
    /// Criteria or description for the `false` outcome.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub false_description: Option<String>,
}

impl NoulCriteria {
    /// Creates an empty [`NoulCriteria`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the description for the `true` outcome.
    pub fn with_true_description(mut self, description: impl Into<String>) -> Self {
        self.true_description = Some(description.into());
        self
    }

    /// Sets the description for the `false` outcome.
    pub fn with_false_description(mut self, description: impl Into<String>) -> Self {
        self.false_description = Some(description.into());
        self
    }
}

/// A single-select categorical decision question (`"type": "choice"`) over mutually exclusive options.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ChoiceQuestion {
    /// The question or classification instructions.
    pub instructions: DecisionInstructions,
    /// Map of option IDs to their rubric descriptions (between 2 and 255 options;
    /// an empty string `""` indicates no description for that option).
    pub criteria: HashMap<String, String>,
}

impl ChoiceQuestion {
    /// Creates a new [`ChoiceQuestion`] from instructions and `(option_id, description)` pairs.
    pub fn new<I, K, V>(instructions: impl Into<DecisionInstructions>, criteria: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        Self {
            instructions: instructions.into(),
            criteria: criteria
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        }
    }

    /// Creates a new [`ChoiceQuestion`] from option IDs without descriptions (each mapped to `""`).
    pub fn from_options<I, S>(instructions: impl Into<DecisionInstructions>, options: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            instructions: instructions.into(),
            criteria: options
                .into_iter()
                .map(|opt| (opt.into(), String::new()))
                .collect(),
        }
    }

    /// Adds or updates a single option ID and its description.
    pub fn add_option(
        mut self,
        option_id: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        self.criteria.insert(option_id.into(), description.into());
        self
    }
}

/// A quantitative rubric scoring question (`"type": "score"`) over discrete numeric levels.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ScoreQuestion {
    /// The question or scoring instructions.
    pub instructions: DecisionInstructions,
    /// Discrete score levels in the rubric (between 2 and 255 non-empty levels).
    ///
    /// Each 0-based index `i` corresponds to numeric level value `i` with rubric
    /// description `criteria[i]`.
    pub criteria: Vec<String>,
}

impl ScoreQuestion {
    /// Creates a new [`ScoreQuestion`] with the given instructions and rubric level descriptions.
    pub fn new<I, S>(instructions: impl Into<DecisionInstructions>, criteria: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            instructions: instructions.into(),
            criteria: criteria.into_iter().map(Into::into).collect(),
        }
    }

    /// Appends a rubric level description.
    pub fn add_level(mut self, description: impl Into<String>) -> Self {
        self.criteria.push(description.into());
        self
    }
}

// ============================================================================
// Response, Answer, PromptFeedback & Usage Types
// ============================================================================

/// Response returned by `POST /v1beta/decisions` (`decisions.create`).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct CreateDecisionResponse {
    /// Decision answers keyed by the question identifiers from [`CreateDecisionRequest::questions`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answers: Option<HashMap<String, Answer>>,
    /// Feedback from input content filters if the prompt was blocked or flagged.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_feedback: Option<DecisionPromptFeedback>,
    /// Token usage statistics for the decision request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<DecisionUsage>,
}

impl CreateDecisionResponse {
    /// Looks up the [`Answer`] for the given `question_id`.
    pub fn get_answer(&self, question_id: &str) -> Option<&Answer> {
        self.answers.as_ref()?.get(question_id)
    }

    /// Looks up a [`NoulAnswer`] for the given `question_id`.
    pub fn get_noul(&self, question_id: &str) -> Option<&NoulAnswer> {
        self.get_answer(question_id)?.as_noul()
    }

    /// Looks up a [`ChoiceAnswer`] for the given `question_id`.
    pub fn get_choice(&self, question_id: &str) -> Option<&ChoiceAnswer> {
        self.get_answer(question_id)?.as_choice()
    }

    /// Looks up a [`ScoreAnswer`] for the given `question_id`.
    pub fn get_score(&self, question_id: &str) -> Option<&ScoreAnswer> {
        self.get_answer(question_id)?.as_score()
    }
}

/// Decision answer corresponding to a single [`Question`].
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Answer {
    /// Answer for a binary (`noul`) question (`"type": "noul"`).
    Noul(NoulAnswer),
    /// Answer for a categorical (`choice`) question (`"type": "choice"`).
    Choice(ChoiceAnswer),
    /// Answer for a quantitative (`score`) question (`"type": "score"`).
    Score(ScoreAnswer),
}

impl Answer {
    /// Returns the closed-world confidence score (`[0, 1]`) for this answer.
    pub fn confidence(&self) -> f64 {
        match self {
            Self::Noul(ans) => ans.confidence,
            Self::Choice(ans) => ans.confidence,
            Self::Score(ans) => ans.confidence,
        }
    }

    /// Returns a reference to the inner [`NoulAnswer`] if this is [`Answer::Noul`].
    pub fn as_noul(&self) -> Option<&NoulAnswer> {
        match self {
            Self::Noul(ans) => Some(ans),
            _ => None,
        }
    }

    /// Returns a reference to the inner [`ChoiceAnswer`] if this is [`Answer::Choice`].
    pub fn as_choice(&self) -> Option<&ChoiceAnswer> {
        match self {
            Self::Choice(ans) => Some(ans),
            _ => None,
        }
    }

    /// Returns a reference to the inner [`ScoreAnswer`] if this is [`Answer::Score`].
    pub fn as_score(&self) -> Option<&ScoreAnswer> {
        match self {
            Self::Score(ans) => Some(ans),
            _ => None,
        }
    }
}

/// Answer for a binary [`NoulQuestion`] (`"type": "noul"`).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NoulAnswer {
    /// Calibrated probability `P(true)` in `[0.01, 0.99]`.
    pub noul: f64,
    /// Closed-world confidence score in `[0, 1]`, equal to `max(noul, 1 - noul)`.
    pub confidence: f64,
}

/// Answer for a categorical [`ChoiceQuestion`] (`"type": "choice"`).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ChoiceAnswer {
    /// The option ID with the highest probability.
    pub choice: String,
    /// Closed-world confidence score in `[0, 1]`, equal to the probability of `choice`.
    pub confidence: f64,
    /// Normalized probability in `[0, 1]` for each option ID.
    pub probabilities: HashMap<String, f64>,
}

/// Answer for a quantitative [`ScoreQuestion`] (`"type": "score"`).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ScoreAnswer {
    /// Probability-weighted expected score (`sum_k k * probability_k`).
    pub score: f64,
    /// Closed-world confidence score in `[0, 1]`, equal to the highest level probability.
    pub confidence: f64,
    /// Normalized probability in `[0, 1]` for each 0-based level index (formatted as a string key).
    pub probabilities: HashMap<String, f64>,
    /// Rubric description for each 0-based level index (formatted as a string key).
    pub legend: HashMap<String, String>,
}

/// Feedback from input content filters on a [`CreateDecisionResponse`].
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct DecisionPromptFeedback {
    /// The reason why the input was blocked, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_reason: Option<DecisionBlockReason>,
    /// Optional human-readable message detailing the reason for blocking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_reason_message: Option<String>,
}

/// Reason why a decision request input was blocked by content filters.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecisionBlockReason {
    /// Blocked for safety reasons (`"safety"`).
    #[serde(alias = "SAFETY")]
    Safety,
    /// Blocked for other reasons (`"other"`).
    #[serde(alias = "OTHER")]
    Other,
    /// Blocked due to a blocklist match (`"blocklist"`).
    #[serde(alias = "BLOCKLIST")]
    Blocklist,
    /// Blocked due to prohibited content (`"prohibited_content"`).
    #[serde(alias = "PROHIBITED_CONTENT")]
    ProhibitedContent,
    /// Blocked due to image safety reasons (`"image_safety"`).
    #[serde(alias = "IMAGE_SAFETY")]
    ImageSafety,
}

/// Statistics on a decision request's token and tool usage.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct DecisionUsage {
    /// Breakdown of cached token counts by modality.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens_by_modality: Option<Vec<DecisionModalityTokens>>,
    /// Count of grounding tool invocations by tool type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding_tool_count: Option<Vec<DecisionGroundingToolCount>>,
    /// Number of tokens in the initial input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial_input_tokens: Option<i32>,
    /// Breakdown of input token counts by modality.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_by_modality: Option<Vec<DecisionModalityTokens>>,
    /// Breakdown of output token counts by modality.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_by_modality: Option<Vec<DecisionModalityTokens>>,
    /// Breakdown of tool-use token counts by modality.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use_tokens_by_modality: Option<Vec<DecisionModalityTokens>>,
    /// Total number of cached tokens used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_cached_tokens: Option<i32>,
    /// Total number of input tokens billed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_input_tokens: Option<i32>,
    /// Total number of output tokens billed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_output_tokens: Option<i32>,
    /// Total number of reasoning/thought tokens billed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_thought_tokens: Option<i32>,
    /// Total number of tokens across the request and response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<i32>,
    /// Total number of tool-use tokens billed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tool_use_tokens: Option<i32>,
}

/// Token count attributed to a specific modality in [`DecisionUsage`].
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DecisionModalityTokens {
    /// The content modality.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modality: Option<DecisionModality>,
    /// Number of tokens for this modality.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<i32>,
}

/// Content modality reported in [`DecisionModalityTokens`].
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecisionModality {
    /// Text modality (`"text"`).
    #[serde(alias = "TEXT")]
    Text,
    /// Image modality (`"image"`).
    #[serde(alias = "IMAGE")]
    Image,
    /// Audio modality (`"audio"`).
    #[serde(alias = "AUDIO")]
    Audio,
    /// Video modality (`"video"`).
    #[serde(alias = "VIDEO")]
    Video,
    /// Document modality (`"document"`).
    #[serde(alias = "DOCUMENT")]
    Document,
}

/// Grounding tool usage count reported in [`DecisionUsage`].
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DecisionGroundingToolCount {
    /// Number of invocations of the grounding tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<i32>,
    /// Type of grounding tool invoked.
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<DecisionGroundingToolType>,
}

/// Type of grounding tool reported in [`DecisionGroundingToolCount`].
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecisionGroundingToolType {
    /// Google Search grounding (`"google_search"`).
    GoogleSearch,
    /// Google Maps grounding (`"google_maps"`).
    GoogleMaps,
    /// Retrieval grounding (`"retrieval"`).
    Retrieval,
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn serialize_and_deserialize_full_decision_request_and_response() {
        let request = CreateDecisionRequest::builder()
            .model("gemini-decision-1")
            .text_input("Kitchen counter has a puddle of spilled coffee near the edge.")
            .add_noul_question(
                "has_spill",
                NoulQuestion::new("Is there an uncleaned liquid spill?")
                    .true_description("Visible liquid puddle or wet spill on the surface")
                    .false_description("Surface is dry and clean"),
            )
            .add_choice_question(
                "urgency",
                ChoiceQuestion::new(
                    "What is the cleanup urgency?",
                    [
                        ("low", "Cosmetic smudge only"),
                        ("medium", "Small contained spill"),
                        ("high", "Slip or electrical hazard"),
                    ],
                ),
            )
            .add_score_question(
                "cleanliness_score",
                ScoreQuestion::new(
                    "Rate the cleanliness of the counter.",
                    [
                        "Severely soiled with active spills",
                        "Minor crumbs or small spill",
                        "Spotless and sanitized",
                    ],
                ),
            )
            .build();

        let serialized = serde_json::to_value(&request).expect("serialize request");
        assert_eq!(serialized["model"], "gemini-decision-1");
        assert_eq!(
            serialized["input"],
            "Kitchen counter has a puddle of spilled coffee near the edge."
        );
        assert_eq!(serialized["questions"]["has_spill"]["type"], "noul");
        assert_eq!(
            serialized["questions"]["has_spill"]["instructions"],
            "Is there an uncleaned liquid spill?"
        );
        assert_eq!(
            serialized["questions"]["has_spill"]["criteria"]["true_description"],
            "Visible liquid puddle or wet spill on the surface"
        );
        assert_eq!(serialized["questions"]["urgency"]["type"], "choice");
        assert_eq!(
            serialized["questions"]["urgency"]["criteria"]["medium"],
            "Small contained spill"
        );
        assert_eq!(
            serialized["questions"]["cleanliness_score"]["type"],
            "score"
        );
        assert_eq!(
            serialized["questions"]["cleanliness_score"]["criteria"],
            json!([
                "Severely soiled with active spills",
                "Minor crumbs or small spill",
                "Spotless and sanitized"
            ])
        );

        let round_trip: CreateDecisionRequest =
            serde_json::from_value(serialized).expect("deserialize request");
        assert_eq!(round_trip, request);

        let response_json = json!({
            "answers": {
                "has_spill": {
                    "type": "noul",
                    "noul": 0.96,
                    "confidence": 0.96
                },
                "urgency": {
                    "type": "choice",
                    "choice": "medium",
                    "confidence": 0.78,
                    "probabilities": {
                        "low": 0.07,
                        "medium": 0.78,
                        "high": 0.15
                    }
                },
                "cleanliness_score": {
                    "type": "score",
                    "score": 0.85,
                    "confidence": 0.75,
                    "probabilities": {
                        "0": 0.20,
                        "1": 0.75,
                        "2": 0.05
                    },
                    "legend": {
                        "0": "Severely soiled with active spills",
                        "1": "Minor crumbs or small spill",
                        "2": "Spotless and sanitized"
                    }
                }
            },
            "usage": {
                "total_input_tokens": 120,
                "total_output_tokens": 18,
                "total_tokens": 138,
                "input_tokens_by_modality": [
                    { "modality": "text", "tokens": 120 }
                ]
            }
        });

        let response: CreateDecisionResponse =
            serde_json::from_value(response_json.clone()).expect("deserialize response");

        let noul = response.get_noul("has_spill").expect("has_spill noul");
        assert_eq!(noul.noul, 0.96);
        assert_eq!(noul.confidence, 0.96);
        assert_eq!(
            response.get_answer("has_spill").map(Answer::confidence),
            Some(0.96)
        );

        let choice = response.get_choice("urgency").expect("urgency choice");
        assert_eq!(choice.choice, "medium");
        assert_eq!(choice.confidence, 0.78);
        assert_eq!(choice.probabilities.get("medium"), Some(&0.78));

        let score = response
            .get_score("cleanliness_score")
            .expect("cleanliness_score score");
        assert_eq!(score.score, 0.85);
        assert_eq!(score.confidence, 0.75);
        assert_eq!(
            score.legend.get("1").map(String::as_str),
            Some("Minor crumbs or small spill")
        );

        let usage = response.usage.as_ref().expect("usage present");
        assert_eq!(usage.total_input_tokens, Some(120));
        assert_eq!(usage.total_output_tokens, Some(18));
        assert_eq!(usage.total_tokens, Some(138));
        assert_eq!(
            usage.input_tokens_by_modality,
            Some(vec![DecisionModalityTokens {
                modality: Some(DecisionModality::Text),
                tokens: Some(120),
            }])
        );

        let reserialized = serde_json::to_value(&response).expect("reserialize response");
        assert_eq!(reserialized, response_json);
    }

    #[test]
    fn multimodal_input_and_structured_instructions_round_trip() {
        let request = CreateDecisionRequest::builder()
            .text_input("Inspect the following media items:")
            .add_input_content(DecisionInputContent::image_bytes(b"png-bytes", "image/png"))
            .add_input_content(DecisionInputContent::Video(DecisionVideoContent {
                uri: Some("https://generativelanguage.googleapis.com/v1beta/files/vid-1".into()),
                mime_type: Some("video/mp4".into()),
                name: Some("clip_1".into()),
                processing: Some(DecisionVideoProcessing::StaticConfig(
                    StaticMediaProcessing::new()
                        .with_start_offset("0s")
                        .with_end_offset("5s")
                        .with_fps(2.0),
                )),
                resolution: Some(DecisionMediaResolution::High),
                ..Default::default()
            }))
            .add_input_content(DecisionInputContent::audio_uri(
                "https://generativelanguage.googleapis.com/v1beta/files/aud-1",
                "audio/wav",
            ))
            .add_input_content(DecisionInputContent::document_uri(
                "https://generativelanguage.googleapis.com/v1beta/files/doc-1",
                "application/pdf",
            ))
            .add_question(
                "q1",
                Question::Noul(NoulQuestion::new(vec![DecisionInstructionContent::text(
                    "Does the clip match the document?",
                )])),
            )
            .build();

        let value = serde_json::to_value(&request).expect("serialize multimodal request");
        assert_eq!(value["input"][0]["type"], "text");
        assert_eq!(value["input"][1]["type"], "image");
        assert_eq!(value["input"][1]["data"], "cG5nLWJ5dGVz");
        assert_eq!(value["input"][2]["type"], "video");
        assert_eq!(value["input"][2]["processing"]["type"], "static");
        assert_eq!(value["input"][2]["processing"]["fps"], 2.0);
        assert_eq!(value["input"][2]["resolution"], "high");
        assert_eq!(value["input"][3]["type"], "audio");
        assert_eq!(value["input"][4]["type"], "document");
        assert_eq!(
            value["questions"]["q1"]["instructions"][0],
            json!({
                "type": "text",
                "text": "Does the clip match the document?"
            })
        );

        let decoded: CreateDecisionRequest =
            serde_json::from_value(value).expect("deserialize multimodal request");
        assert_eq!(decoded, request);
    }

    #[test]
    fn prompt_feedback_and_annotations_round_trip() {
        let text_with_annotations =
            DecisionTextContent::new("Cited statement").with_annotations(vec![
                DecisionAnnotation::UrlCitation(DecisionUrlCitation {
                    start_index: Some(0),
                    end_index: Some(15),
                    title: Some("Example".into()),
                    url: Some("https://example.com".into()),
                }),
                DecisionAnnotation::WordInfo(DecisionWordInfo {
                    word: Some("Cited".into()),
                    start_offset: Some(0.1),
                    end_offset: Some(0.4),
                    start_index: Some(0),
                    end_index: Some(5),
                }),
            ]);

        let content = DecisionInputContent::Text(text_with_annotations);
        let json_val = serde_json::to_value(&content).expect("serialize annotated text");
        assert_eq!(json_val["annotations"][0]["type"], "url_citation");
        assert_eq!(json_val["annotations"][1]["type"], "word_info");
        let parsed_content: DecisionInputContent =
            serde_json::from_value(json_val).expect("deserialize annotated text");
        assert_eq!(parsed_content, content);

        let feedback_resp: CreateDecisionResponse = serde_json::from_value(json!({
            "prompt_feedback": {
                "block_reason": "safety",
                "block_reason_message": "Prompt blocked by safety filter"
            }
        }))
        .expect("deserialize prompt_feedback");
        assert_eq!(
            feedback_resp.prompt_feedback,
            Some(DecisionPromptFeedback {
                block_reason: Some(DecisionBlockReason::Safety),
                block_reason_message: Some("Prompt blocked by safety filter".into()),
            })
        );
    }

    #[test]
    fn strict_deserialization_rejects_unknown_question_and_answer_types() {
        let unknown_question = json!({
            "type": "freeform",
            "instructions": "Explain why"
        });
        assert!(serde_json::from_value::<Question>(unknown_question).is_err());

        let unknown_answer = json!({
            "type": "freeform",
            "text": "Because..."
        });
        assert!(serde_json::from_value::<Answer>(unknown_answer).is_err());

        let malformed_noul_answer = json!({
            "type": "noul",
            "noul": 0.8
            // missing required `confidence`
        });
        assert!(serde_json::from_value::<Answer>(malformed_noul_answer).is_err());

        let malformed_input_block = json!({
            "type": "text"
            // missing required `text`
        });
        assert!(serde_json::from_value::<DecisionInput>(malformed_input_block).is_err());

        let unknown_input_block = json!({
            "type": "hologram",
            "uri": "files/123"
        });
        assert!(serde_json::from_value::<DecisionInput>(unknown_input_block).is_err());
    }
}
