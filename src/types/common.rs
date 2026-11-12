// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use std::{fmt::Display, str::FromStr, vec};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_with::base64::Base64;
use serde_with::serde_as;

use crate::types::FunctionResponse;

/// A conversation message containing one or more [`Part`]s.
///
/// See <https://ai.google.dev/api/caching#Content>.
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct Content {
    /// The role of the message author (`user` or `model`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    /// The ordered parts that make up this message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parts: Option<Vec<Part>>,
}

impl Content {
    /// Concatenates all [`PartData::Text`] parts into a single string.
    ///
    /// Returns `None` if there are no parts.
    pub fn get_text(&self) -> Option<String> {
        self.parts.as_ref().map(|parts| {
            parts
                .iter()
                .filter_map(|part| match &part.data {
                    PartData::Text(text) => Some(text.clone()),
                    _ => None,
                })
                .collect::<String>()
        })
    }

    /// Finds the first [`PartData::InlineData`] part and decodes its base64 payload into raw bytes,
    /// returning `(mime_type, decoded_bytes)`.
    ///
    /// Returns `None` if there are no `InlineData` parts, or `Some(Err(...))` if base64 decoding fails.
    pub fn get_inline_data(&self) -> Option<crate::error::Result<(&str, Vec<u8>)>> {
        self.parts
            .as_ref()?
            .iter()
            .find_map(Part::decode_inline_data)
    }

    /// Creates a [`Content`] containing a single text part, suitable for use as a system instruction.
    pub fn system_prompt<S: Into<String>>(system_prompt: S) -> Self {
        Self::builder().add_text_part(system_prompt).build()
    }

    /// Returns a new [`ContentBuilder`].
    pub fn builder() -> ContentBuilder {
        ContentBuilder::default()
    }
}

/// Builder for constructing [`Content`] values incrementally.
#[derive(Clone, Debug, Default)]
pub struct ContentBuilder {
    content: Content,
}

impl ContentBuilder {
    /// Appends a text part to this content.
    pub fn add_text_part<T: Into<String>>(self, text: T) -> Self {
        self.add_part(Part::from_text(text.into()))
    }

    /// Appends an inline binary [`Part`] (base64-encoded automatically) with the given MIME type.
    pub fn add_inline_data<S: Into<String>>(self, mime_type: S, bytes: &[u8]) -> Self {
        self.add_part(Part::from_inline_data(mime_type, bytes))
    }

    /// Appends an arbitrary [`Part`] to this content.
    pub fn add_part(mut self, part: Part) -> Self {
        match &mut self.content.parts {
            Some(parts) => parts.push(part),
            None => self.content.parts = Some(vec![part]),
        }
        self
    }

    /// Sets the [`Role`] for this content.
    pub fn role(mut self, role: Role) -> Self {
        self.content.role = Some(role);
        self
    }

    /// Consumes the builder and returns the constructed [`Content`].
    pub fn build(self) -> Content {
        self.content
    }
}

/// The role of a message author in a conversation.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Model,
}

impl Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let role_str = match self {
            Role::User => "user",
            Role::Model => "model",
        };
        f.write_str(role_str)
    }
}

impl FromStr for Role {
    type Err = ();

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "user" => Ok(Role::User),
            "model" => Ok(Role::Model),
            _ => Err(()),
        }
    }
}

/// A single unit of content within a [`Content`] message.
///
/// See <https://ai.google.dev/api/caching#Part>.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    // This is of a Struct type, a Map of values, so either a Value or Map<String, Value> are appropriate.
    //See https://protobuf.dev/reference/protobuf/google.protobuf/#struct
    #[serde(skip_serializing_if = "Option::is_none")]
    pub part_metadata: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_resolution: Option<Value>, // TODO: Create type for media_resolution.
    /// Metadata for a video part (clipping offsets and frame rate).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_metadata: Option<VideoMetadata>,
    /// Metadata for a text part used in speech synthesis (speaker and style).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speech_metadata: Option<SpeechMetadata>,
    #[serde(flatten)]
    pub data: PartData, // Create enum for data.
}

/// Metadata describing how a video [`Part`] should be clipped or sampled.
///
/// See <https://ai.google.dev/api/caching#VideoMetadata>.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct VideoMetadata {
    /// Optional start offset of the video segment (e.g. `"0s"` or `"10.5s"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_offset: Option<String>,
    /// Optional end offset of the video segment (e.g. `"60s"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_offset: Option<String>,
    /// Optional frame rate of the video sent to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fps: Option<f64>,
}

/// Metadata describing speaker identity and vocal style for a text [`Part`] during speech synthesis.
///
/// See <https://ai.google.dev/api/caching#SpeechMetadata>.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SpeechMetadata {
    /// Optional speaker identifier matching a configured speaker voice.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    /// Optional natural-language description of the vocal style or delivery tone.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
}

/// Information about a single recognized word within an [`AudioTranscription`].
///
/// See <https://ai.google.dev/api/caching#WordInfo>.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct WordInfo {
    /// Transcript of the word.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub word: Option<String>,
    /// Optional start offset in time of the word relative to the start of the audio.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_offset: Option<String>,
    /// Optional end offset in time of the word relative to the start of the audio.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_offset: Option<String>,
}

/// Transcription of an audio segment returned by the model.
///
/// See <https://ai.google.dev/api/caching#AudioTranscription>.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AudioTranscription {
    /// The transcribed text of the audio segment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Optional label identifying the speaker of the audio segment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_label: Option<String>,
    /// Optional word-level transcription and timing metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<WordInfo>>,
}

/// A server-side tool call predicted by the model.
///
/// Returned when [`ToolConfig::include_server_side_tool_invocations`](crate::types::ToolConfig::include_server_side_tool_invocations) is enabled.
///
/// See <https://ai.google.dev/api/caching#ToolCall>.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    /// Optional unique identifier for the tool call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Optional name of the tool being called.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// The type of tool being invoked (e.g. `"GOOGLE_SEARCH_WEB"`, `"URL_CONTEXT"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<String>,
    /// Optional arguments passed to the tool, as a JSON object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<Value>,
}

/// The result of a server-side tool invocation.
///
/// Returned when [`ToolConfig::include_server_side_tool_invocations`](crate::types::ToolConfig::include_server_side_tool_invocations) is enabled.
///
/// See <https://ai.google.dev/api/caching#ToolResponse>.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ToolResponse {
    /// Optional unique identifier matching the corresponding [`ToolCall::id`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The type of tool that was called (e.g. `"GOOGLE_SEARCH_WEB"`, `"URL_CONTEXT"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<String>,
    /// Optional output returned by the tool execution, as a JSON object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<Value>,
}

/// The payload of a [`Part`], representing different content types.
///
/// See <https://ai.google.dev/api/caching#Part>.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum PartData {
    /// Plain text content.
    Text(String),
    /// Binary data encoded inline. See <https://ai.google.dev/api/caching#Blob>.
    #[serde(rename_all = "camelCase")]
    InlineData {
        /// Optional display name of the blob.
        #[serde(skip_serializing_if = "Option::is_none")]
        display_name: Option<String>,
        /// The IANA MIME type of the data (e.g. `"image/png"`).
        mime_type: String,
        /// Base64-encoded binary data.
        data: String,
    },
    /// A function call requested by the model. See <https://ai.google.dev/api/caching#FunctionCall>.
    FunctionCall {
        /// Optional unique identifier for the function call.
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// The name of the function to call.
        name: String,
        /// The arguments to pass, as a JSON object.
        #[serde(skip_serializing_if = "Option::is_none")]
        args: Option<Value>,
    },
    /// A response to a function call. See <https://ai.google.dev/api/caching#FunctionResponse>.
    FunctionResponse(FunctionResponse),
    /// A reference to a file stored in the API.
    FileData(Value),
    /// Code to be executed by the model.
    ExecutableCode(Value),
    /// The result of executing code.
    CodeExecutionResult(Value),
    /// Transcription of an audio segment. See <https://ai.google.dev/api/caching#AudioTranscription>.
    AudioTranscription(AudioTranscription),
    /// Media processing details for a part.
    MediaProcessing(Value),
    /// A server-side tool call predicted by the model. See <https://ai.google.dev/api/caching#ToolCall>.
    ToolCall(ToolCall),
    /// The result of a server-side tool invocation. See <https://ai.google.dev/api/caching#ToolResponse>.
    ToolResponse(ToolResponse),
}

#[serde_as]
#[derive(Serialize, Deserialize)]
#[serde(transparent)]
struct Base64Bytes(#[serde_as(as = "Base64")] Vec<u8>);

pub(crate) fn encode_base64(bytes: &[u8]) -> String {
    let Ok(Value::String(s)) = serde_json::to_value(Base64Bytes(bytes.to_vec())) else {
        unreachable!("Base64 serialization always produces a JSON string");
    };
    s
}

fn decode_base64(encoded: &str) -> crate::error::Result<Vec<u8>> {
    let value = Value::String(encoded.to_owned());
    let bytes: Base64Bytes = serde_json::from_value(value)?;
    Ok(bytes.0)
}

impl Part {
    /// Creates a [`Part`] containing only text.
    pub fn from_text<S: Into<String>>(text: S) -> Self {
        Self {
            thought: None,
            thought_signature: None,
            part_metadata: None,
            media_resolution: None,
            video_metadata: None,
            speech_metadata: None,
            data: PartData::Text(text.into()),
        }
    }

    /// Creates a [`Part`] containing inline binary data, base64-encoding `bytes`.
    pub fn from_inline_data<S: Into<String>>(mime_type: S, bytes: &[u8]) -> Self {
        Self {
            thought: None,
            thought_signature: None,
            part_metadata: None,
            media_resolution: None,
            video_metadata: None,
            speech_metadata: None,
            data: PartData::InlineData {
                display_name: None,
                mime_type: mime_type.into(),
                data: encode_base64(bytes),
            },
        }
    }

    /// If this part's data is [`PartData::InlineData`], decodes its base64 data into raw bytes
    /// and returns `Some(Ok((mime_type, bytes)))` (or `Some(Err(...))` if base64 decoding fails).
    ///
    /// Returns `None` for any other [`PartData`] variant.
    pub fn decode_inline_data(&self) -> Option<crate::error::Result<(&str, Vec<u8>)>> {
        match &self.data {
            PartData::InlineData {
                mime_type, data, ..
            } => Some(decode_base64(data).map(|bytes| (mime_type.as_str(), bytes))),
            _ => None,
        }
    }
}

/// Serde adapter for Protobuf `int64` optional fields, which serialize as decimal JSON strings
/// (`"1048576"`) and deserialize from either decimal strings or JSON integers per Proto3 JSON
/// mapping.
pub(crate) mod opt_int64_str {
    use std::fmt;

    use serde::de::{self, Deserializer, Visitor};
    use serde::ser::Serializer;

    pub fn serialize<S>(value: &Option<i64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(v) => serializer.serialize_str(&v.to_string()),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct OptInt64Visitor;

        impl<'de> Visitor<'de> for OptInt64Visitor {
            type Value = Option<i64>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an optional 64-bit integer or decimal string")
            }

            fn visit_none<E>(self) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(None)
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(None)
            }

            fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where
                D: Deserializer<'de>,
            {
                deserializer.deserialize_any(Int64Visitor).map(Some)
            }
        }

        struct Int64Visitor;

        impl Visitor<'_> for Int64Visitor {
            type Value = i64;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a 64-bit integer or decimal string")
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(value)
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                i64::try_from(value).map_err(de::Error::custom)
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                value.parse::<i64>().map_err(de::Error::custom)
            }
        }

        deserializer.deserialize_option(OptInt64Visitor)
    }
}

#[cfg(test)]
mod test {
    use serde_json::json;

    use crate::types::{
        AudioTranscription, Content, Part, PartData, Role, SpeechMetadata, ToolCall, ToolResponse,
        VideoMetadata, WordInfo,
    };

    fn part(data: PartData) -> Part {
        Part {
            thought: None,
            thought_signature: None,
            part_metadata: None,
            media_resolution: None,
            video_metadata: None,
            speech_metadata: None,
            data,
        }
    }

    #[test]
    fn inline_data_omits_unset_display_name() -> serde_json::Result<()> {
        let part = part(PartData::InlineData {
            display_name: None,
            mime_type: "image/png".to_string(),
            data: "aGVsbG8=".to_string(),
        });

        let value = serde_json::to_value(&part)?;

        assert_eq!(
            value,
            json!({
                "inlineData": {
                    "mimeType": "image/png",
                    "data": "aGVsbG8="
                }
            })
        );
        Ok(())
    }

    #[test]
    fn inline_data_includes_set_display_name() -> serde_json::Result<()> {
        let part = part(PartData::InlineData {
            display_name: Some("photo.png".to_string()),
            mime_type: "image/png".to_string(),
            data: "aGVsbG8=".to_string(),
        });

        let value = serde_json::to_value(&part)?;

        assert_eq!(
            value,
            json!({
                "inlineData": {
                    "displayName": "photo.png",
                    "mimeType": "image/png",
                    "data": "aGVsbG8="
                }
            })
        );
        Ok(())
    }

    #[test]
    fn inline_data_parses_without_display_name() -> serde_json::Result<()> {
        let input = json!({
            "inlineData": {
                "mimeType": "image/png",
                "data": "aGVsbG8="
            }
        });

        let part: Part = serde_json::from_value(input)?;

        assert!(matches!(
            part.data,
            PartData::InlineData {
                display_name: None,
                ref mime_type,
                ref data
            } if mime_type == "image/png" && data == "aGVsbG8="
        ));
        Ok(())
    }

    #[test]
    fn inline_data_parses_with_display_name() -> serde_json::Result<()> {
        let input = json!({
            "inlineData": {
                "displayName": "photo.png",
                "mimeType": "image/png",
                "data": "aGVsbG8="
            }
        });

        let part: Part = serde_json::from_value(input)?;

        assert!(matches!(
            part.data,
            PartData::InlineData {
                display_name: Some(ref name),
                ref mime_type,
                ref data
            } if name == "photo.png" && mime_type == "image/png" && data == "aGVsbG8="
        ));
        Ok(())
    }

    #[test]
    fn function_call_omits_unset_id_and_args() -> serde_json::Result<()> {
        let part = part(PartData::FunctionCall {
            id: None,
            name: "get_weather".to_string(),
            args: None,
        });

        let value = serde_json::to_value(&part)?;

        assert_eq!(value, json!({ "functionCall": { "name": "get_weather" } }));
        Ok(())
    }

    #[test]
    fn function_call_includes_set_id_and_args() -> serde_json::Result<()> {
        let part = part(PartData::FunctionCall {
            id: Some("call-1".to_string()),
            name: "get_weather".to_string(),
            args: Some(json!({ "city": "Lisbon" })),
        });

        let value = serde_json::to_value(&part)?;

        assert_eq!(
            value,
            json!({
                "functionCall": {
                    "id": "call-1",
                    "name": "get_weather",
                    "args": { "city": "Lisbon" }
                }
            })
        );
        Ok(())
    }

    #[test]
    fn function_call_without_id_and_args_parses_as_none() -> serde_json::Result<()> {
        let input = json!({ "functionCall": { "name": "get_weather" } });

        let part: Part = serde_json::from_value(input)?;

        assert!(matches!(
            part.data,
            PartData::FunctionCall { id: None, ref name, args: None } if name == "get_weather"
        ));
        Ok(())
    }

    #[test]
    fn system_prompt_omits_unset_role() -> serde_json::Result<()> {
        let content = Content::system_prompt("You are a helpful assistant.");

        let value = serde_json::to_value(&content)?;

        assert_eq!(
            value,
            json!({ "parts": [{ "text": "You are a helpful assistant." }] })
        );
        Ok(())
    }

    #[test]
    fn content_omits_unset_parts() -> serde_json::Result<()> {
        let content = Content::builder().role(Role::User).build();

        let value = serde_json::to_value(&content)?;

        assert_eq!(value, json!({ "role": "user" }));
        Ok(())
    }

    #[test]
    fn content_includes_set_role_and_parts() -> serde_json::Result<()> {
        let content = Content::builder()
            .role(Role::Model)
            .add_text_part("Hello!")
            .build();

        let value = serde_json::to_value(&content)?;

        assert_eq!(
            value,
            json!({ "role": "model", "parts": [{ "text": "Hello!" }] })
        );
        Ok(())
    }

    #[test]
    fn content_without_role_parses_as_none() -> serde_json::Result<()> {
        let input = json!({ "parts": [{ "text": "Hello!" }] });

        let content: Content = serde_json::from_value(input)?;

        assert!(content.role.is_none());
        assert_eq!(content.get_text().as_deref(), Some("Hello!"));
        Ok(())
    }

    #[test]
    pub fn parses_text_part() {
        let input = r#"
        {
          "text": "**What do you mean? An African or a European swallow?**\n\nIf you are looking for the actual physics rather than the *Monty Python and the Holy Grail* reference, here is the breakdown:\n\n**1. The European Swallow**\nBased on an analysis published by Jonathan Corum (using data on the Strouhal number of cruising flight), the estimated airspeed velocity of an unladen European Swallow is roughly **11 meters per second**, or **24 miles per hour**.\n\n**2. The African Swallow**\nData on the African swallow is scarcer, mostly because—as the guard in the movie points out—African swallows are non-migratory. However, since they are similar in size to their European counterparts, their cruising speed would likely be comparable.\n\nBut of course, the real question is: *Could it carry a coconut?* (A five-ounce bird could not carry a one-pound coconut. It is a simple question of weight ratios.)",
          "thoughtSignature": "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"
        }"#;

        let _ = serde_json::from_str::<Part>(input).unwrap();
    }

    #[test]
    fn video_metadata_and_speech_metadata_round_trip_on_part() -> serde_json::Result<()> {
        let video_part = Part {
            video_metadata: Some(VideoMetadata {
                start_offset: Some("0s".to_string()),
                end_offset: Some("60s".to_string()),
                fps: Some(2.0),
            }),
            ..part(PartData::FileData(json!({
                "fileUri": "https://generativelanguage.googleapis.com/v1beta/files/video123",
                "mimeType": "video/mp4"
            })))
        };

        let video_json = serde_json::to_value(&video_part)?;
        assert_eq!(
            video_json,
            json!({
                "fileData": {
                    "fileUri": "https://generativelanguage.googleapis.com/v1beta/files/video123",
                    "mimeType": "video/mp4"
                },
                "videoMetadata": {
                    "startOffset": "0s",
                    "endOffset": "60s",
                    "fps": 2.0
                }
            })
        );
        assert_eq!(serde_json::from_value::<Part>(video_json)?, video_part);

        let speech_part = Part {
            speech_metadata: Some(SpeechMetadata {
                speaker: Some("Joe".to_string()),
                style: Some("cheerful and friendly".to_string()),
            }),
            ..Part::from_text("How's it going today?")
        };

        let speech_json = serde_json::to_value(&speech_part)?;
        assert_eq!(
            speech_json,
            json!({
                "text": "How's it going today?",
                "speechMetadata": {
                    "speaker": "Joe",
                    "style": "cheerful and friendly"
                }
            })
        );
        assert_eq!(serde_json::from_value::<Part>(speech_json)?, speech_part);

        assert_eq!(serde_json::to_value(VideoMetadata::default())?, json!({}));
        assert_eq!(serde_json::to_value(SpeechMetadata::default())?, json!({}));
        Ok(())
    }

    #[test]
    fn audio_transcription_and_media_processing_round_trip() -> serde_json::Result<()> {
        let transcription_part = part(PartData::AudioTranscription(AudioTranscription {
            text: Some("Hello world".to_string()),
            speaker_label: Some("spk_1".to_string()),
            words: Some(vec![
                WordInfo {
                    word: Some("Hello".to_string()),
                    start_offset: Some("0.100s".to_string()),
                    end_offset: Some("0.400s".to_string()),
                },
                WordInfo {
                    word: Some("world".to_string()),
                    start_offset: Some("0.500s".to_string()),
                    end_offset: Some("0.900s".to_string()),
                },
            ]),
        }));

        let transcription_json = serde_json::to_value(&transcription_part)?;
        assert_eq!(
            transcription_json,
            json!({
                "audioTranscription": {
                    "text": "Hello world",
                    "speakerLabel": "spk_1",
                    "words": [
                        { "word": "Hello", "startOffset": "0.100s", "endOffset": "0.400s" },
                        { "word": "world", "startOffset": "0.500s", "endOffset": "0.900s" }
                    ]
                }
            })
        );
        assert_eq!(
            serde_json::from_value::<Part>(transcription_json)?,
            transcription_part
        );
        assert_eq!(
            serde_json::to_value(AudioTranscription::default())?,
            json!({})
        );
        assert_eq!(serde_json::to_value(WordInfo::default())?, json!({}));

        let media_processing_part = part(PartData::MediaProcessing(json!({
            "mode": "AGENTIC"
        })));

        let media_processing_json = serde_json::to_value(&media_processing_part)?;
        assert_eq!(
            media_processing_json,
            json!({
                "mediaProcessing": {
                    "mode": "AGENTIC"
                }
            })
        );
        assert_eq!(
            serde_json::from_value::<Part>(media_processing_json)?,
            media_processing_part
        );
        Ok(())
    }

    #[test]
    fn tool_call_and_tool_response_parse_and_round_trip() -> serde_json::Result<()> {
        let call_json = json!({
            "toolCall": {
                "id": "tool-call-1",
                "toolName": "google_search",
                "toolType": "GOOGLE_SEARCH_WEB",
                "args": { "query": "rust serde flatten enum" }
            }
        });

        let call_part: Part = serde_json::from_value(call_json.clone())?;
        assert_eq!(
            call_part,
            part(PartData::ToolCall(ToolCall {
                id: Some("tool-call-1".to_string()),
                tool_name: Some("google_search".to_string()),
                tool_type: Some("GOOGLE_SEARCH_WEB".to_string()),
                args: Some(json!({ "query": "rust serde flatten enum" })),
            }))
        );
        assert_eq!(serde_json::to_value(&call_part)?, call_json);

        let response_json = json!({
            "toolResponse": {
                "id": "tool-call-1",
                "toolType": "GOOGLE_SEARCH_WEB",
                "response": { "results": [{ "title": "Serde" }] }
            }
        });

        let response_part: Part = serde_json::from_value(response_json.clone())?;
        assert_eq!(
            response_part,
            part(PartData::ToolResponse(ToolResponse {
                id: Some("tool-call-1".to_string()),
                tool_type: Some("GOOGLE_SEARCH_WEB".to_string()),
                response: Some(json!({ "results": [{ "title": "Serde" }] })),
            }))
        );
        assert_eq!(serde_json::to_value(&response_part)?, response_json);

        assert_eq!(serde_json::to_value(ToolCall::default())?, json!({}));
        assert_eq!(serde_json::to_value(ToolResponse::default())?, json!({}));
        Ok(())
    }

    #[test]
    fn unrecognized_part_kind_fails_deserialization() {
        let input = json!({
            "futurePart": { "nested": [1, 2, 3] }
        });
        assert!(serde_json::from_value::<Part>(input).is_err());
    }

    #[test]
    fn malformed_part_data_fails_deserialization() {
        // A functionCall missing its required "name" must fail deserialization rather than being silently ignored.
        let input = json!({
            "functionCall": {
                "args": { "city": "Lisbon" }
            }
        });
        assert!(serde_json::from_value::<Part>(input).is_err());
    }

    #[derive(Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct DummyOptInt64 {
        #[serde(
            default,
            with = "super::opt_int64_str",
            skip_serializing_if = "Option::is_none"
        )]
        count: Option<i64>,
    }

    #[derive(Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct DummyOptInt64AlwaysSerialized {
        #[serde(default, with = "super::opt_int64_str")]
        count: Option<i64>,
    }

    #[test]
    fn opt_int64_str_serialization_encodes_as_string_and_omits_none() -> serde_json::Result<()> {
        let none = DummyOptInt64 { count: None };
        assert_eq!(serde_json::to_value(&none)?, json!({}));

        let none_always = DummyOptInt64AlwaysSerialized { count: None };
        assert_eq!(
            serde_json::to_value(&none_always)?,
            json!({ "count": null })
        );

        let zero = DummyOptInt64 { count: Some(0) };
        assert_eq!(serde_json::to_value(&zero)?, json!({ "count": "0" }));

        let positive = DummyOptInt64 {
            count: Some(1_048_576),
        };
        assert_eq!(
            serde_json::to_value(&positive)?,
            json!({ "count": "1048576" })
        );

        let negative = DummyOptInt64 { count: Some(-42) };
        assert_eq!(serde_json::to_value(&negative)?, json!({ "count": "-42" }));

        let max_val = DummyOptInt64 {
            count: Some(i64::MAX),
        };
        assert_eq!(
            serde_json::to_value(&max_val)?,
            json!({ "count": "9223372036854775807" })
        );

        let min_val = DummyOptInt64 {
            count: Some(i64::MIN),
        };
        assert_eq!(
            serde_json::to_value(&min_val)?,
            json!({ "count": "-9223372036854775808" })
        );

        Ok(())
    }

    #[test]
    fn opt_int64_str_deserialization_from_strings_and_integers() -> serde_json::Result<()> {
        // From decimal strings
        let from_str: DummyOptInt64 = serde_json::from_value(json!({ "count": "1048576" }))?;
        assert_eq!(from_str.count, Some(1_048_576));

        let from_negative_str: DummyOptInt64 = serde_json::from_value(json!({ "count": "-42" }))?;
        assert_eq!(from_negative_str.count, Some(-42));

        let from_max_str: DummyOptInt64 =
            serde_json::from_value(json!({ "count": "9223372036854775807" }))?;
        assert_eq!(from_max_str.count, Some(i64::MAX));

        let from_min_str: DummyOptInt64 =
            serde_json::from_value(json!({ "count": "-9223372036854775808" }))?;
        assert_eq!(from_min_str.count, Some(i64::MIN));

        // From integer literals (i64 / u64)
        let from_int: DummyOptInt64 = serde_json::from_value(json!({ "count": 1048576 }))?;
        assert_eq!(from_int.count, Some(1_048_576));

        let from_negative_int: DummyOptInt64 = serde_json::from_value(json!({ "count": -42 }))?;
        assert_eq!(from_negative_int.count, Some(-42));

        let from_zero_int: DummyOptInt64 = serde_json::from_value(json!({ "count": 0 }))?;
        assert_eq!(from_zero_int.count, Some(0));

        let from_max_int: DummyOptInt64 = serde_json::from_value(json!({ "count": i64::MAX }))?;
        assert_eq!(from_max_int.count, Some(i64::MAX));

        // From null
        let from_null: DummyOptInt64 = serde_json::from_value(json!({ "count": null }))?;
        assert_eq!(from_null.count, None);

        // From missing field
        let from_empty: DummyOptInt64 = serde_json::from_value(json!({}))?;
        assert_eq!(from_empty.count, None);

        Ok(())
    }

    #[test]
    fn opt_int64_str_round_trip() -> serde_json::Result<()> {
        let values = [
            None,
            Some(0),
            Some(1),
            Some(-1),
            Some(1_048_576),
            Some(-1_048_576),
            Some(i64::MAX),
            Some(i64::MIN),
        ];

        for &val in &values {
            let item = DummyOptInt64 { count: val };
            let serialized = serde_json::to_value(&item)?;
            let deserialized: DummyOptInt64 = serde_json::from_value(serialized)?;
            assert_eq!(item, deserialized);
        }

        Ok(())
    }

    #[test]
    fn opt_int64_str_rejects_malformed_and_overflow_payloads() {
        // Non-numeric string
        assert!(
            serde_json::from_value::<DummyOptInt64>(json!({ "count": "not_a_number" })).is_err()
        );

        // Empty string
        assert!(serde_json::from_value::<DummyOptInt64>(json!({ "count": "" })).is_err());

        // Float string
        assert!(serde_json::from_value::<DummyOptInt64>(json!({ "count": "12.34" })).is_err());

        // Overflow string (> i64::MAX)
        assert!(
            serde_json::from_value::<DummyOptInt64>(json!({
                "count": "9223372036854775808"
            }))
            .is_err()
        );

        // Underflow string (< i64::MIN)
        assert!(
            serde_json::from_value::<DummyOptInt64>(json!({
                "count": "-9223372036854775809"
            }))
            .is_err()
        );

        // Float JSON number
        assert!(serde_json::from_value::<DummyOptInt64>(json!({ "count": 12.34 })).is_err());

        // u64 integer exceeding i64::MAX
        assert!(serde_json::from_value::<DummyOptInt64>(json!({ "count": u64::MAX })).is_err());

        // Boolean
        assert!(serde_json::from_value::<DummyOptInt64>(json!({ "count": true })).is_err());

        // Array
        assert!(serde_json::from_value::<DummyOptInt64>(json!({ "count": [1, 2] })).is_err());

        // Object
        assert!(serde_json::from_value::<DummyOptInt64>(json!({ "count": { "a": 1 } })).is_err());
    }

    #[test]
    fn part_from_inline_data_serializes_and_decodes() -> crate::error::Result<()> {
        let part = Part::from_inline_data("image/png", b"hello");

        let serialized = serde_json::to_value(&part)?;
        assert_eq!(
            serialized,
            json!({
                "inlineData": {
                    "mimeType": "image/png",
                    "data": "aGVsbG8="
                }
            })
        );

        let decoded = part.decode_inline_data();
        assert!(matches!(decoded, Some(Ok(("image/png", ref bytes))) if bytes == b"hello"));
        assert_eq!(decoded.transpose()?, Some(("image/png", b"hello".to_vec())));

        Ok(())
    }

    #[test]
    fn part_from_text_decode_inline_data_returns_none() {
        let part = Part::from_text("hi");
        assert!(part.decode_inline_data().is_none());
    }

    #[test]
    fn content_builder_add_inline_data_and_get_inline_data() -> crate::error::Result<()> {
        let content = Content::builder()
            .add_text_part("caption")
            .add_inline_data("image/jpeg", &[1, 2, 3, 4])
            .build();

        let inline_data = content.get_inline_data();
        assert!(
            matches!(inline_data, Some(Ok(("image/jpeg", ref bytes))) if bytes == &[1, 2, 3, 4])
        );
        assert_eq!(
            inline_data.transpose()?,
            Some(("image/jpeg", vec![1, 2, 3, 4]))
        );

        Ok(())
    }

    #[test]
    fn content_get_inline_data_returns_none_when_empty_or_text_only() {
        let empty = Content::default();
        assert!(empty.get_inline_data().is_none());

        let text_only = Content::system_prompt("hello");
        assert!(text_only.get_inline_data().is_none());
    }

    #[test]
    fn invalid_base64_in_inline_data_returns_serde_error() {
        let part = Part {
            thought: None,
            thought_signature: None,
            part_metadata: None,
            media_resolution: None,
            video_metadata: None,
            speech_metadata: None,
            data: PartData::InlineData {
                display_name: None,
                mime_type: "image/png".to_string(),
                data: "not valid base64!@#$%".to_string(),
            },
        };

        let decoded = part.decode_inline_data();
        assert!(matches!(decoded, Some(Err(crate::error::Error::Serde(_)))));

        let content = Content {
            role: None,
            parts: Some(vec![part]),
        };
        let content_decoded = content.get_inline_data();
        assert!(matches!(
            content_decoded,
            Some(Err(crate::error::Error::Serde(_)))
        ));
    }
}
