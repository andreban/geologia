use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{Content, VertexApiError};
use crate::error::Result;

/// Request body for the `generateContent` and `streamGenerateContent` endpoints.
///
/// Use [`GenerateContentRequest::builder`] for ergonomic construction.
///
/// See <https://ai.google.dev/api/generate-content#request-body>.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentRequest {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tools>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_settings: Option<Vec<SafetySetting>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
}

impl GenerateContentRequest {
    /// Returns a new [`GenerateContentRequestBuilder`].
    pub fn builder() -> GenerateContentRequestBuilder {
        GenerateContentRequestBuilder::new()
    }
}

/// Builder for [`GenerateContentRequest`].
#[derive(Debug)]
pub struct GenerateContentRequestBuilder {
    request: GenerateContentRequest,
}

impl GenerateContentRequestBuilder {
    fn new() -> Self {
        GenerateContentRequestBuilder {
            request: GenerateContentRequest::default(),
        }
    }

    /// Sets the conversation contents.
    pub fn contents(mut self, contents: Vec<Content>) -> Self {
        self.request.contents = contents;
        self
    }

    /// Sets the generation configuration.
    pub fn generation_config(mut self, generation_config: GenerationConfig) -> Self {
        self.request.generation_config = Some(generation_config);
        self
    }

    /// Sets the tools available to the model (e.g. function calling, Google Search).
    pub fn tools(mut self, tools: Vec<Tools>) -> Self {
        self.request.tools = Some(tools);
        self
    }

    /// Sets the safety filter settings.
    pub fn safety_settings(mut self, safety_settings: Vec<SafetySetting>) -> Self {
        self.request.safety_settings = Some(safety_settings);
        self
    }

    /// Sets a system instruction to guide the model's behavior.
    pub fn system_instruction(mut self, system_instruction: Content) -> Self {
        self.request.system_instruction = Some(system_instruction);
        self
    }

    /// Consumes the builder and returns the constructed [`GenerateContentRequest`].
    pub fn build(self) -> GenerateContentRequest {
        self.request
    }
}

/// A set of tool declarations the model may use during generation.
///
/// See <https://ai.google.dev/api/caching#Tool>.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Tools {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_declarations: Option<Vec<FunctionDeclaration>>,

    #[serde(rename = "googleSearchRetrieval")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub google_search_retrieval: Option<GoogleSearchRetrieval>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub google_search: Option<GoogleSearch>,
}

/// Enables the Google Search grounding tool (no configuration required).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GoogleSearch {}

/// Configuration for dynamic retrieval in Google Search grounding.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DynamicRetrievalConfig {
    /// The retrieval mode (e.g. `"MODE_DYNAMIC"`).
    pub mode: String,
    /// The threshold for triggering retrieval. Defaults to `0.7`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynamic_threshold: Option<f32>,
}

impl Default for DynamicRetrievalConfig {
    fn default() -> Self {
        Self {
            mode: "MODE_DYNAMIC".to_string(),
            dynamic_threshold: Some(0.7),
        }
    }
}

/// Google Search retrieval tool with dynamic retrieval configuration.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleSearchRetrieval {
    /// Configuration controlling when retrieval is triggered.
    pub dynamic_retrieval_config: DynamicRetrievalConfig,
}

/// Parameters that control how the model generates content.
///
/// Use [`GenerationConfig::builder`] for ergonomic construction.
///
/// See <https://ai.google.dev/api/generate-content#generationconfig>.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,
}

impl GenerationConfig {
    /// Returns a new [`GenerationConfigBuilder`].
    pub fn builder() -> GenerationConfigBuilder {
        GenerationConfigBuilder::new()
    }
}

/// Builder for [`GenerationConfig`].
#[derive(Debug)]
pub struct GenerationConfigBuilder {
    generation_config: GenerationConfig,
}

impl GenerationConfigBuilder {
    fn new() -> Self {
        Self {
            generation_config: Default::default(),
        }
    }

    pub fn max_output_tokens<T: Into<i32>>(mut self, max_output_tokens: T) -> Self {
        self.generation_config.max_output_tokens = Some(max_output_tokens.into());
        self
    }

    pub fn temperature<T: Into<f32>>(mut self, temperature: T) -> Self {
        self.generation_config.temperature = Some(temperature.into());
        self
    }

    pub fn top_p<T: Into<f32>>(mut self, top_p: T) -> Self {
        self.generation_config.top_p = Some(top_p.into());
        self
    }

    pub fn top_k<T: Into<i32>>(mut self, top_k: T) -> Self {
        self.generation_config.top_k = Some(top_k.into());
        self
    }

    pub fn stop_sequences<T: Into<Vec<String>>>(mut self, stop_sequences: T) -> Self {
        self.generation_config.stop_sequences = Some(stop_sequences.into());
        self
    }

    pub fn candidate_count<T: Into<u32>>(mut self, candidate_count: T) -> Self {
        self.generation_config.candidate_count = Some(candidate_count.into());
        self
    }

    pub fn response_mime_type<T: Into<String>>(mut self, response_mime_type: T) -> Self {
        self.generation_config.response_mime_type = Some(response_mime_type.into());
        self
    }

    pub fn response_schema<T: Into<Value>>(mut self, response_schema: T) -> Self {
        self.generation_config.response_schema = Some(response_schema.into());
        self
    }

    pub fn thinking_config(mut self, thinking_config: ThinkingConfig) -> Self {
        self.generation_config.thinking_config = Some(thinking_config);
        self
    }

    /// Consumes the builder and returns the constructed [`GenerationConfig`].
    pub fn build(self) -> GenerationConfig {
        self.generation_config
    }
}

/// Configuration for the model's "thinking" (chain-of-thought) behavior.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingConfig {
    pub include_thoughts: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<isize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<ThinkingLevel>,
}

/// The level of thinking effort the model should use.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ThinkingLevel {
    ThinkingLevelUnspecified,
    Low,
    High,
}

/// A safety filter configuration that controls blocking thresholds for harmful content.
///
/// See <https://ai.google.dev/api/generate-content#safetysetting>.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SafetySetting {
    pub category: HarmCategory,
    pub threshold: HarmBlockThreshold,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<HarmBlockMethod>,
}

/// Categories of potentially harmful content.
///
/// See <https://ai.google.dev/api/generate-content#harmcategory>.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum HarmCategory {
    #[serde(rename = "HARM_CATEGORY_UNSPECIFIED")]
    Unspecified,
    #[serde(rename = "HARM_CATEGORY_HATE_SPEECH")]
    HateSpeech,
    #[serde(rename = "HARM_CATEGORY_DANGEROUS_CONTENT")]
    DangerousContent,
    #[serde(rename = "HARM_CATEGORY_HARASSMENT")]
    Harassment,
    #[serde(rename = "HARM_CATEGORY_SEXUALLY_EXPLICIT")]
    SexuallyExplicit,
}

/// The threshold at which harmful content is blocked.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum HarmBlockThreshold {
    #[serde(rename = "HARM_BLOCK_THRESHOLD_UNSPECIFIED")]
    Unspecified,
    #[serde(rename = "BLOCK_LOW_AND_ABOVE")]
    BlockLowAndAbove,
    #[serde(rename = "BLOCK_MEDIUM_AND_ABOVE")]
    BlockMediumAndAbove,
    #[serde(rename = "BLOCK_ONLY_HIGH")]
    BlockOnlyHigh,
    #[serde(rename = "BLOCK_NONE")]
    BlockNone,
}

/// The method used to evaluate harm (severity-based or probability-based).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum HarmBlockMethod {
    #[serde(rename = "HARM_BLOCK_METHOD_UNSPECIFIED")]
    Unspecified, // HARM_BLOCK_METHOD_UNSPECIFIED
    #[serde(rename = "SEVERITY")]
    Severity, // SEVERITY
    #[serde(rename = "PROBABILITY")]
    Probability, // PROBABILITY
}

/// A single candidate response generated by the model.
///
/// See <https://ai.google.dev/api/generate-content#candidate>.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citation_metadata: Option<CitationMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_ratings: Option<Vec<SafetyRating>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    pub index: u32,
}

impl Candidate {
    /// Returns the concatenated text from this candidate's content, if any.
    pub fn get_text(&self) -> Option<String> {
        match &self.content {
            Some(content) => content.get_text(),
            None => None,
        }
    }
}

/// A citation to a source used by the model in its response.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Citation {
    pub start_index: Option<i32>,
    pub end_index: Option<i32>,
    pub uri: Option<String>,
}

/// Metadata containing citations for a candidate's content.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CitationMetadata {
    #[serde(alias = "citationSources")]
    pub citations: Vec<Citation>,
}

/// A safety rating for a piece of content across a specific harm category.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SafetyRating {
    pub category: String,
    pub probability: String,
    pub probability_score: Option<f32>,
    pub severity: Option<String>,
    pub severity_score: Option<f32>,
}

/// Token usage statistics for a generate content request/response.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    pub candidates_token_count: Option<u32>,
    pub prompt_token_count: Option<u32>,
    pub total_token_count: Option<u32>,
}

/// A declaration of a function the model may call.
///
/// See <https://ai.google.dev/api/caching#FunctionDeclaration>.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionDeclaration {
    pub name: String,
    pub description: String,
    // TODO: add behaviour field -  https://ai.google.dev/api/caching#Behavior
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters_json_schema: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_json_schema: Option<Value>,
}

/// See <https://ai.google.dev/api/caching#FunctionResponse>.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub name: String,
    pub response: Value,
    pub parts: Option<Vec<FunctionResponsePart>>, // TODO: Add missing properties from docs.
    pub will_continue: Option<bool>,
    pub scheduling: Option<Scheduling>,
}

/// See <https://ai.google.dev/api/caching#FunctionResponsePart>.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FunctionResponsePart {
    InlineData(FunctionResponseBlob),
}

/// See <https://ai.google.dev/api/caching#FunctionResponseBlob>.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionResponseBlob {
    pub mime_type: String,
    pub data: String,
}

/// See <https://ai.google.dev/api/caching#Scheduling>.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Scheduling {
    SchedulingUnspecified,
    Silent,
    WhenIdle,
    Interrupt,
}

/// A single property within a function's parameter schema.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionParametersProperty {
    pub r#type: String,
    pub description: String,
}

/// The raw response from the `generateContent` endpoint, which may be a success or an error.
///
/// Use [`into_result`](GenerateContentResponse::into_result) to convert into a standard
/// `Result<GenerateContentResponseResult>`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GenerateContentResponse {
    Ok(GenerateContentResponseResult),
    Error(GenerateContentResponseError),
}

impl From<GenerateContentResponse> for Result<GenerateContentResponseResult> {
    fn from(val: GenerateContentResponse) -> Self {
        match val {
            GenerateContentResponse::Ok(result) => Ok(result),
            GenerateContentResponse::Error(error) => Err(error.error.into()),
        }
    }
}

/// A successful response from the `generateContent` endpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentResponseResult {
    pub candidates: Vec<Candidate>,
    pub usage_metadata: Option<UsageMetadata>,
}

/// An error response from the `generateContent` endpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerateContentResponseError {
    pub error: VertexApiError,
}

impl GenerateContentResponse {
    /// Converts this response into a `Result`, mapping the error variant to [`crate::error::Error`].
    pub fn into_result(self) -> Result<GenerateContentResponseResult> {
        match self {
            GenerateContentResponse::Ok(result) => Ok(result),
            GenerateContentResponse::Error(error) => Err(error.error.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::types::{Candidate, UsageMetadata};

    use super::GenerateContentResponseResult;

    #[test]
    pub fn parses_usage_metadata() {
        let input = r#"
        {
          "promptTokenCount": 11,
          "candidatesTokenCount": 202,
          "totalTokenCount": 1041,
          "promptTokensDetails": [
            {
              "modality": "TEXT",
              "tokenCount": 11
            }
          ],
          "thoughtsTokenCount": 828
        }"#;
        let _ = serde_json::from_str::<UsageMetadata>(input).unwrap();
    }

    #[test]
    pub fn parses_candidate() {
        let input = r#"
          {
            "content": {
              "parts": [
                {
                  "text": "**What do you mean? An African or a European swallow?**\n\nIf you are looking for the actual physics rather than the *Monty Python and the Holy Grail* reference, here is the breakdown:\n\n**1. The European Swallow**\nBased on an analysis published by Jonathan Corum (using data on the Strouhal number of cruising flight), the estimated airspeed velocity of an unladen European Swallow is roughly **11 meters per second**, or **24 miles per hour**.\n\n**2. The African Swallow**\nData on the African swallow is scarcer, mostly because—as the guard in the movie points out—African swallows are non-migratory. However, since they are similar in size to their European counterparts, their cruising speed would likely be comparable.\n\nBut of course, the real question is: *Could it carry a coconut?* (A five-ounce bird could not carry a one-pound coconut. It is a simple question of weight ratios.)",
                  "thoughtSignature": "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"
                }
              ],
              "role": "model"
            },
            "finishReason": "STOP",
            "index": 0
          }"#;
        let _ = serde_json::from_str::<Candidate>(input).unwrap();
    }

    #[test]
    pub fn parses_google_ai_response() {
        let input = r#"
        {
          "candidates": [
            {
              "content": {
                "parts": [
                  {
                    "text": "**What do you mean? An African or a European swallow?**\n\nIf you are looking for the actual physics rather than the *Monty Python and the Holy Grail* reference, here is the breakdown:\n\n**1. The European Swallow**\nBased on an analysis published by Jonathan Corum (using data on the Strouhal number of cruising flight), the estimated airspeed velocity of an unladen European Swallow is roughly **11 meters per second**, or **24 miles per hour**.\n\n**2. The African Swallow**\nData on the African swallow is scarcer, mostly because—as the guard in the movie points out—African swallows are non-migratory. However, since they are similar in size to their European counterparts, their cruising speed would likely be comparable.\n\nBut of course, the real question is: *Could it carry a coconut?* (A five-ounce bird could not carry a one-pound coconut. It is a simple question of weight ratios.)",
                    "thoughtSignature": "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"
                  }
                ],
                "role": "model"
              },
              "finishReason": "STOP",
              "index": 0
            }
          ],
          "usageMetadata": {
            "promptTokenCount": 11,
            "candidatesTokenCount": 202,
            "totalTokenCount": 1041,
            "promptTokensDetails": [
              {
                "modality": "TEXT",
                "tokenCount": 11
              }
            ],
            "thoughtsTokenCount": 828
          },
          "modelVersion": "gemini-3-pro-preview",
          "responseId": "2uUdaYPkG73WvdIP2aPs2Ak"
        }
      "#;
        let _ = serde_json::from_str::<GenerateContentResponseResult>(input).unwrap();
    }
}
