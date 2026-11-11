// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Response, answer, prompt feedback, and token usage types for the Decisions API.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Response returned by `POST /v1beta/decisions` (`decisions.create`).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct CreateDecisionResponse {
    /// Decision answers keyed by the question identifiers from
    /// [`CreateDecisionRequest::questions`](super::CreateDecisionRequest::questions).
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

/// Decision answer corresponding to a single [`Question`](super::Question).
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

/// Answer for a binary [`NoulQuestion`](super::NoulQuestion) (`"type": "noul"`).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NoulAnswer {
    /// Calibrated probability `P(true)` in `[0.01, 0.99]`.
    pub noul: f64,
    /// Closed-world confidence score in `[0, 1]`, equal to `max(noul, 1 - noul)`.
    pub confidence: f64,
}

/// Answer for a categorical [`ChoiceQuestion`](super::ChoiceQuestion) (`"type": "choice"`).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ChoiceAnswer {
    /// The option ID with the highest probability.
    pub choice: String,
    /// Closed-world confidence score in `[0, 1]`, equal to the probability of `choice`.
    pub confidence: f64,
    /// Normalized probability in `[0, 1]` for each option ID.
    pub probabilities: HashMap<String, f64>,
}

/// Answer for a quantitative [`ScoreQuestion`](super::ScoreQuestion) (`"type": "score"`).
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
