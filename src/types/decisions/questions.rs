// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Question, instruction, and rubric criteria types for the Decisions API.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::content::DecisionTextContent;

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

/// Instructions for a decision question ([`NoulQuestion`], [`ChoiceQuestion`], or [`ScoreQuestion`]).
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
