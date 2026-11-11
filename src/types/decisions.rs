// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Types for the Gemini Decisions API (`POST /v1beta/decisions`).
//!
//! The Decisions API evaluates one or more structured decision questions (`noul`,
//! `choice`, or `score`) against a shared text or multimodal input context using
//! a decision model (such as `"gemini-decision-1"`), returning calibrated
//! probabilities and closed-world confidence scores.

mod content;
mod questions;
mod response;

pub use content::*;
pub use questions::*;
pub use response::*;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

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
    /// Sets the decision model name (for example, `"gemini-decision-1"`).
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
    pub fn build(self) -> CreateDecisionRequest {
        CreateDecisionRequest {
            model: self.model.unwrap_or_default(),
            input: self
                .input
                .unwrap_or_else(|| DecisionInput::Text(String::new())),
            questions: self.questions,
        }
    }
}

#[cfg(test)]
#[path = "decisions_tests.rs"]
mod tests;
