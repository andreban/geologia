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

use crate::error::{Error, Result};

/// Maximum number of questions allowed in a single [`CreateDecisionRequest`].
pub const MAX_DECISION_QUESTIONS: usize = 1_000;

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

    /// Validates that `model`, `input`, and `questions` satisfy the Decisions API contract.
    pub fn validate(&self) -> Result<()> {
        if self.model.trim().is_empty() {
            return Err(Error::InvalidArgument {
                message: "decision request `model` must not be empty".into(),
            });
        }
        self.input.validate()?;
        if self.questions.is_empty() {
            return Err(Error::InvalidArgument {
                message: "decision request requires at least 1 question".into(),
            });
        }
        if self.questions.len() > MAX_DECISION_QUESTIONS {
            return Err(Error::InvalidArgument {
                message: format!(
                    "decision request supports at most {MAX_DECISION_QUESTIONS} questions, got {}",
                    self.questions.len()
                ),
            });
        }
        for (id, question) in &self.questions {
            if id.trim().is_empty() {
                return Err(Error::InvalidArgument {
                    message: "decision question ID must not be empty".into(),
                });
            }
            question.validate()?;
        }
        Ok(())
    }
}

/// Builder for [`CreateDecisionRequest`].
#[derive(Default, Debug, Clone)]
pub struct CreateDecisionRequestBuilder {
    model: Option<String>,
    input: Option<DecisionInput>,
    questions: HashMap<String, Question>,
    validation_error: Option<String>,
}

impl CreateDecisionRequestBuilder {
    fn record_error(&mut self, err: Error) {
        if self.validation_error.is_none() {
            self.validation_error = Some(match err {
                Error::InvalidArgument { message } => message,
                other => other.to_string(),
            });
        }
    }

    /// Sets the decision model name (for example, `"gemini-decision-1"`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        let model = model.into();
        if model.trim().is_empty() {
            self.record_error(Error::InvalidArgument {
                message: "decision request `model` must not be empty".into(),
            });
        }
        self.model = Some(model);
        self
    }

    /// Sets the shared input context from any type convertible into [`DecisionInput`].
    pub fn input(mut self, input: impl Into<DecisionInput>) -> Self {
        let input = input.into();
        if let Err(err) = input.validate() {
            self.record_error(err);
        }
        self.input = Some(input);
        self
    }

    /// Sets the shared input context to a plain text string.
    pub fn text_input(mut self, text: impl Into<String>) -> Self {
        let input = DecisionInput::Text(text.into());
        if let Err(err) = input.validate() {
            self.record_error(err);
        }
        self.input = Some(input);
        self
    }

    /// Appends a multimodal [`DecisionInputContent`] block to the request input.
    ///
    /// If the builder currently holds a plain text input or a single content block,
    /// it is converted into a sequence of content blocks (`DecisionInput::Contents`).
    pub fn add_input_content(mut self, content: DecisionInputContent) -> Self {
        if let Err(err) = content.validate() {
            self.record_error(err);
        }
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
        if self.questions.is_empty() {
            self.record_error(Error::InvalidArgument {
                message: "decision request requires at least 1 question".into(),
            });
        } else if self.questions.len() > MAX_DECISION_QUESTIONS {
            self.record_error(Error::InvalidArgument {
                message: format!(
                    "decision request supports at most {MAX_DECISION_QUESTIONS} questions, got {}",
                    self.questions.len()
                ),
            });
        }
        let mut first_err = None;
        for (id, question) in &self.questions {
            if id.trim().is_empty() {
                first_err = Some(Error::InvalidArgument {
                    message: "decision question ID must not be empty".into(),
                });
                break;
            }
            if let Err(err) = question.validate() {
                first_err = Some(err);
                break;
            }
        }
        if let Some(err) = first_err {
            self.record_error(err);
        }
        self
    }

    /// Inserts a single decision question under the given identifier.
    pub fn add_question(mut self, id: impl Into<String>, question: impl Into<Question>) -> Self {
        let id = id.into();
        let question = question.into();
        if id.trim().is_empty() {
            self.record_error(Error::InvalidArgument {
                message: "decision question ID must not be empty".into(),
            });
        }
        if let Err(err) = question.validate() {
            self.record_error(err);
        }
        self.questions.insert(id, question);
        if self.questions.len() > MAX_DECISION_QUESTIONS {
            let count = self.questions.len();
            self.record_error(Error::InvalidArgument {
                message: format!(
                    "decision request supports at most {MAX_DECISION_QUESTIONS} questions, got {count}"
                ),
            });
        }
        self
    }

    /// Inserts a binary [`NoulQuestion`] under the given identifier.
    pub fn add_noul_question(self, id: impl Into<String>, question: NoulQuestion) -> Self {
        self.add_question(id, Question::Noul(question))
    }

    /// Inserts a categorical [`ChoiceQuestion`] under the given identifier.
    pub fn add_choice_question(self, id: impl Into<String>, question: ChoiceQuestion) -> Self {
        self.add_question(id, Question::Choice(question))
    }

    /// Inserts a quantitative [`ScoreQuestion`] under the given identifier.
    pub fn add_score_question(self, id: impl Into<String>, question: ScoreQuestion) -> Self {
        self.add_question(id, Question::Score(question))
    }

    /// Validates and builds the [`CreateDecisionRequest`].
    ///
    /// Returns [`Error::InvalidArgument`] if any setter received invalid data, or if `model`,
    /// `input`, or `questions` (between 1 and [`MAX_DECISION_QUESTIONS`]) are missing or invalid.
    pub fn build(self) -> Result<CreateDecisionRequest> {
        if let Some(message) = self.validation_error {
            return Err(Error::InvalidArgument { message });
        }
        let Some(model) = self.model else {
            return Err(Error::InvalidArgument {
                message: "decision request requires `model` to be set".into(),
            });
        };
        let Some(input) = self.input else {
            return Err(Error::InvalidArgument {
                message: "decision request requires `input` to be set".into(),
            });
        };
        let request = CreateDecisionRequest {
            model,
            input,
            questions: self.questions,
        };
        request.validate()?;
        Ok(request)
    }
}

#[cfg(test)]
#[path = "decisions_tests.rs"]
mod tests;
