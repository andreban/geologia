use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

use super::Content;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CountTokensRequest {
    pub contents: Content,
}

impl CountTokensRequest {
    pub fn builder() -> CountTokensRequestBuilder {
        CountTokensRequestBuilder::default()
    }
}

#[derive(Debug, Default)]
pub struct CountTokensRequestBuilder {
    contents: Content,
}

impl CountTokensRequestBuilder {
    pub fn from_prompt(prompt: &str) -> Self {
        CountTokensRequestBuilder {
            contents: Content {
                parts: Some(vec![super::Part::from_text(prompt.to_string())]),
                ..Default::default()
            },
        }
    }

    pub fn build(self) -> CountTokensRequest {
        CountTokensRequest {
            contents: self.contents,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CountTokensResponse {
    Ok(CountTokensResponseResult),
    Error { error: super::VertexApiError },
}

impl CountTokensResponse {
    pub fn into_result(self) -> Result<CountTokensResponseResult> {
        match self {
            CountTokensResponse::Ok(result) => Ok(result),
            CountTokensResponse::Error { error } => Err(Error::VertexError(error)),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CountTokensResponseResult {
    pub total_tokens: i32,
    pub total_billable_characters: u32,
}
