use serde::{Deserialize, Serialize};

use crate::{error::Result, prelude::*};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub text: String,
}

impl Message {
    pub fn new(role: Role, text: &str) -> Self {
        Message {
            role,
            text: text.to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dialogue {
    model: String,
    messages: Vec<Message>,
}

impl Dialogue {
    pub fn new(model: &str) -> Self {
        Dialogue {
            model: model.to_string(),
            messages: vec![],
        }
    }

    pub async fn do_turn(&mut self, gemini: &GeminiClient, message: &str) -> Result<Message> {
        self.messages.push(Message::new(Role::User, message));
        let response = gemini
            .prompt_conversation(&self.messages, &self.model)
            .await?;
        self.messages.push(response.clone());
        Ok(response)
    }
}
