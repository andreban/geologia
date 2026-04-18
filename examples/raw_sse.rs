use std::env;

use geologia::{
    network::event_source::EventSource,
    prelude::{Content, GenerateContentRequest, Role},
};
use tokio_stream::StreamExt;

static MODEL: &str = "gemini-2.5-flash";

#[tokio::main]
pub async fn main() {
    let prompt = vec![
        Content::builder()
            .role(Role::User)
            .add_text_part("What is the airspeed of an unladen swallow?")
            .build(),
    ];
    let request = GenerateContentRequest::builder().contents(prompt).build();
    let _ = dotenvy::dotenv();
    let api_key = env::var("GEMINI_API_KEY").unwrap();
    let client = reqwest::Client::new();
    let endpoint_url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:streamGenerateContent?alt=sse"
    );
    let mut event_stream = client
        .post(&endpoint_url)
        .header("x-goog-api-key", api_key)
        .json(&request)
        .send()
        .await
        .unwrap()
        .event_stream();
    while let Some(event) = event_stream.next().await {
        println!("{event:?}")
    }
}
