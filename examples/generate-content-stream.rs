use std::{env, error::Error};

use geologia::prelude::{Content, GeminiClient, GenerateContentRequest, Role};
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt().init();
    let _ = dotenvy::dotenv();

    let api_key = env::var("GEMINI_API_KEY")?;
    let gemini_client = GeminiClient::new(api_key);
    let prompt = vec![
        Content::builder()
            .role(Role::User)
            .add_text_part("What is the airspeed of an unladen swallow?")
            .build(),
    ];

    let request = GenerateContentRequest::builder().contents(prompt).build();
    let mut response_stream = gemini_client
        .stream_generate_content(&request, "gemini-3-pro-preview")
        .await?;

    while let Some(content) = response_stream.next().await {
        match content {
            Ok(response) => println!("Response: {:?}", response.candidates[0].get_text().unwrap()),
            Err(err) => eprintln!("Error: {}", err),
        }
    }
    Ok(())
}
