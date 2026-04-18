# geologia

A Rust client library for the Google Generative AI API.

## Building

To build the library, run the following command:

```bash
cargo build --release
```

## Running Examples

The project includes several examples in the `examples` directory. To run an example, you'll first need to set up your environment with your Gemini API key.

1.  Create a `.env` file in the root of the project:
    ```
    GEMINI_API_KEY=your_api_key_here
    ```

2.  Run an example using `cargo run`:
    ```bash
    cargo run --example generate-content
    ```

## Usage

To use this library in your own Rust application, add it as a dependency in your `Cargo.toml` file:

```toml
[dependencies]
geologia = "0.1.0" # Or the latest version
```

Here is a basic example of how to use the `GeminiClient` to generate content:

```rust
use std::env;
use geologia::prelude::{Content, GeminiClient, GenerateContentRequest, Role};

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    // Get the API key from the environment.
    let api_key = env::var("GEMINI_API_KEY")?;

    // Create a new GeminiClient.
    let gemini_client = GeminiClient::new(api_key);

    // Build a prompt with a single user message.
    let prompt = vec![
        Content::builder()
            .role(Role::User)
            .add_text_part("What is the airspeed of an unladen swallow?")
            .build(),
    ];

    // Create the content generation request.
    let request = GenerateContentRequest::builder().contents(prompt).build();

    // Call the API to generate content.
    let response = gemini_client
        .generate_content(&request, "gemini-3-pro-preview")
        .await?;

    // Print the response.
    println!("Response: {:?}", response.candidates[0].get_text().unwrap());
    
    Ok(())
}
```
