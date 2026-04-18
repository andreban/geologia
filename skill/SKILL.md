---
name: geologia
description: >
  Help Rust developers use the geologia crate to integrate Google's Generative AI (Gemini) API
  into their projects. Use this skill whenever the user is working with geologia, building Rust
  apps that call Gemini models, generating text or images from Rust, using function calling with
  Gemini in Rust, or asking how to add AI features to a Rust project using the geologia crate.
  Also trigger when the user asks about streaming Gemini responses, token counting, text
  embeddings, or structured output in Rust — even if they don't explicitly mention geologia.
---

# geologia

`geologia` is an async Rust client for Google's Generative AI API (Gemini). It supports text
generation (including streaming), token counting, text embeddings, image generation (Imagen),
and function calling.

## Adding the dependency

```toml
[dependencies]
geologia = "0.1"
tokio = { version = "1", features = ["full"] }
```

For streaming, also add:
```toml
tokio-stream = "0.1"
```

## Authentication

`GeminiClient` takes an API key string. The standard pattern is to read it from the environment:

```rust
let api_key = std::env::var("GEMINI_API_KEY")?;
let client = GeminiClient::new(api_key);
```

## Imports

Everything is re-exported from `geologia::prelude`:

```rust
use geologia::prelude::*;
```

---

## Text generation

```rust
use geologia::prelude::{Content, GeminiClient, GenerateContentRequest, Role};

let client = GeminiClient::new(std::env::var("GEMINI_API_KEY")?);

let request = GenerateContentRequest::builder()
    .contents(vec![
        Content::builder()
            .role(Role::User)
            .add_text_part("Explain Rust lifetimes in simple terms.")
            .build(),
    ])
    .build();

let response = client.generate_content(&request, "gemini-2.5-pro").await?;
println!("{}", response.candidates[0].get_text().unwrap());
```

## Streaming text generation

```rust
use geologia::prelude::{Content, GeminiClient, GenerateContentRequest, Role};
use tokio_stream::StreamExt;

let mut stream = client
    .stream_generate_content(&request, "gemini-2.5-pro")
    .await?;

while let Some(chunk) = stream.next().await {
    match chunk {
        Ok(response) => print!("{}", response.candidates[0].get_text().unwrap()),
        Err(err) => eprintln!("Error: {}", err),
    }
}
```

## Multi-turn conversation

Build `contents` as a `Vec<Content>`, alternating `Role::User` and `Role::Model`:

```rust
let contents = vec![
    Content::builder()
        .role(Role::User)
        .add_text_part("Hello!")
        .build(),
    Content::builder()
        .role(Role::Model)
        .add_text_part("Hi! How can I help?")
        .build(),
    Content::builder()
        .role(Role::User)
        .add_text_part("What's 2 + 2?")
        .build(),
];

let request = GenerateContentRequest::builder().contents(contents).build();
```

## System instructions

```rust
let request = GenerateContentRequest::builder()
    .system_instruction(
        Content::builder()
            .add_text_part("You are a terse Rust expert. Be concise.")
            .build(),
    )
    .contents(vec![...])
    .build();
```

## Generation config

```rust
use geologia::prelude::GenerationConfig;

let config = GenerationConfig::builder()
    .temperature(0.7)
    .max_output_tokens(1024)
    .top_p(0.9)
    .build();

let request = GenerateContentRequest::builder()
    .generation_config(config)
    .contents(vec![...])
    .build();
```

---

## Function calling

Declare functions using JSON Schema, then handle the model's `FunctionCall` response:

```rust
use geologia::prelude::{
    Content, FunctionDeclaration, FunctionResponse, GeminiClient,
    GenerateContentRequest, Part, PartData, Role, Tools,
};
use serde_json::json;

let function = FunctionDeclaration {
    name: "get_weather".to_string(),
    description: "Get current weather for a city".to_string(),
    parameters: None,
    parameters_json_schema: Some(json!({
        "type": "object",
        "properties": {
            "city": { "type": "string", "description": "City name" }
        },
        "required": ["city"]
    })),
    response: None,
    response_json_schema: Some(json!({
        "type": "object",
        "properties": {
            "temperature": { "type": "number" },
            "condition": { "type": "string" }
        }
    })),
};

let tools = Tools {
    function_declarations: Some(vec![function]),
    ..Default::default()
};

let mut contents = vec![
    Content::builder()
        .role(Role::User)
        .add_text_part("What's the weather in London?")
        .build(),
];

let request = GenerateContentRequest::builder()
    .contents(contents.clone())
    .tools(vec![tools.clone()])
    .build();

let response = client.generate_content(&request, "gemini-2.5-pro").await?;

// Check if the model wants to call a function
if let Some(candidate) = response.candidates.last()
    && let Some(content) = &candidate.content
    && let Some(parts) = &content.parts
    && let Some(part) = parts.last()
    && let PartData::FunctionCall { id, name, args, .. } = &part.data
{
    // Execute your function, then send back the result
    contents.push(content.clone());
    contents.push(Content {
        role: Some(Role::User),
        parts: Some(vec![Part {
            data: PartData::FunctionResponse(FunctionResponse {
                id: id.clone(),
                name: name.clone(),
                response: json!({ "temperature": 15.0, "condition": "cloudy" }),
                will_continue: None,
                parts: None,
                scheduling: None,
            }),
            media_resolution: None,
            part_metadata: None,
            thought: None,
            thought_signature: part.thought_signature.clone(),
        }]),
    });

    let request = GenerateContentRequest::builder()
        .contents(contents)
        .tools(vec![tools])
        .build();
    let final_response = client.generate_content(&request, "gemini-2.5-pro").await?;
    println!("{}", final_response.candidates[0].get_text().unwrap());
}
```

---

## Image generation (Imagen)

```rust
use geologia::prelude::{
    GeminiClient, PersonGeneration, PredictImageRequest, PredictImageRequestParameters,
    PredictImageRequestParametersOutputOptions, PredictImageRequestPrompt,
    PredictImageSafetySetting,
};

let request = PredictImageRequest {
    instances: vec![PredictImageRequestPrompt {
        prompt: "A tuxedo cat riding a rocket to the moon".to_string(),
    }],
    parameters: PredictImageRequestParameters {
        sample_count: 1,
        aspect_ratio: Some("16:9".to_string()),
        output_options: Some(PredictImageRequestParametersOutputOptions {
            mime_type: Some("image/jpeg".to_string()),
            compression_quality: Some(85),
        }),
        person_generation: Some(PersonGeneration::AllowAdult),
        safety_setting: Some(PredictImageSafetySetting::BlockLowAndAbove),
        ..Default::default()
    },
};

let mut result = client
    .predict_image(&request, "imagen-4.0-generate-001")
    .await?;

let image = result.predictions.pop().unwrap();
// image.bytes_base64_encoded: Vec<u8> — the raw image bytes
// image.mime_type: String — e.g., "image/jpeg"
std::fs::write("output.jpg", &image.bytes_base64_encoded)?;
```

---

## Token counting

```rust
use geologia::prelude::{Content, CountTokensRequest, GeminiClient, Role};

let request = CountTokensRequest {
    contents: vec![
        Content::builder()
            .role(Role::User)
            .add_text_part("How many tokens is this?")
            .build(),
    ],
};

let result = client.count_tokens(&request, "gemini-2.5-pro").await?;
println!("Total tokens: {}", result.total_tokens);
```

---

## Text embeddings

```rust
use geologia::prelude::{GeminiClient, TextEmbeddingRequest, TextEmbeddingRequestContent};

let request = TextEmbeddingRequest {
    content: TextEmbeddingRequestContent {
        parts: vec![/* Part with text */],
    },
    task_type: Some("RETRIEVAL_DOCUMENT".to_string()),
    title: Some("My document title".to_string()),
};

let result = client
    .text_embeddings(&request, "text-embedding-004")
    .await?;
// result.embedding.values: Vec<f32>
```

---

## Structured JSON output

Use `response_mime_type` and `response_schema` in `GenerationConfig` to enforce a JSON schema:

```rust
use geologia::prelude::{GenerationConfig, ResponseSchema};
use serde_json::json;

let schema = ResponseSchema {
    r#type: "object".to_string(),
    properties: Some(json!({
        "name": { "type": "string" },
        "age": { "type": "integer" }
    })),
    ..Default::default()
};

let config = GenerationConfig::builder()
    .response_mime_type("application/json".to_string())
    .response_schema(schema)
    .build();
```

---

## Error handling

All methods return `Result<_, geologia::Error>`. The error type covers HTTP errors, API-level
errors (with status codes and messages from Google), and serialization failures. Use standard
`?` propagation or match on the error for specific handling.

---

## Common model IDs

- `"gemini-2.5-pro"` — most capable text model
- `"gemini-2.5-flash"` — fast and cost-efficient
- `"imagen-4.0-generate-001"` — image generation
- `"text-embedding-004"` — text embeddings
