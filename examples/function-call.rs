use std::{env, error::Error};

use google_genai::prelude::{
    Content, FunctionDeclaration, FunctionResponse, GeminiClient, GenerateContentRequest, Part,
    PartData, Role, Tools,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt().init();
    let _ = dotenvy::dotenv();

    let api_key = env::var("GEMINI_API_KEY")?;
    let gemini_client = GeminiClient::new(api_key);

    let mut contents = vec![
        Content::builder()
            .role(Role::User)
            .add_text_part("What is the sbrubbles value of 213 and 231?")
            .build(),
    ];

    let sum_parameters = json!({
        "type": "object",
        "properties": {
            "left": {
                "type": "integer",
                "description": "The first value for the sbrubbles calculation"
            },
            "right": {
                "type": "integer",
                "description": "The second value for the sbrubbles calculation"
            }
        },
        "required": ["left", "right"]
    });

    let sum_result = json!({
        "type": "object",
        "properties": {
            "result": {
                "type": "integer",
                "description": "The sbrubbles value calculation result",
            }
        }
    });

    let sum_function = FunctionDeclaration {
        name: String::from("sbrubbles"),
        description: String::from("Calculates the sbrubbles value"),
        parameters: None,
        parameters_json_schema: Some(sum_parameters),
        response: None,
        response_json_schema: Some(sum_result),
    };

    println!("{}", serde_json::to_string_pretty(&sum_function).unwrap());

    let tools = Tools {
        function_declarations: Some(vec![sum_function]),
        ..Default::default()
    };

    let request = GenerateContentRequest::builder()
        .contents(contents.clone())
        .tools(vec![tools.clone()])
        .build();

    let mut response = gemini_client
        .generate_content(&request, "gemini-3-pro-preview")
        .await?;

    while let Some(candidate) = response.candidates.last()
        && let Some(content) = &candidate.content
        && let Some(parts) = &content.parts
        && let Some(part) = parts.last()
        && let PartData::FunctionCall { id, name, args, .. } = &part.data
    {
        contents.push(content.clone());
        match args {
            Some(args) => println!("Function call: {name}, {args}"),
            None => println!("Function call: {name}"),
        }

        contents.push(Content {
            role: Some(Role::User),
            parts: Some(vec![Part {
                data: PartData::FunctionResponse(FunctionResponse {
                    id: id.clone(),
                    name: name.clone(),
                    response: json!({"result": 1234}),
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
            .contents(contents.clone())
            .tools(vec![tools.clone()])
            .build();

        println!("{contents:?}");
        response = gemini_client
            .generate_content(&request, "gemini-3-pro-preview")
            .await?;
    }

    println!("Response: {:#?}", response.candidates);
    Ok(())
}
