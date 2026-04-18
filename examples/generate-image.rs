// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use std::{env, error::Error, io::Cursor};

use geologia::prelude::{
    GeminiClient, PersonGeneration, PredictImageRequest, PredictImageRequestParameters,
    PredictImageRequestParametersOutputOptions, PredictImageRequestPrompt,
    PredictImageSafetySetting,
};
use image::{ImageFormat, ImageReader};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt().init();
    let _ = dotenvy::dotenv();

    let api_key = env::var("GEMINI_API_KEY")?;
    let gemini_client = GeminiClient::new(api_key);

    let prompt = "
    Create an image of a tuxedo cat riding a rocket to the moon.";
    let request = PredictImageRequest {
        instances: vec![PredictImageRequestPrompt {
            prompt: prompt.to_string(),
        }],
        parameters: PredictImageRequestParameters {
            sample_count: 1,
            aspect_ratio: Some("1:1".to_string()),
            output_options: Some(PredictImageRequestParametersOutputOptions {
                mime_type: Some("image/jpeg".to_string()),
                compression_quality: Some(75),
            }),
            person_generation: Some(PersonGeneration::AllowAdult),
            safety_setting: Some(PredictImageSafetySetting::BlockLowAndAbove),
            ..Default::default()
        },
    };

    println!("Request: {:#?}", serde_json::to_string(&request).unwrap());

    let mut result = gemini_client
        .predict_image(&request, "imagen-4.0-generate-001")
        .await?;

    let result = result.predictions.pop().unwrap();

    let format = ImageFormat::from_mime_type(result.mime_type).unwrap();
    let img =
        ImageReader::with_format(Cursor::new(result.bytes_base64_encoded), format).decode()?;
    img.save("output.jpg")?;
    Ok(())
}
