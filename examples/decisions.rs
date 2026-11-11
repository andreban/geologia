// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use std::{env, error::Error};

use geologia::prelude::{
    ChoiceQuestion, CreateDecisionRequest, GeminiClient, NoulQuestion, ScoreQuestion,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt().init();
    let _ = dotenvy::dotenv();

    let api_key = env::var("GEMINI_API_KEY")?;
    let gemini_client = GeminiClient::new(api_key);

    let request = CreateDecisionRequest::builder()
        .model("gemini-decision-1")
        .text_input(
            "Customer support ticket #1084: 'I received my espresso machine yesterday, \
             but the water reservoir arrived cracked and leaks all over the counter when filled. \
             I need a replacement sent as soon as possible.'",
        )
        .add_noul_question(
            "is_hardware_defect",
            NoulQuestion::new("Does the customer report a physical or functional product defect?")
                .true_description("Customer reports damaged, leaking, or non-working hardware")
                .false_description("Customer asks a general question or billing inquiry"),
        )
        .add_choice_question(
            "routing_queue",
            ChoiceQuestion::new(
                "Which support team should handle this ticket?",
                [
                    ("billing", "Payment, refund status, or invoice inquiries"),
                    (
                        "warranty_replacement",
                        "Defective or damaged hardware requiring replacement",
                    ),
                    (
                        "general_support",
                        "Product usage, recipes, or setup questions",
                    ),
                ],
            ),
        )
        .add_score_question(
            "urgency_score",
            ScoreQuestion::new(
                "Rate the urgency of resolving this customer ticket.",
                [
                    "Low urgency: general feedback or feature question",
                    "Medium urgency: minor inconvenience, product still usable",
                    "High urgency: product unusable or leaking out of the box",
                ],
            ),
        )
        .build();

    let response = gemini_client.create_decision(&request).await?;

    if let Some(defect) = response.get_noul("is_hardware_defect") {
        println!(
            "is_hardware_defect -> P(true): {:.2}, confidence: {:.2}",
            defect.noul, defect.confidence
        );
    }

    if let Some(queue) = response.get_choice("routing_queue") {
        println!(
            "routing_queue      -> choice: {:?}, confidence: {:.2}, probabilities: {:?}",
            queue.choice, queue.confidence, queue.probabilities
        );
    }

    if let Some(urgency) = response.get_score("urgency_score") {
        println!(
            "urgency_score      -> expected score: {:.2}, confidence: {:.2}, probabilities: {:?}",
            urgency.score, urgency.confidence, urgency.probabilities
        );
    }

    if let Some(usage) = &response.usage {
        println!("Token usage        -> {:?}", usage);
    }

    Ok(())
}
