// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;

use super::*;

type Result<T> = core::result::Result<T, Box<dyn std::error::Error>>;

#[test]
fn serialize_and_deserialize_full_decision_request_and_response() -> Result<()> {
    let request = CreateDecisionRequest::builder()
        .model("gemini-decision-1")
        .text_input("Kitchen counter has a puddle of spilled coffee near the edge.")
        .add_noul_question(
            "has_spill",
            NoulQuestion::new("Is there an uncleaned liquid spill?")
                .true_description("Visible liquid puddle or wet spill on the surface")
                .false_description("Surface is dry and clean"),
        )
        .add_choice_question(
            "urgency",
            ChoiceQuestion::new(
                "What is the cleanup urgency?",
                [
                    ("low", "Cosmetic smudge only"),
                    ("medium", "Small contained spill"),
                    ("high", "Slip or electrical hazard"),
                ],
            ),
        )
        .add_score_question(
            "cleanliness_score",
            ScoreQuestion::new(
                "Rate the cleanliness of the counter.",
                [
                    "Severely soiled with active spills",
                    "Minor crumbs or small spill",
                    "Spotless and sanitized",
                ],
            ),
        )
        .build();

    let serialized = serde_json::to_value(&request)?;
    assert_eq!(serialized["model"], "gemini-decision-1");
    assert_eq!(
        serialized["input"],
        "Kitchen counter has a puddle of spilled coffee near the edge."
    );
    assert_eq!(serialized["questions"]["has_spill"]["type"], "noul");
    assert_eq!(
        serialized["questions"]["has_spill"]["instructions"],
        "Is there an uncleaned liquid spill?"
    );
    assert_eq!(
        serialized["questions"]["has_spill"]["criteria"]["true_description"],
        "Visible liquid puddle or wet spill on the surface"
    );
    assert_eq!(serialized["questions"]["urgency"]["type"], "choice");
    assert_eq!(
        serialized["questions"]["urgency"]["criteria"]["medium"],
        "Small contained spill"
    );
    assert_eq!(
        serialized["questions"]["cleanliness_score"]["type"],
        "score"
    );
    assert_eq!(
        serialized["questions"]["cleanliness_score"]["criteria"],
        json!([
            "Severely soiled with active spills",
            "Minor crumbs or small spill",
            "Spotless and sanitized"
        ])
    );

    let round_trip: CreateDecisionRequest = serde_json::from_value(serialized)?;
    assert_eq!(round_trip, request);

    let response_json = json!({
        "answers": {
            "has_spill": {
                "type": "noul",
                "noul": 0.96,
                "confidence": 0.96
            },
            "urgency": {
                "type": "choice",
                "choice": "medium",
                "confidence": 0.78,
                "probabilities": {
                    "low": 0.07,
                    "medium": 0.78,
                    "high": 0.15
                }
            },
            "cleanliness_score": {
                "type": "score",
                "score": 0.85,
                "confidence": 0.75,
                "probabilities": {
                    "0": 0.20,
                    "1": 0.75,
                    "2": 0.05
                },
                "legend": {
                    "0": "Severely soiled with active spills",
                    "1": "Minor crumbs or small spill",
                    "2": "Spotless and sanitized"
                }
            }
        },
        "usage": {
            "total_input_tokens": 120,
            "total_output_tokens": 18,
            "total_tokens": 138,
            "input_tokens_by_modality": [
                { "modality": "text", "tokens": 120 }
            ]
        }
    });

    let response: CreateDecisionResponse = serde_json::from_value(response_json.clone())?;

    let noul = response.get_noul("has_spill").ok_or("missing has_spill")?;
    assert_eq!(noul.noul, 0.96);
    assert_eq!(noul.confidence, 0.96);
    assert_eq!(
        response.get_answer("has_spill").map(Answer::confidence),
        Some(0.96)
    );

    let choice = response.get_choice("urgency").ok_or("missing urgency")?;
    assert_eq!(choice.choice, "medium");
    assert_eq!(choice.confidence, 0.78);
    assert_eq!(choice.probabilities.get("medium"), Some(&0.78));

    let score = response
        .get_score("cleanliness_score")
        .ok_or("missing cleanliness_score")?;
    assert_eq!(score.score, 0.85);
    assert_eq!(score.confidence, 0.75);
    assert_eq!(
        score.legend.get("1").map(String::as_str),
        Some("Minor crumbs or small spill")
    );

    let usage = response.usage.as_ref().ok_or("missing usage")?;
    assert_eq!(usage.total_input_tokens, Some(120));
    assert_eq!(usage.total_output_tokens, Some(18));
    assert_eq!(usage.total_tokens, Some(138));
    assert_eq!(
        usage.input_tokens_by_modality,
        Some(vec![DecisionModalityTokens {
            modality: Some(DecisionModality::Text),
            tokens: Some(120),
        }])
    );

    let reserialized = serde_json::to_value(&response)?;
    assert_eq!(reserialized, response_json);
    Ok(())
}

#[test]
fn multimodal_input_and_structured_instructions_round_trip() -> Result<()> {
    let request = CreateDecisionRequest::builder()
        .model("gemini-decision-1")
        .text_input("Inspect the following media items:")
        .add_input_content(DecisionInputContent::image_bytes(b"png-bytes", "image/png"))
        .add_input_content(DecisionInputContent::Video(DecisionVideoContent {
            uri: Some("https://generativelanguage.googleapis.com/v1beta/files/vid-1".into()),
            mime_type: Some("video/mp4".into()),
            name: Some("clip_1".into()),
            processing: Some(DecisionVideoProcessing::StaticConfig(
                StaticMediaProcessing::new()
                    .with_start_offset("0s")
                    .with_end_offset("5s")
                    .with_fps(2.0),
            )),
            resolution: Some(DecisionMediaResolution::High),
            ..Default::default()
        }))
        .add_input_content(DecisionInputContent::audio_uri(
            "https://generativelanguage.googleapis.com/v1beta/files/aud-1",
            "audio/wav",
        ))
        .add_input_content(DecisionInputContent::document_uri(
            "https://generativelanguage.googleapis.com/v1beta/files/doc-1",
            "application/pdf",
        ))
        .add_question(
            "q1",
            Question::Noul(NoulQuestion::new(vec![DecisionInstructionContent::text(
                "Does the clip match the document?",
            )])),
        )
        .build();

    let value = serde_json::to_value(&request)?;
    assert_eq!(value["input"][0]["type"], "text");
    assert_eq!(value["input"][1]["type"], "image");
    assert_eq!(value["input"][1]["data"], "cG5nLWJ5dGVz");
    assert_eq!(value["input"][2]["type"], "video");
    assert_eq!(value["input"][2]["processing"]["type"], "static");
    assert_eq!(value["input"][2]["processing"]["fps"], 2.0);
    assert_eq!(value["input"][2]["resolution"], "high");
    assert_eq!(value["input"][3]["type"], "audio");
    assert_eq!(value["input"][4]["type"], "document");
    assert_eq!(
        value["questions"]["q1"]["instructions"][0],
        json!({
            "type": "text",
            "text": "Does the clip match the document?"
        })
    );

    let decoded: CreateDecisionRequest = serde_json::from_value(value)?;
    assert_eq!(decoded, request);
    Ok(())
}

#[test]
fn prompt_feedback_and_annotations_round_trip() -> Result<()> {
    let text_with_annotations = DecisionTextContent::new("Cited statement").with_annotations(vec![
        DecisionAnnotation::UrlCitation(DecisionUrlCitation {
            start_index: Some(0),
            end_index: Some(15),
            title: Some("Example".into()),
            url: Some("https://example.com".into()),
        }),
        DecisionAnnotation::WordInfo(DecisionWordInfo {
            word: Some("Cited".into()),
            start_offset: Some(0.1),
            end_offset: Some(0.4),
            start_index: Some(0),
            end_index: Some(5),
        }),
    ]);

    let content = DecisionInputContent::Text(text_with_annotations);
    let json_val = serde_json::to_value(&content)?;
    assert_eq!(json_val["annotations"][0]["type"], "url_citation");
    assert_eq!(json_val["annotations"][1]["type"], "word_info");

    let parsed_content: DecisionInputContent = serde_json::from_value(json_val)?;
    assert_eq!(parsed_content, content);

    let feedback_resp: CreateDecisionResponse = serde_json::from_value(json!({
        "prompt_feedback": {
            "block_reason": "safety",
            "block_reason_message": "Prompt blocked by safety filter"
        }
    }))?;

    assert_eq!(
        feedback_resp.prompt_feedback,
        Some(DecisionPromptFeedback {
            block_reason: Some(DecisionBlockReason::Safety),
            block_reason_message: Some("Prompt blocked by safety filter".into()),
        })
    );
    Ok(())
}

#[test]
fn strict_deserialization_rejects_unknown_question_and_answer_types() -> Result<()> {
    let unknown_question = json!({
        "type": "freeform",
        "instructions": "Explain why"
    });
    assert!(serde_json::from_value::<Question>(unknown_question).is_err());

    let unknown_answer = json!({
        "type": "freeform",
        "text": "Because..."
    });
    assert!(serde_json::from_value::<Answer>(unknown_answer).is_err());

    let malformed_noul_answer = json!({
        "type": "noul",
        "noul": 0.8
        // missing required `confidence`
    });
    assert!(serde_json::from_value::<Answer>(malformed_noul_answer).is_err());

    let malformed_input_block = json!({
        "type": "text"
        // missing required `text`
    });
    assert!(serde_json::from_value::<DecisionInput>(malformed_input_block).is_err());

    let unknown_input_block = json!({
        "type": "hologram",
        "uri": "files/123"
    });
    assert!(serde_json::from_value::<DecisionInput>(unknown_input_block).is_err());
    Ok(())
}
