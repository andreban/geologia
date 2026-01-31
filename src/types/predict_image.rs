use serde::{Deserialize, Serialize};
use serde_with::base64::Base64;
use serde_with::serde_as;

/// Request body for the Imagen image generation `predict` endpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictImageRequest {
    pub instances: Vec<PredictImageRequestPrompt>,
    pub parameters: PredictImageRequestParameters,
}

/// A text prompt instance for image generation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictImageRequestPrompt {
    /// The text prompt for the image.
    /// The following models support different values for this parameter:
    ///  - `imagen-3.0-generate-001`: up to 480 tokens.
    ///  - `imagen-3.0-fast-generate-001`: up to 480 tokens.
    ///  - `imagegeneration@006`: up to 128 tokens.
    ///  - `imagegeneration@005`: up to 128 tokens.
    ///  - `imagegeneration@002`: up to 64 tokens.
    pub prompt: String,
}

/// Parameters controlling image generation behavior.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PredictImageRequestParameters {
    /// The number of images to generate. The default value is 4.
    /// The following models support different values for this parameter:
    ///  - `imagen-3.0-generate-001`:  1 to 4.
    ///  - `imagen-3.0-fast-generate-001`:  1 to 4.
    ///  - `imagegeneration@006`: 1 to 4.
    ///  - `imagegeneration@005`: 1 to 4.
    ///  - `imagegeneration@002`: 1 to 8.
    pub sample_count: i32,

    /// The random seed for image generation. This is not available when addWatermark is set to
    /// true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,

    /// Optional. An optional parameter to use an LLM-based prompt rewriting feature to deliver
    /// higher quality images that better reflect the original prompt's intent. Disabling this
    /// feature may impact image quality and prompt adherence
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enhance_prompt: Option<bool>,

    /// A description of what to discourage in the generated images.
    /// The following models support this parameter:
    ///  - `imagen-3.0-generate-001`: up to 480 tokens.
    ///  - `imagen-3.0-fast-generate-001`: up to 480 tokens.
    ///  - `imagegeneration@006`: up to 128 tokens.
    ///  - `imagegeneration@005`: up to 128 tokens.
    ///  - `imagegeneration@002`: up to 64 tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,

    /// The aspect ratio for the image. The default value is "1:1".
    /// The following models support different values for this parameter:
    ///  - `imagen-3.0-generate-001`: "1:1", "9:16", "16:9", "3:4", or "4:3".
    ///  - `imagen-3.0-fast-generate-001`: "1:1", "9:16", "16:9", "3:4", or "4:3".
    ///  - `imagegeneration@006`: "1:1", "9:16", "16:9", "3:4", or "4:3".
    ///  - `imagegeneration@005`: "1:1" or "9:16".
    ///  - `imagegeneration@002`: "1:1".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,

    /// Describes the output image format in an `PredictImageRequestParametersOutputOptions
    /// object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_options: Option<PredictImageRequestParametersOutputOptions>,

    /// Describes the style for the generated images. The following values are supported:
    ///  - "photograph"
    ///  - "digital_art"
    ///  - "landscape"
    ///  - "sketch"
    ///  - "watercolor"
    ///  - "cyberpunk"
    ///  - "pop_art"
    ///
    /// Pre-defined styles is only supported for model imagegeneration@002
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_image_style: Option<String>,

    /// Allow generation of people by the model. The following values are supported:
    ///  - `"dont_allow"`: Disallow the inclusion of people or faces in images.
    ///  - `"allow_adult"`: Allow generation of adults only.
    ///  - `"allow_all"`: Allow generation of people of all ages.
    ///
    /// The default value is `"allow_adult"`.
    ///
    /// Supported by the models `imagen-3.0-generate-001`, `imagen-3.0-fast-generate-001`, and
    /// `imagegeneration@006` only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub person_generation: Option<PersonGeneration>,

    /// Optional. The language code that corresponds to your text prompt language.
    /// The following values are supported:
    ///  - auto: Automatic detection. If Imagen detects a supported language, the prompt and an
    ///    optional negative prompt are translated to English. If the language detected isn't
    ///    supported, Imagen uses the input text verbatim, which might result in an unexpected
    ///    output. No error code is returned.
    ///  - en: English (if omitted, the default value)
    ///  - zh or zh-CN: Chinese (simplified)
    ///  - zh-TW: Chinese (traditional)
    ///  - hi: Hindi
    ///  - ja: Japanese
    ///  - ko: Korean
    ///  - pt: Portuguese
    ///  - es: Spanish
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// Adds a filter level to safety filtering. The following values are supported:
    ///
    /// - "block_low_and_above": Strongest filtering level, most strict blocking.
    ///   Deprecated value: "block_most".
    /// - "block_medium_and_above": Block some problematic prompts and responses.
    ///   Deprecated value: "block_some".
    /// - "block_only_high": Reduces the number of requests blocked due to safety filters. May
    ///   increase objectionable content generated by Imagen. Deprecated value: "block_few".
    /// - "block_none": Block very few problematic prompts and responses. Access to this feature
    ///   is restricted. Previous field value: "block_fewest".
    ///
    /// The default value is "block_medium_and_above".
    ///
    /// Supported by the models `imagen-3.0-generate-001`, `imagen-3.0-fast-generate-001`, and
    /// `imagegeneration@006` only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_setting: Option<PredictImageSafetySetting>,

    /// Add an invisible watermark to the generated images. The default value is `false` for the
    /// `imagegeneration@002` and `imagegeneration@005` models, and `true` for the
    /// `imagen-3.0-fast-generate-001`, `imagegeneration@006`, and imagegeneration@006 models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub add_watermark: Option<bool>,

    /// Cloud Storage URI to store the generated images.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage_uri: Option<String>,
}

/// Output format options for generated images.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PredictImageRequestParametersOutputOptions {
    /// The image format that the output should be saved as. The following values are supported:
    ///
    /// - "image/png": Save as a PNG image
    /// - "image/jpeg": Save as a JPEG image
    ///
    /// The default value is "image/png".v
    pub mime_type: Option<String>,

    /// The level of compression if the output type is "image/jpeg".
    /// Accepted values are 0 through 100. The default value is 75.
    pub compression_quality: Option<i32>,
}

/// A successful response from the Imagen `predict` endpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictImageResponse {
    pub predictions: Vec<PredictImageResponsePrediction>,
}

/// A single generated image from the prediction response.
#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PredictImageResponsePrediction {
    #[serde_as(as = "Base64")]
    pub bytes_base64_encoded: Vec<u8>,
    pub mime_type: String,
}

/// Controls whether generated images may include people.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PersonGeneration {
    DontAllow,
    AllowAdult,
    AllowAll,
}

/// Safety filter level for image generation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictImageSafetySetting {
    BlockLowAndAbove,
    BlockMediumAndAbove,
    BlockOnlyHigh,
    BlockNone,
}
