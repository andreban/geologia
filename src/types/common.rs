use std::{fmt::Display, str::FromStr, vec};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::FunctionResponse;

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct Content {
    pub role: Option<Role>,
    pub parts: Option<Vec<Part>>,
}

impl Content {
    pub fn get_text(&self) -> Option<String> {
        self.parts.as_ref().map(|parts| {
            parts
                .iter()
                .filter_map(|part| match &part.data {
                    PartData::Text(text) => Some(text.clone()),
                    _ => None,
                })
                .collect::<String>()
        })
    }

    pub fn system_prompt<S: Into<String>>(system_prompt: S) -> Self {
        Self::builder().add_text_part(system_prompt).build()
    }

    pub fn builder() -> ContentBuilder {
        ContentBuilder::default()
    }
}

#[derive(Clone, Debug, Default)]
pub struct ContentBuilder {
    content: Content,
}

impl ContentBuilder {
    pub fn add_text_part<T: Into<String>>(self, text: T) -> Self {
        self.add_part(Part::from_text(text.into()))
    }

    pub fn add_part(mut self, part: Part) -> Self {
        match &mut self.content.parts {
            Some(parts) => parts.push(part),
            None => self.content.parts = Some(vec![part]),
        }
        self
    }

    pub fn role(mut self, role: Role) -> Self {
        self.content.role = Some(role);
        self
    }

    pub fn build(self) -> Content {
        self.content
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Model,
}

impl Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let role_str = match self {
            Role::User => "user",
            Role::Model => "model",
        };
        f.write_str(role_str)
    }
}

impl FromStr for Role {
    type Err = ();

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "user" => Ok(Role::User),
            "model" => Ok(Role::Model),
            _ => Err(()),
        }
    }
}

/// See https://ai.google.dev/api/caching#Part
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    // This is of a Struct type, a Map of values, so either a Value or Map<String, Value> are appropriate.
    //See https://protobuf.dev/reference/protobuf/google.protobuf/#struct
    #[serde(skip_serializing_if = "Option::is_none")]
    pub part_metadata: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_resolution: Option<Value>, // TODO: Create type for media_resolution.
    #[serde(flatten)]
    pub data: PartData, // Create enum for data.
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PartData {
    Text(String),
    // https://ai.google.dev/api/caching#Blob
    InlineData {
        mime_type: String,
        data: String,
    },
    // https://ai.google.dev/api/caching#FunctionCall
    FunctionCall {
        id: Option<String>,
        name: String,
        args: Option<Value>,
    },
    // https://ai.google.dev/api/caching#FunctionResponse
    FunctionResponse(FunctionResponse),
    FileData(Value),
    ExecutableCode(Value),
    CodeExecutionResult(Value),
}

impl Part {
    pub fn from_text<S: Into<String>>(text: S) -> Self {
        Self {
            thought: None,
            thought_signature: None,
            part_metadata: None,
            media_resolution: None,
            data: PartData::Text(text.into()),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::types::Part;

    #[test]
    pub fn parses_text_part() {
        let input = r#"
        {
          "text": "**What do you mean? An African or a European swallow?**\n\nIf you are looking for the actual physics rather than the *Monty Python and the Holy Grail* reference, here is the breakdown:\n\n**1. The European Swallow**\nBased on an analysis published by Jonathan Corum (using data on the Strouhal number of cruising flight), the estimated airspeed velocity of an unladen European Swallow is roughly **11 meters per second**, or **24 miles per hour**.\n\n**2. The African Swallow**\nData on the African swallow is scarcer, mostly because—as the guard in the movie points out—African swallows are non-migratory. However, since they are similar in size to their European counterparts, their cruising speed would likely be comparable.\n\nBut of course, the real question is: *Could it carry a coconut?* (A five-ounce bird could not carry a one-pound coconut. It is a simple question of weight ratios.)",
          "thoughtSignature": "EqcZCqQZAdHtim/53UNFI7YRLcEDch1I/mLfWNT6lVjgXb7RsNnYn8JLU8Y6UhAi4nkLJ/nK2l44Y+JJZimQ2rLpRfdlBAPkhVsuZYenAY7MRXG9GQrSzz1elR+L6FAb0dyb9snnGz5NdlKCyS9VIWKIhghmHA60oEnEUexaJD2mq3ZV4kJ8R/d+UJEEdOD9CdlnB1WnOvHaiT15mLSj8JxclI+1mml86b5hjA0F+MLVWesa4gjo6/OfNo1k+tA+JioUAu8hgZ5DJttNxs/BvrLMyY/+d6qm40Ht45BuNlKUjFTkrUOIx5oAld3PnNj804Ou3F/sv8i5UMh9TcWyuiOjP3lZU5t1GEKQJ/YY9CxN/Zl71Kzk51Z+92IV2tKLqZVsEkrIr5o33QmNRTIeX0zMSQRdhlTBPuwSa+l91SV56cPK0I7P6UPguc3qGD8E3wfUC+fByDzX4JZ6OuhyrwcCCgbyjnBgI/FoWBA364cKONEH69p851Jy+zRaI9hWKKOQ/hqHqpWL266vgnALkvjcfZS3Frc6rRTvRIzetVufrJM3i9OAfnoLPZz5crraRQgUpgcPUd9fYhl59PIK35jRaENXunDUa8NE/J8kObcZE+910NxsUo7LzsGssr6UOPM6slKhnocnbqCrrNLhoF0jLXbSObuCXKh5HuGV8Y51UdsK6oUuct+ScfOZGBl+/6LhaGmlS0Ab58R7CO8UqhX4j91H8YW6xtDTQoAIXNU2j4Zq7lkpH0b5Vv7ZhFnbbc1OgTtboTcKwyRXgZFlBa6NNIb7GvRMyKdWW+sHXFAXGohZubp7DXsr6gQ/8eqcTuiiLKChRbY6MhG14OkGw4/LcuBAxEg6Fy7JX3tlMfto3LcfhFVvlmM1XuWACR9OJLr49YAkBYsMWl95qK5tSG0Wo/hAqjcPWPszrzK9Uo9AsDpsCHGnX57Ytcsi60y+jnV7iQqhoWtaT+UJW9FbxOPpKTsQw0k2GPM/1d+ulMz2IYPrN/Bsuk34OyAUID1zEUnSro0Q4camHfW2wnJvW77rLmfqO2b0M4+UuEgbgB/dyQtICsNndaO1x6S3pL8/typqoakwx/9xg02QVzLLRvfs4Su9eSAsKL/QfQCI9dmS8O0kvA1DqbUdxO6HfrfCVpGKoLajB4dZ/1nplNFFL+ap7vXOU9F4foXemT4f71T3S93NWb6gFU8jB8WxNaoWVBoeuP7iJNMqqBZPvV9SJ94lELlV/LZKlZ+pqQML/Gfe565AmXD34ekgE5ZGkwQxSoP8BksbDnL41GxEZtvWHcr+kSZK2FoTBwsXBye43qy1ZFYV+guSPqgsy5S215c2r4g+zfJ2vlC5+k2621Dwex7POA68LrtfbyeFJ8gQY7nZMPNp2gZQHmY/imA1Fb0jiCfMzYUiWumJeyOeiSUE5p/slwV0SryaYtT73fjx37F/iUAE5zl6yEo8v45aiB2XNgxdTU4bjHEFD+sj/6DGp27ukt6vLxN/QhmPvU7yYUA+u1WbQblof6VN7AwhVUqgqUx9Je0kSXPrI12K/2yC6eZnGuXeicqwIxCQWh9z9o24NzUkaiVC7VnSItVgXDWwviwAe4H1LxNU9y6j+Y0R8iGclRQVN8haBc1x7BWO6raGsLRrKblykBsIydnuz1Bvjk4eEaoH1rCzzIiuj1ZqG3bo/bLxjJw1h1KmnXkywo8alCusMIog71a3FQnST+idwJ9+tJU31rqMxinD1kUwG5ZYmFnpRZWHD57gsa5rzFptjbnkUxfBhHD3+7mO6qlgMidjzfv77MuFWRVyglDMD+eNvlX6vmPm93Qq4rDZTDssck6IYCaQ6TuqXJ2WEal0HDgaX/rlyhUL/4T7Ptk2/QoQqekUasvbjPhpn25R9AGTIcEwdoVsK2kC4ftvtkc2g1jE4PK2fLqe6sNfCEebZT18nx5FdgELbkSB+ss3aLfvWVVC0EJJmdlW+F1mxxPnkfvwcCfj4YKsfhEMoiPxbs0As2dtbaV9xcrhFlGZFoA/idudJqRPEuZvhtiJ2L0MQMuDWqT6kDr6wqnAghj2olacMb9rU5IlK9hfoCalMp7/adEJLpzJ7RdZd6o8cGq0D2v9lsT/2OJtq+kiMIG3gzIDrHSCK7v3XFpmA6DcMsgUHyYGSe1Mfe6fD+mPXyKWEi+hp3SJjDHa3Xk0bx5java0fZc/q/t9yxxjijIVGlRrduMj0GQpi3JHOL/JZoGWHrMSQFBmLIEypj+Dp1nImOja7j69VlK6q1dxELdx1sE5eIzTpk0/bRZ3oyqFtXYwyWUJsx5evdJSPIGbM8lgQsV8yO9U8LRot2BhWyfsU8NWRsHY5ihYb2K/Y9saE1iML4uqvIAK36eG9DuRaz2zIa6K3G5Xr/U8c0BxUxNNcWIra7TPyVmIXhLm85ghX9qKWNM2YQO/02tvIAI/9+8qANblayjg31j+FjME1NNGQg3jxA28QyfN39b0Fg8sD5MWmHP6MtvfVwx0JM88n1eCJiZ0No5BFUOB/EfgtiXp48ledg66cLjPmU9rjKPNyK4iUsRO7IY9X0/7L4M+d+8tBOy14Bfjn0ELi6HdF5+HVgWp3DViCn8iX4HCVrTX9S4/ZrgJVDJdI5axuGlsaH3VqCV0Rfes/p3MfcjUVOpBja+byTWMbM0ZONjrF3NAtzwZwLN+QDVEVS8Hso11mYsL6IvEbKsGYySBcX6qZ57p0MlPeC0GPPy0DkDca19W/fWFkrlPP60plNymq+c9HZ1Ghmg9YSGluckJLidqR6wuCSSkyaSwjJaJYnu4MIfXrLP4Q0UmKwvVJFSNqhtDSaus+U2+m8sl6CadTs4trw2iVh78/Wpghvido18f7A40MFo8E3OLN9XEgXA2FLMPrGiZM3JFTMutokburAgTAxs7CmbqilP4ArWvxEvG+TbmCatA5PhhGibms3OO910cjaToRUXriE8K7kHRM7Miui7qDcCM+wcgPOV+sYNNucAAbseGi+Mej1tmMLTUO4k8q2bRcadMaijASasX6Q8k6k1YGy89HTh1UkwCLdd6F4eYHsDFpMGjwJ2I1fJ/4lmTAUYOHP3n4p4ovOSoptgIul9sty7iqZnQlkQHeVWQSwMzyBbcxTqA6GDsdNk5GF+Wjaf3C3F+uOhRY+yD0wbb43d3rpEMPkThbTTsN8ricg0bDSIWnM2FKfsQ0QFbZuC2JrkeSEZuLd3RldLsUXBzrQl2ub49oztmjEQSu6GePyz9LAeQRJd6EUQ4/I/vu1SLyHcXZAch4zrzk2u+7OWehE+i/CGzRWL14/x+z3PPmguYOqS1rJdCWDIKlIXD9nZc/heFhQ4QiV2pvr0ElYHCDnAq/SgpPC7EFy4BGmz6cMJ2Az44cijzOFbYZ1+rkbxvLV4Q2QVDj5tgBNYrV7FYBs+B0kF3D/ijbp1JGowGDsXJC1KaUpu01OL9962042O3b4RIU6NsGa0irMip/IAlFYhEW72Aj6oNvqNKDf7VjT3GYvRRz51zPMaKymBLCDw2lSrz7tTkN8L3w7dyLzBpzNI894Id3B6lf+ummAp+w0y0Q/jQnNzUFJznXIoais7JwcC+jxkolAW6iCXwGYGbYLTgV1jKH1GdJf10yzMo/obPF2F4vtRITmq3PGRV1DEm9ELbu3ajhSP4vh9eUqxki/ORrJibn6MVBz1GtzOzFBZ8br2ZZLCxqq4bTVj/BXPngVZ6bxmxdn7rf15a4IcPRZ9hPEl/M3vIl6cSJLb3M45lADfDtBW70dXMFAcof2ipkngcOf2NY/dYuGUMMyOp/Xetvy4kFY2ye2nU0PEq0GhwxxCB/zrGzxprC7W93sVETAlPXyb9yirlo4elyaNIZMt+sqlUHGoFyK3xDPlkNAwrsQWgghNwMrtZ7Fm683n38X9HVwgGjJpeoODfIph+f0vDl+ncO2GywdSbJXQg5Tf5PTONvZb+8Kd2F7Lv8mljtqAKHoh/b7MyogGvA914hUL3jKFClnAaD9xXWCK83stRL2Hqg2PmY+aNwB3m/Y54QEYdq+Xu7nIWo8EkncKTB4GwLb7Cyep88E5WNnyaU1Y337xAEGN9403pqCp+abrFgMOLl1MAPoWXNEGsQIEqVJECkPgpdR1eU83LjPXjSthCe5mo2Vc35IgOOA94UEDfaXyRqQEE5CH+QRdXCc4oMKt3cTUHiPlPbHayKVH5d1lntDxMgJ5tSN1kwcFQMKYXJdYSZqatoYNar0tnSF2EuGPs2ium1h4Il/NKCPiZySDbYRwDITMu+RVMvr5CbmXHF93bz/d0n8Qg8A2qmrU="
        }"#;

        let _ = serde_json::from_str::<Part>(input).unwrap();
    }
}
