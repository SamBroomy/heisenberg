use anyhow::Result;
use dotenv::dotenv;
use serde_json::json;
use twitter_v2::authorization::{BearerToken, Oauth2Token};
use twitter_v2::query::{TweetField, UserField};
use twitter_v2::TwitterApi;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Hello, world!");

    let stored_oauth2_token = std::fs::read_to_string(".token.json")?;

    let auth: Oauth2Token = serde_json::from_str(&stored_oauth2_token)?;

    let my_followers = TwitterApi::new(auth)
        .with_user_ctx()
        .await?
        .get_my_followers()
        .user_fields([UserField::Username])
        .max_results(20)
        .send()
        .await?
        .into_data();

    println!("My followers: {:#?}", my_followers);

    Ok(())
}
