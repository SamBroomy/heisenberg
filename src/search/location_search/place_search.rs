use crate::FTSIndex;
use anyhow::Result;
use polars::prelude::*;
use tantivy::index;

pub fn place_search(
    term: &str,
    data: LazyFrame,
    index: &FTSIndex,
    previous_result: Option<LazyFrame>,
    limit: Option<usize>,
    all_cols: bool,
    min_importance_tier: Option<u8>,
) -> Result<Option<LazyFrame>> {
    let limit = limit.unwrap_or(100);
    let min_importance_tier = min_importance_tier.unwrap_or(5);

    let join_cols_expr = match &previous_result {
        Some(prev_lf) => {
            //let prev_lf = dbg!(prev_lf.collect()?).lazy();
            get_join_keys(prev_lf).context("Failed to get join keys")?
        }
        None => {
            vec![]
        }
    };
    dbg!(&join_cols_expr);

    todo!()
}
