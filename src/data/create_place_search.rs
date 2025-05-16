use anyhow::Result;
use itertools::multiunzip;
use polars::prelude::*;
use std::collections::HashMap;

fn get_category_features() -> Result<LazyFrame> {
    let categories = [
        ("capitals", vec!["PPLC", "PPLG", "PPLCH"]),
        (
            "major_populated",
            vec!["PPLA", "PPLA2", "PPLA3", "PPLA4", "PPLF", "PPLS"],
        ),
        ("populated", vec!["PPL", "PPLL", "STLMT"]),
        (
            "landmarks",
            vec!["CSTL", "MNMT", "TOWR", "ARCH", "CAVE", "ANS"],
        ),
        ("historic", vec!["MNMT", "RUIN", "HSTS", "ARCH"]),
        (
            "religious",
            vec!["CH", "MSQE", "TMPL", "SHRN", "SYG", "CVNT"],
        ),
        (
            "transportation",
            vec!["AIRP", "RSTN", "BUSTN", "MAR", "PRT", "STNM", "STNR"],
        ),
        ("commercial", vec!["MALL", "MKT", "RECR", "RECG", "SHOP"]),
        ("education", vec!["UNIV", "SCHC"]),
        ("medical", vec!["HSP"]),
        ("facilities", vec!["HTL"]),
        (
            "infrastructure",
            vec![
                "BDG", "DAM", "LOCK", "LTHSE", "BRKW", "PIER", "QUAY", "PRMN", "OILR", "PS", "PSH",
                "PSN", "CTRF",
            ],
        ),
        (
            "nature_reserve",
            vec!["RESN", "RESW", "RESH", "RESF", "RESA"],
        ),
        (
            "water_features",
            vec!["LK", "STM", "BAY", "RSV", "FLLS", "CNYN", "VAL", "GLCR"],
        ),
        (
            "mountain_features",
            vec!["MT", "PK", "VLC", "HLL", "HLLS", "MTS", "PLAT"],
        ),
        ("area", vec!["RGN"]),
        (
            "government",
            vec![
                "ADMF", "GOVL", "CTHSE", "DIP", "BANK", "PO", "PP", "CSTM", "MILB", "INSM",
            ],
        ),
    ];

    let category_weights = HashMap::from([
        ("capitals", 1.0),
        ("major_populated", 0.8),
        ("populated", 0.6),
        ("landmarks", 0.7),
        ("historic", 0.5),
        ("religious", 0.5),
        ("transportation", 0.5),
        ("commercial", 0.3),
        ("education", 0.3),
        ("medical", 0.3),
        ("facilities", 0.2),
        ("infrastructure", 0.15),
        ("nature_reserve", 0.2),
        ("water_features", 0.2),
        ("mountain_features", 0.2),
        ("area", 0.3),
        ("government", 0.4),
    ]);

    let mut category_map = Vec::new();
    for (cat, codes) in categories {
        for code in codes {
            category_map.push((code, cat, category_weights.get(cat).unwrap_or(&0.1)));
        }
    }

    let (feature_code, category_name, category_weight): (Vec<&str>, Vec<&str>, Vec<f64>) =
        multiunzip(category_map);

    Ok(df!(
        "feature_code" => feature_code,
        "category_name" => category_name,
        "category_weight" => category_weight
    )?
    .lazy())
}

fn get_class_defaults() -> Result<LazyFrame> {
    let class_category_weights = vec![
        ("P", "populated", 0.2),
        ("A", "area", 0.2),
        ("S", "facilities", 0.15),
        ("T", "mountain_features", 0.15),
        ("H", "water_features", 0.15),
        ("L", "area", 0.1),
        ("R", "infrastructure", 0.1),
        ("V", "area", 0.1),
        ("U", "area", 0.1),
    ];
    let (feature_class, category_name, category_weight): (Vec<&str>, Vec<&str>, Vec<f64>) =
        multiunzip(class_category_weights);
    Ok(df!(
        "feature_class" => feature_class,
        "category_name" => category_name,
        "category_weight" => category_weight
    )?
    .lazy())
}

fn get_category_features_lf(feature_codes: LazyFrame) -> Result<LazyFrame> {
    let category_features_lf = get_category_features()?;
    let class_defaults_lf = get_class_defaults()?;

    Ok(feature_codes
        .join(
            category_features_lf,
            [col("feature_code")],
            [col("feature_code")],
            JoinArgs {
                how: JoinType::Left,
                ..Default::default()
            },
        )
        .join(
            class_defaults_lf,
            [col("feature_class")],
            [col("feature_class")],
            JoinArgs {
                how: JoinType::Left,
                suffix: Some("_class_default".into()),
                ..Default::default()
            },
        )
        .with_columns([
            coalesce(&[
                col("category_name"),
                col("category_name_class_default"),
                lit("other"),
            ])
            .alias("category_name"),
            coalesce(&[
                col("category_weight"),
                col("category_weight_class_default"),
                lit(0.1),
            ])
            .alias("category_weight"),
        ])
        .select([
            col("feature_class"),
            col("feature_code"),
            col("name"),
            col("description"),
            col("category_name"),
            col("category_weight"),
        ]))
}

pub fn get_place_search_lf(
    all_countries: LazyFrame,
    feature_codes: LazyFrame,
    admin_search: LazyFrame,
) -> Result<LazyFrame> {
    let category_features = get_category_features_lf(feature_codes)?;

    Ok(all_countries
        .join(
            admin_search,
            [col("geonameId")],
            [col("geonameId")],
            JoinArgs {
                how: JoinType::Anti,
                ..Default::default()
            },
        )
        .filter(
            (col("admin0_code").is_not_null())
                .and(col("feature_class").is_in(lit(Series::new(
                    "feature_class_to_keep".into(),
                    ["P", "S", "T", "H", "L", "V", "R"],
                ))))
                .and(
                    col("feature_code")
                        .str()
                        .contains(lit(r"^ADM[A-Z]*|PCL[A-Z]*|TERR"), false)
                        .not(),
                ),
        )
        .with_columns([
            col("population").log1p().alias("pop_log1p"),
            col("alternatenames")
                .list()
                .len()
                .fill_null(0)
                .log1p()
                .alias("name_len_log1p"),
        ])
        .with_columns([
            ((col("pop_log1p") - col("pop_log1p").min())
                / (col("pop_log1p").max() - col("pop_log1p").min()))
            .fill_nan(0.0)
            .alias("pop_boost"),
            ((col("name_len_log1p") - col("name_len_log1p").min())
                / (col("name_len_log1p").max() - col("name_len_log1p").min()))
            .fill_nan(0.0)
            .alias("name_boost"),
        ])
        .join(
            category_features,
            [col("feature_code")],
            [col("feature_code")],
            JoinArgs {
                how: JoinType::Left,
                ..Default::default()
            },
        )
        .with_columns([((col("pop_boost") * lit(0.5))
            + (col("category_weight") * lit(0.3))
            + (col("name_boost") * lit(0.2)))
        .alias("importance_score")])
        .with_columns([(lit(1.0)
            / (lit(1.0)
                + (((col("importance_score") - col("importance_score").mean())
                    / col("importance_score").std(1))
                    * lit(-1.5))
                .exp()))
        .alias("importance_score")])
        .filter(col("importance_score").is_not_null())
        .select([
            col("geonameId"),
            col("name"),
            col("asciiname"),
            col("admin0_code"),
            col("admin1_code"),
            col("admin2_code"),
            col("admin3_code"),
            col("admin4_code"),
            col("feature_class"),
            col("feature_code"),
            col("latitude"),
            col("longitude"),
            col("population"),
            col("elevation"),
            col("alternatenames"),
            col("importance_score"),
        ]))
}
