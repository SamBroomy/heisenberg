use heisenberg_data_processing::{
    DataSource, processed::generate_processed_data, raw::fetch::download_data,
};
use tracing_subscriber::fmt::format::FmtSpan;

fn main() {
    tracing_subscriber::fmt::fmt()
        .with_env_filter("debug")
        .with_span_events(FmtSpan::CLOSE)
        .init();
    let temp_files = download_data(&DataSource::Cities15000).unwrap();
    let (admin, places) = generate_processed_data(temp_files).unwrap();

    println!("{admin:}");
    println!();
    println!("{places:#?}");
}
