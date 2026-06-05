use sglang_srt::cli::ServerArgs;
use sglang_srt::server::{launch_grpc_server, launch_http_server};
use std::io::Write as _;

#[tokio::main]
async fn main() {
    match ServerArgs::parse_from(std::env::args().skip(1)) {
        Ok(args) => {
            if args.grpc_mode {
                println!(
                    "sglang-rs serve grpc: model_path={} host={} port={} tp_size={} dp_size={} grpc_mode=true",
                    args.model_path, args.host, args.port, args.tp_size, args.dp_size
                );
                std::io::stdout().flush().expect("stdout should flush");
                if let Err(error) = launch_grpc_server(args).await {
                    eprintln!("{error}");
                    std::process::exit(1);
                }
            } else {
                println!(
                    "sglang-rs serve http: model_path={} host={} port={} tp_size={} dp_size={} grpc_mode=false",
                    args.model_path, args.host, args.port, args.tp_size, args.dp_size
                );
                std::io::stdout().flush().expect("stdout should flush");
                if let Err(error) = launch_http_server(args).await {
                    eprintln!("{error}");
                    std::process::exit(1);
                }
            }
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    }
}
