use sglang_srt::cli::ServerArgs;
use sglang_srt::server::launch_grpc_server;

#[tokio::main]
async fn main() {
    match ServerArgs::parse_from(std::env::args().skip(1)) {
        Ok(args) => {
            if args.grpc_mode {
                println!(
                    "sglang serve grpc: model_path={} host={} port={} tp_size={} dp_size={} grpc_mode=true",
                    args.model_path, args.host, args.port, args.tp_size, args.dp_size
                );
                if let Err(error) = launch_grpc_server(args).await {
                    eprintln!("{error}");
                    std::process::exit(1);
                }
            } else {
                println!(
                    "sglang serve placeholder: model_path={} host={} port={} tp_size={} dp_size={} grpc_mode=false",
                    args.model_path, args.host, args.port, args.tp_size, args.dp_size
                );
            }
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    }
}
