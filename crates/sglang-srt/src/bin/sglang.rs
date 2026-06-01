use sglang_srt::cli::ServerArgs;

fn main() {
    match ServerArgs::parse_from(std::env::args().skip(1)) {
        Ok(args) => {
            println!(
                "sglang serve placeholder: model_path={} host={} port={} tp_size={} dp_size={} grpc_mode={}",
                args.model_path, args.host, args.port, args.tp_size, args.dp_size, args.grpc_mode
            );
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    }
}
