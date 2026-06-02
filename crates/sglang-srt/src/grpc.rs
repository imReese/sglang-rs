use tonic::{Code, Status};

use crate::router::{RouterProtocolError, RouterStatusCode};

pub fn router_status_code_to_grpc_code(status_code: RouterStatusCode) -> Code {
    match status_code {
        RouterStatusCode::InvalidArgument => Code::InvalidArgument,
        RouterStatusCode::ResourceExhausted => Code::ResourceExhausted,
        RouterStatusCode::FailedPrecondition => Code::FailedPrecondition,
    }
}

pub fn router_protocol_error_to_status(error: RouterProtocolError) -> Status {
    Status::new(
        router_status_code_to_grpc_code(error.status_code()),
        error.to_string(),
    )
}
