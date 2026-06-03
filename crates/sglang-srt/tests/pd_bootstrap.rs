use sglang_srt::transfer::{
    DecodeBootstrapRegistry, DecodeBootstrapRegistryError, DecodeBootstrapSession, KvPoll,
};
use sglang_srt::types::{DisaggregatedParams, RequestId};

#[test]
fn decode_bootstrap_registry_tracks_session_dp_rank_and_status_by_room() {
    let params = DisaggregatedParams {
        bootstrap_host: "10.0.0.7".to_string(),
        bootstrap_port: 8998,
        bootstrap_room: 42,
    };
    let session = DecodeBootstrapSession::new(RequestId::from("decode-req"), params.clone(), 3);
    let mut registry = DecodeBootstrapRegistry::default();

    registry
        .register(session)
        .expect("bootstrap room should register");

    let registered = registry.get(42).expect("session should be registered");
    assert_eq!(registered.request_id(), &RequestId::from("decode-req"));
    assert_eq!(registered.disaggregated_params(), &params);
    assert_eq!(registered.data_parallel_rank(), 3);
    assert_eq!(registered.status(), KvPoll::Bootstrapping);
    assert_eq!(registry.query_data_parallel_rank(42), Some(3));

    registry
        .update_status(42, KvPoll::WaitingForInput)
        .expect("status should update");
    assert_eq!(
        registry.get(42).expect("session should remain").status(),
        KvPoll::WaitingForInput
    );

    let removed = registry.remove(42).expect("session should remove");
    assert_eq!(removed.request_id(), &RequestId::from("decode-req"));
    assert!(registry.is_empty());
}

#[test]
fn decode_bootstrap_registry_rejects_duplicate_rooms() {
    let mut registry = DecodeBootstrapRegistry::default();
    registry
        .register(session("first", 8))
        .expect("first room should register");

    let error = registry
        .register(session("second", 8))
        .expect_err("duplicate room should be rejected");

    assert_eq!(
        error,
        DecodeBootstrapRegistryError::DuplicateBootstrapRoom(8)
    );
    assert_eq!(registry.len(), 1);
    assert_eq!(
        registry
            .get(8)
            .expect("original session should remain")
            .request_id(),
        &RequestId::from("first")
    );
}

#[test]
fn decode_bootstrap_registry_reports_missing_room_on_status_update() {
    let mut registry = DecodeBootstrapRegistry::default();

    let error = registry
        .update_status(99, KvPoll::Failed)
        .expect_err("missing room should be reported");

    assert_eq!(
        error,
        DecodeBootstrapRegistryError::MissingBootstrapRoom(99)
    );
}

fn session(request_id: &str, bootstrap_room: i32) -> DecodeBootstrapSession {
    DecodeBootstrapSession::new(
        RequestId::from(request_id),
        DisaggregatedParams {
            bootstrap_host: "10.0.0.7".to_string(),
            bootstrap_port: 8998,
            bootstrap_room,
        },
        0,
    )
}
