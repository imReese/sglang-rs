// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

pub(crate) fn openai_response_id(prefix: &str, request_id: &str) -> String {
    if request_id.starts_with(prefix) {
        request_id.to_string()
    } else {
        format!("{prefix}{request_id}")
    }
}
