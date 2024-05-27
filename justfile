quick_dev:
    cargo watch -q -c -w examples/ -x "run --example quick_dev"
auth:
    cargo watch -q -c -w examples/ -x "run --example oauth2_callback"
dev:
    cargo watch -q -c -w src/ -x "run"
