//! Polyfill Sidecar — A Rust service that streams Polymarket orderbooks
//! and serves them to the Python trading agent via a local HTTP API.
//!
//! Architecture:
//! - WebSocket connection to Polymarket CLOB for real-time book updates
//! - In-memory BTreeMap-based orderbooks (via polyfill-rs) for O(log n) lookups
//! - Axum HTTP server on :8080 serving JSON snapshots to the Python agent

mod orderbook_cache;
mod routes;
mod types;
mod ws_manager;

use orderbook_cache::OrderBookCache;
use routes::AppState;
use ws_manager::WsManager;

use axum::routing::{get, post};
use axum::Router;
use std::time::Instant;
use tokio::sync::watch;
use tower_http::cors::CorsLayer;
use tracing::info;

const DEFAULT_PORT: u16 = 8080;
const CLOB_HOST: &str = "https://clob.polymarket.com";

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "polyfill_sidecar=info,polyfill_rs=warn".into()),
        )
        .init();

    info!("Starting Polyfill Sidecar...");

    // Create shared orderbook cache
    let cache = OrderBookCache::new();

    // Shutdown signal
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // Spawn WebSocket manager in background
    let ws_cache = cache.clone();
    let ws_handle = tokio::spawn(async move {
        let ws_manager = WsManager::new(ws_cache, CLOB_HOST.to_string());
        ws_manager.run(shutdown_rx).await;
    });

    // Build HTTP router
    let state = AppState {
        cache,
        start_time: Instant::now(),
    };

    let app = Router::new()
        .route("/health", get(routes::health))
        .route("/book/{token_id}", get(routes::get_book))
        .route("/book/{token_id}/summary", get(routes::get_book_summary))
        .route("/books", get(routes::get_all_books))
        .route("/price/{token_id}", get(routes::get_price))
        .route("/spread/{token_id}", get(routes::get_spread))
        .route("/subscribe", post(routes::subscribe))
        .route("/unsubscribe", post(routes::unsubscribe))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let port = std::env::var("SIDECAR_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(DEFAULT_PORT);

    let addr = format!("0.0.0.0:{}", port);
    info!("HTTP server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();

    // Run server with graceful shutdown on Ctrl+C
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            tokio::signal::ctrl_c().await.ok();
            info!("Shutdown signal received, stopping...");
            let _ = shutdown_tx.send(true);
        })
        .await
        .unwrap();

    // Wait for WS manager to finish
    let _ = ws_handle.await;
    info!("Sidecar shutdown complete.");
}
