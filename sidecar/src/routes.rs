//! HTTP route handlers for the sidecar REST API.
//!
//! All routes return JSON and are consumed by the Python agent.

use crate::orderbook_cache::OrderBookCache;
use crate::types::{HealthResponse, SubscribeRequest};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Json;
use std::time::Instant;

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    pub cache: OrderBookCache,
    pub start_time: Instant,
}

/// GET /health — Health check.
pub async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        tracked_books: state.cache.num_tracked(),
        uptime_seconds: state.start_time.elapsed().as_secs(),
    })
}

/// GET /book/:token_id — Full orderbook snapshot.
pub async fn get_book(
    State(state): State<AppState>,
    Path(token_id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.cache.get_book(&token_id) {
        Some(book) => Ok(Json(serde_json::to_value(book).unwrap())),
        None => Err(StatusCode::NOT_FOUND),
    }
}

/// GET /book/:token_id/summary — Orderbook summary (best bid/ask, spread, depth).
pub async fn get_book_summary(
    State(state): State<AppState>,
    Path(token_id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.cache.get_summary(&token_id) {
        Some(summary) => Ok(Json(serde_json::to_value(summary).unwrap())),
        None => Err(StatusCode::NOT_FOUND),
    }
}

/// GET /books — All tracked orderbook summaries.
pub async fn get_all_books(State(state): State<AppState>) -> Json<serde_json::Value> {
    let summaries = state.cache.get_all_summaries();
    Json(serde_json::to_value(summaries).unwrap())
}

/// GET /price/:token_id — Current midpoint price.
pub async fn get_price(
    State(state): State<AppState>,
    Path(token_id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.cache.get_summary(&token_id) {
        Some(summary) => Ok(Json(serde_json::json!({
            "token_id": token_id,
            "midpoint": summary.midpoint,
            "best_bid": summary.best_bid,
            "best_ask": summary.best_ask,
        }))),
        None => Err(StatusCode::NOT_FOUND),
    }
}

/// GET /spread/:token_id — Current bid-ask spread.
pub async fn get_spread(
    State(state): State<AppState>,
    Path(token_id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match state.cache.get_summary(&token_id) {
        Some(summary) => Ok(Json(serde_json::json!({
            "token_id": token_id,
            "spread": summary.spread,
            "best_bid": summary.best_bid,
            "best_ask": summary.best_ask,
        }))),
        None => Err(StatusCode::NOT_FOUND),
    }
}

/// POST /subscribe — Subscribe to token_ids (start tracking orderbooks).
pub async fn subscribe(
    State(state): State<AppState>,
    Json(req): Json<SubscribeRequest>,
) -> Json<serde_json::Value> {
    state.cache.subscribe(&req.token_ids);
    Json(serde_json::json!({
        "subscribed": req.token_ids,
        "total_tracked": state.cache.num_tracked(),
    }))
}

/// POST /unsubscribe — Unsubscribe from token_ids (stop tracking).
pub async fn unsubscribe(
    State(state): State<AppState>,
    Json(req): Json<SubscribeRequest>,
) -> Json<serde_json::Value> {
    state.cache.unsubscribe(&req.token_ids);
    Json(serde_json::json!({
        "unsubscribed": req.token_ids,
        "total_tracked": state.cache.num_tracked(),
    }))
}
