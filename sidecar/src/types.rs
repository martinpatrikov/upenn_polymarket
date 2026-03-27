//! Shared types for the sidecar REST API.
//!
//! These types are serialized to JSON and sent to the Python agent.

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// A single price level in the order book.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookLevelJson {
    pub price: f64,
    pub size: f64,
}

/// Full order book snapshot returned by GET /book/:token_id
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookJson {
    pub token_id: String,
    pub bids: Vec<BookLevelJson>,
    pub asks: Vec<BookLevelJson>,
    pub timestamp: String,
    pub sequence: u64,
}

/// Summary of an order book returned by GET /book/:token_id/summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookSummaryJson {
    pub token_id: String,
    pub best_bid: Option<f64>,
    pub best_ask: Option<f64>,
    pub midpoint: Option<f64>,
    pub spread: Option<f64>,
    pub bid_depth: f64,
    pub ask_depth: f64,
    pub num_bid_levels: usize,
    pub num_ask_levels: usize,
    pub timestamp: String,
}

/// Request body for POST /subscribe
#[derive(Debug, Deserialize)]
pub struct SubscribeRequest {
    pub token_ids: Vec<String>,
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub tracked_books: usize,
    pub uptime_seconds: u64,
}

/// Helper to convert Decimal to f64 for JSON serialization.
pub fn decimal_to_f64(d: Decimal) -> f64 {
    use rust_decimal::prelude::ToPrimitive;
    d.to_f64().unwrap_or(0.0)
}
