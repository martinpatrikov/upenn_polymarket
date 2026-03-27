//! In-memory orderbook cache using polyfill-rs's BTreeMap-based OrderBook.
//!
//! Maintains one OrderBook per token_id. Thread-safe via RwLock.
//! The Python agent queries snapshots from this cache via HTTP.

use crate::types::{BookLevelJson, BookSummaryJson, OrderBookJson, decimal_to_f64};
use chrono::Utc;
use polyfill_rs::book::OrderBookManager;
use polyfill_rs::types::{BookUpdate, OrderSummary};
use std::collections::HashSet;
use std::sync::{Arc, RwLock};
use tracing::{debug, info};

/// Maximum depth of orderbook levels to maintain per side.
const MAX_DEPTH: usize = 50;

/// Thread-safe orderbook cache.
///
/// Wraps polyfill-rs's OrderBookManager with additional tracking of
/// which token_ids are actively subscribed.
#[derive(Clone)]
pub struct OrderBookCache {
    manager: Arc<OrderBookManager>,
    subscribed: Arc<RwLock<HashSet<String>>>,
}

impl OrderBookCache {
    pub fn new() -> Self {
        Self {
            manager: Arc::new(OrderBookManager::new(MAX_DEPTH)),
            subscribed: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    /// Subscribe to a set of token_ids (mark them for tracking).
    pub fn subscribe(&self, token_ids: &[String]) {
        let mut subs = self.subscribed.write().unwrap();
        for id in token_ids {
            subs.insert(id.clone());
            // Pre-create the book in the manager
            let _ = self.manager.get_or_create_book(id);
            info!("Subscribed to token: {}", id);
        }
    }

    /// Unsubscribe from a set of token_ids.
    pub fn unsubscribe(&self, token_ids: &[String]) {
        let mut subs = self.subscribed.write().unwrap();
        for id in token_ids {
            subs.remove(id);
            debug!("Unsubscribed from token: {}", id);
        }
    }

    /// Get the set of currently subscribed token_ids.
    pub fn subscribed_tokens(&self) -> Vec<String> {
        self.subscribed.read().unwrap().iter().cloned().collect()
    }

    /// Get a reference to the underlying OrderBookManager (for applying deltas).
    pub fn manager(&self) -> &OrderBookManager {
        &self.manager
    }

    /// Get the full orderbook snapshot for a token.
    pub fn get_book(&self, token_id: &str) -> Option<OrderBookJson> {
        let book = self.manager.get_or_create_book(token_id).ok()?;
        let bids = book.bids(Some(MAX_DEPTH));
        let asks = book.asks(Some(MAX_DEPTH));

        Some(OrderBookJson {
            token_id: token_id.to_string(),
            bids: bids
                .iter()
                .map(|l| BookLevelJson {
                    price: decimal_to_f64(l.price),
                    size: decimal_to_f64(l.size),
                })
                .collect(),
            asks: asks
                .iter()
                .map(|l| BookLevelJson {
                    price: decimal_to_f64(l.price),
                    size: decimal_to_f64(l.size),
                })
                .collect(),
            timestamp: Utc::now().to_rfc3339(),
            sequence: book.sequence,
        })
    }

    /// Get a summary of the orderbook for a token.
    pub fn get_summary(&self, token_id: &str) -> Option<BookSummaryJson> {
        let book = self.manager.get_or_create_book(token_id).ok()?;
        let bids = book.bids(Some(MAX_DEPTH));
        let asks = book.asks(Some(MAX_DEPTH));

        let bid_depth: f64 = bids.iter().map(|l| decimal_to_f64(l.size)).sum();
        let ask_depth: f64 = asks.iter().map(|l| decimal_to_f64(l.size)).sum();

        Some(BookSummaryJson {
            token_id: token_id.to_string(),
            best_bid: book.best_bid().map(|l| decimal_to_f64(l.price)),
            best_ask: book.best_ask().map(|l| decimal_to_f64(l.price)),
            midpoint: book.mid_price().map(decimal_to_f64),
            spread: book.spread().map(decimal_to_f64),
            bid_depth,
            ask_depth,
            num_bid_levels: bids.len(),
            num_ask_levels: asks.len(),
            timestamp: Utc::now().to_rfc3339(),
        })
    }

    /// Seed a book from an API-fetched OrderBookSummary.
    pub fn seed_from_api(
        &self,
        token_id: &str,
        market: &str,
        api_bids: &[OrderSummary],
        api_asks: &[OrderSummary],
    ) {
        let update = BookUpdate {
            asset_id: token_id.to_string(),
            market: market.to_string(),
            timestamp: Utc::now().timestamp() as u64,
            bids: api_bids.to_vec(),
            asks: api_asks.to_vec(),
            hash: None,
        };
        if let Err(e) = self.manager.apply_book_update(&update) {
            tracing::warn!("Failed to seed book for {}: {}", token_id, e);
        }
    }

    /// Get summaries for all tracked books.
    pub fn get_all_summaries(&self) -> Vec<BookSummaryJson> {
        let subs = self.subscribed.read().unwrap();
        subs.iter()
            .filter_map(|id| self.get_summary(id))
            .collect()
    }

    /// Number of tracked orderbooks.
    pub fn num_tracked(&self) -> usize {
        self.subscribed.read().unwrap().len()
    }
}
