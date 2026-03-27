//! WebSocket manager for streaming Polymarket orderbook updates.
//!
//! Connects to the Polymarket CLOB WebSocket, subscribes to book channels
//! for tracked tokens, and applies deltas to the in-memory OrderBookCache.
//! Automatically reconnects when new tokens are subscribed.

use crate::orderbook_cache::OrderBookCache;
use polyfill_rs::client::ClobClient;
use polyfill_rs::stream::WebSocketStream;
use polyfill_rs::types::{StreamMessage, WssSubscription};
use std::collections::HashSet;
use std::time::Duration;
use tokio::sync::watch;
use tracing::{debug, error, info, warn};

/// Polymarket CLOB WebSocket URL.
const WS_URL: &str = "wss://ws-subscriptions-clob.polymarket.com/ws/market";

/// How often to check for new subscriptions while streaming (seconds).
const SUBSCRIPTION_CHECK_INTERVAL: Duration = Duration::from_secs(10);

pub struct WsManager {
    cache: OrderBookCache,
    clob_host: String,
}

impl WsManager {
    pub fn new(cache: OrderBookCache, clob_host: String) -> Self {
        Self { cache, clob_host }
    }

    /// Run the WebSocket streaming loop.
    /// Reconnects when new tokens are subscribed or on errors.
    pub async fn run(&self, mut shutdown: watch::Receiver<bool>) {
        info!("WebSocket manager starting...");
        let mut connected_tokens: HashSet<String> = HashSet::new();

        loop {
            if *shutdown.borrow() {
                info!("WebSocket manager shutting down");
                return;
            }

            let token_ids = self.cache.subscribed_tokens();
            if token_ids.is_empty() {
                debug!("No tokens subscribed, waiting...");
                connected_tokens.clear();
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(5)) => continue,
                    _ = shutdown.changed() => return,
                }
            }

            let token_set: HashSet<String> = token_ids.iter().cloned().collect();

            // Only reconnect if the token set actually changed
            if token_set != connected_tokens {
                info!(
                    "Token set changed ({} -> {} tokens), (re)connecting...",
                    connected_tokens.len(),
                    token_set.len()
                );
                connected_tokens = token_set;

                let tokens_vec: Vec<String> = connected_tokens.iter().cloned().collect();
                match self.stream_loop(&tokens_vec, &mut shutdown).await {
                    Ok(reason) => {
                        match reason {
                            StreamEndReason::NewTokens => {
                                info!("Reconnecting for new token subscriptions...");
                                continue; // Immediately reconnect with new tokens
                            }
                            StreamEndReason::Shutdown => return,
                            StreamEndReason::Disconnected => {
                                info!("Stream ended, reconnecting in 5s...");
                            }
                        }
                    }
                    Err(e) => {
                        warn!("WebSocket error: {}, reconnecting in 5s...", e);
                        // Force re-check of tokens on next iteration
                        connected_tokens.clear();
                    }
                }

                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(5)) => {},
                    _ = shutdown.changed() => return,
                }
            } else {
                // Tokens haven't changed, just wait
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(5)) => {},
                    _ = shutdown.changed() => return,
                }
            }
        }
    }

    /// Inner streaming loop. Returns the reason it ended.
    async fn stream_loop(
        &self,
        token_ids: &[String],
        shutdown: &mut watch::Receiver<bool>,
    ) -> Result<StreamEndReason, anyhow::Error> {
        // Fetch initial orderbook snapshots via REST
        let client = ClobClient::new(&self.clob_host);
        for token_id in token_ids {
            match client.get_order_book(token_id).await {
                Ok(summary) => {
                    self.cache
                        .seed_from_api(token_id, &summary.market, &summary.bids, &summary.asks);
                    info!(
                        "Seeded book for {} ({} bids, {} asks)",
                        &token_id[..12.min(token_id.len())],
                        summary.bids.len(),
                        summary.asks.len()
                    );
                }
                Err(e) => {
                    warn!("Failed to fetch initial book for {}: {}", &token_id[..12.min(token_id.len())], e);
                }
            }
        }

        // Connect WebSocket and subscribe
        let mut ws = WebSocketStream::new(WS_URL);
        let subscription = WssSubscription {
            channel_type: "market".to_string(),
            operation: Some("subscribe".to_string()),
            markets: Vec::new(),
            asset_ids: token_ids.to_vec(),
            initial_dump: Some(true),
            custom_feature_enabled: None,
            auth: None,
        };
        ws.subscribe_async(subscription).await?;
        info!("WebSocket connected and subscribed to {} tokens", token_ids.len());

        let current_set: HashSet<String> = token_ids.iter().cloned().collect();
        let mut check_interval = tokio::time::interval(SUBSCRIPTION_CHECK_INTERVAL);

        use futures_util::StreamExt;
        loop {
            tokio::select! {
                msg = ws.next() => {
                    match msg {
                        Some(Ok(stream_msg)) => self.handle_message(stream_msg),
                        Some(Err(e)) => {
                            error!("WebSocket message error: {}", e);
                            return Err(anyhow::anyhow!("WebSocket error: {}", e));
                        }
                        None => {
                            warn!("WebSocket stream ended");
                            return Ok(StreamEndReason::Disconnected);
                        }
                    }
                }
                _ = check_interval.tick() => {
                    // Check if subscriptions changed
                    let new_tokens: HashSet<String> = self.cache.subscribed_tokens().into_iter().collect();
                    if new_tokens != current_set {
                        info!("Detected new token subscriptions, will reconnect");
                        return Ok(StreamEndReason::NewTokens);
                    }
                }
                _ = shutdown.changed() => {
                    info!("Shutdown signal received");
                    return Ok(StreamEndReason::Shutdown);
                }
            }
        }
    }

    fn handle_message(&self, msg: StreamMessage) {
        match msg {
            StreamMessage::Book(book_update) => {
                debug!("Book update for {}", &book_update.asset_id[..12.min(book_update.asset_id.len())]);
                if let Err(e) = self.cache.manager().apply_book_update(&book_update) {
                    warn!("Failed to apply book update: {}", e);
                }
            }
            StreamMessage::PriceChange(_) => {}
            StreamMessage::LastTradePrice(_) => {}
            _ => {}
        }
    }
}

enum StreamEndReason {
    NewTokens,
    Shutdown,
    Disconnected,
}
