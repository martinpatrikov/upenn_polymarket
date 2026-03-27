//! WebSocket manager for streaming Polymarket orderbook updates.
//!
//! Connects to the Polymarket CLOB WebSocket, subscribes to book channels
//! for tracked tokens, and applies deltas to the in-memory OrderBookCache.

use crate::orderbook_cache::OrderBookCache;
use polyfill_rs::client::ClobClient;
use polyfill_rs::stream::WebSocketStream;
use polyfill_rs::types::{StreamMessage, WssSubscription};
use std::time::Duration;
use tokio::sync::watch;
use tracing::{debug, error, info, warn};

/// Polymarket CLOB WebSocket URL.
const WS_URL: &str = "wss://ws-subscriptions-clob.polymarket.com/ws/market";

/// Manages WebSocket connections and applies book updates to the cache.
pub struct WsManager {
    cache: OrderBookCache,
    clob_host: String,
}

impl WsManager {
    pub fn new(cache: OrderBookCache, clob_host: String) -> Self {
        Self { cache, clob_host }
    }

    /// Run the WebSocket streaming loop.
    ///
    /// Connects to Polymarket, subscribes to book updates for all
    /// tracked tokens, and continuously applies deltas to the cache.
    /// Reconnects automatically on failure.
    pub async fn run(&self, mut shutdown: watch::Receiver<bool>) {
        info!("WebSocket manager starting...");

        loop {
            if *shutdown.borrow() {
                info!("WebSocket manager shutting down");
                return;
            }

            let token_ids = self.cache.subscribed_tokens();
            if token_ids.is_empty() {
                debug!("No tokens subscribed, waiting...");
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(5)) => continue,
                    _ = shutdown.changed() => return,
                }
            }

            info!("Connecting for {} tokens...", token_ids.len());

            match self.stream_loop(&token_ids, &mut shutdown).await {
                Ok(()) => info!("WebSocket stream ended gracefully"),
                Err(e) => warn!("WebSocket error: {}, reconnecting in 5s...", e),
            }

            tokio::select! {
                _ = tokio::time::sleep(Duration::from_secs(5)) => {},
                _ = shutdown.changed() => return,
            }
        }
    }

    /// Inner streaming loop: fetch initial snapshots, connect WS, process messages.
    async fn stream_loop(
        &self,
        token_ids: &[String],
        shutdown: &mut watch::Receiver<bool>,
    ) -> Result<(), anyhow::Error> {
        // Fetch initial orderbook snapshots via REST
        let client = ClobClient::new(&self.clob_host);
        for token_id in token_ids {
            match client.get_order_book(token_id).await {
                Ok(summary) => {
                    self.cache
                        .seed_from_api(token_id, &summary.market, &summary.bids, &summary.asks);
                    info!(
                        "Seeded book for {} ({} bids, {} asks)",
                        token_id,
                        summary.bids.len(),
                        summary.asks.len()
                    );
                }
                Err(e) => {
                    warn!("Failed to fetch initial book for {}: {}", token_id, e);
                }
            }
        }

        // Create WebSocket and subscribe to market channel
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

        // Process incoming messages
        use futures_util::StreamExt;
        loop {
            tokio::select! {
                msg = ws.next() => {
                    match msg {
                        Some(Ok(stream_msg)) => {
                            self.handle_message(stream_msg);
                        }
                        Some(Err(e)) => {
                            error!("WebSocket message error: {}", e);
                            return Err(anyhow::anyhow!("WebSocket error: {}", e));
                        }
                        None => {
                            warn!("WebSocket stream ended");
                            return Ok(());
                        }
                    }
                }
                _ = shutdown.changed() => {
                    info!("Shutdown signal received");
                    return Ok(());
                }
            }
        }
    }

    /// Handle a single stream message by applying it to the cache.
    fn handle_message(&self, msg: StreamMessage) {
        match msg {
            StreamMessage::Book(book_update) => {
                debug!("Book update for {}", book_update.asset_id);
                if let Err(e) = self.cache.manager().apply_book_update(&book_update) {
                    warn!("Failed to apply book update: {}", e);
                }
            }
            StreamMessage::PriceChange(_) => {
                debug!("Price change received");
            }
            StreamMessage::LastTradePrice(_) => {
                debug!("Last trade price received");
            }
            _ => {
                debug!("Other stream message received");
            }
        }
    }
}
