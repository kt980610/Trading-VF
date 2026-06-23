//! Live trading binary. Real order execution requires the `live-http` feature
//! AND `live.real_money=true` with `live.shadow_mode=false`.

use trading_live::config::Config;

fn config_path() -> String {
    std::env::args()
        .nth(1)
        .unwrap_or_else(|| "config/live_config.yaml".to_string())
}

#[cfg(feature = "live-http")]
fn run() -> anyhow::Result<()> {
    use std::time::Duration;
    use trading_live::config::ApiCredentials;
    use trading_live::decision::MarketContext;
    use trading_live::engine::Engine;
    use trading_live::exchange::binance::BinanceClient;
    use trading_live::exchange::ExchangeClient;
    use trading_live::features::MarketVolume;

    let _ = dotenvy::dotenv();
    let cfg = Config::load(config_path())?;

    let creds = ApiCredentials::from_env(&cfg.exchange.base_url)?;
    let client = BinanceClient::new(creds)?;
    client.ping()?; // connectivity test

    // Parse the halving anchors + season seed once (used for the season one-hot
    // features each minute). Must match the Python training configuration.
    let halvings = std::sync::Arc::new(cfg.halving.epoch_secs());
    let season_seed = cfg.halving.season_seed;

    let mut engine = Engine::new(cfg, client)?;

    let report = engine.startup_checks();
    for issue in &report.issues {
        eprintln!("[startup] {issue}");
    }

    if engine.kill_switch_active() {
        eprintln!("[safe-idle] kill switch present; not trading. Manual reset required.");
        return Ok(());
    }

    if report.can_open_new_trades {
        match engine.rebalance_to_targets() {
            Ok(s) => eprintln!(
                "[rebalance] equity={:.2} opened={} reduced={} closed={}",
                s.total_equity, s.opened, s.reduced, s.closed
            ),
            Err(e) => eprintln!("[rebalance] error: {e}"),
        }
    } else {
        eprintln!("[safe-idle] startup checks failed; managing existing positions only");
    }

    loop {
        let symbols: Vec<String> = engine.state.symbols.keys().cloned().collect();
        let mut contexts: std::collections::HashMap<String, MarketContext> =
            std::collections::HashMap::new();
        let now_secs = trading_live::clock::now_unix_secs();
        for symbol in symbols {
            if let Ok(price) = engine.exchange.mark_price(&symbol) {
                contexts.insert(
                    symbol.clone(),
                    MarketContext {
                        timestamp: trading_live::clock::now_rfc3339(),
                        date: trading_live::clock::date_str(now_secs),
                        current_price: price,
                        hour_of_day: trading_live::clock::hour_of_day(now_secs),
                        day_of_week: trading_live::clock::day_of_week(now_secs),
                        market_volume: MarketVolume::default(),
                        halvings: std::sync::Arc::clone(&halvings),
                        season_seed,
                    },
                );
            }
        }

        if let Err(e) = engine.run_cycle(&contexts) {
            eprintln!("[cycle] error: {e}");
        }
        if engine.halted {
            eprintln!("[halt] engine halted; exiting loop");
            break;
        }
        std::thread::sleep(Duration::from_secs(60));
    }

    Ok(())
}

#[cfg(not(feature = "live-http"))]
fn run() -> anyhow::Result<()> {
    let _cfg = Config::load(config_path())?;
    eprintln!(
        "[safe-idle] built without the `live-http` feature: no Binance client compiled in. \
         Rebuild with `--features live-http` to enable live order execution."
    );
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[fatal] {e}");
        std::process::exit(1);
    }
}
