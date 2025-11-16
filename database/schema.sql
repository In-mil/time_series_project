-- Prediction Logging Schema for Time Series API
-- Stores all predictions for monitoring, analytics, and drift detection

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Request metadata
    request_id VARCHAR(255),
    user_id VARCHAR(255),

    -- Input sequence (stored as JSON for flexibility)
    input_sequence JSONB NOT NULL,
    sequence_length INTEGER NOT NULL,

    -- Model predictions (individual + ensemble)
    prediction_ann FLOAT NOT NULL,
    prediction_gru FLOAT NOT NULL,
    prediction_lstm FLOAT NOT NULL,
    prediction_transformer FLOAT NOT NULL,
    prediction_ensemble FLOAT NOT NULL,

    -- Performance metrics
    latency_ms FLOAT,
    latency_ann_ms FLOAT,
    latency_gru_ms FLOAT,
    latency_lstm_ms FLOAT,
    latency_transformer_ms FLOAT,

    -- Model version tracking
    model_version VARCHAR(50),

    -- Actual value (for later validation/drift detection)
    actual_value FLOAT,

    -- Error metrics (computed after actual value is known)
    error_ann FLOAT,
    error_gru FLOAT,
    error_lstm FLOAT,
    error_transformer FLOAT,
    error_ensemble FLOAT
);

-- Index for time-based queries
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp DESC);

-- Index for user tracking
CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id);

-- Index for request tracking
CREATE INDEX IF NOT EXISTS idx_predictions_request ON predictions(request_id);

-- View for recent predictions
CREATE OR REPLACE VIEW recent_predictions AS
SELECT
    id,
    timestamp,
    prediction_ensemble,
    prediction_ann,
    prediction_gru,
    prediction_lstm,
    prediction_transformer,
    latency_ms,
    actual_value,
    error_ensemble
FROM predictions
ORDER BY timestamp DESC
LIMIT 100;

-- View for model performance comparison
CREATE OR REPLACE VIEW model_performance AS
SELECT
    COUNT(*) as total_predictions,
    AVG(latency_ann_ms) as avg_latency_ann,
    AVG(latency_gru_ms) as avg_latency_gru,
    AVG(latency_lstm_ms) as avg_latency_lstm,
    AVG(latency_transformer_ms) as avg_latency_transformer,
    AVG(ABS(error_ann)) as mae_ann,
    AVG(ABS(error_gru)) as mae_gru,
    AVG(ABS(error_lstm)) as mae_lstm,
    AVG(ABS(error_transformer)) as mae_transformer,
    AVG(ABS(error_ensemble)) as mae_ensemble
FROM predictions
WHERE actual_value IS NOT NULL;

-- View for daily statistics
CREATE OR REPLACE VIEW daily_stats AS
SELECT
    DATE(timestamp) as date,
    COUNT(*) as prediction_count,
    AVG(prediction_ensemble) as avg_prediction,
    MIN(prediction_ensemble) as min_prediction,
    MAX(prediction_ensemble) as max_prediction,
    AVG(latency_ms) as avg_latency
FROM predictions
GROUP BY DATE(timestamp)
ORDER BY date DESC;

COMMENT ON TABLE predictions IS 'Stores all predictions from the time series API for monitoring and analysis';
COMMENT ON COLUMN predictions.input_sequence IS 'Input sequence as JSON array';
COMMENT ON COLUMN predictions.actual_value IS 'Actual observed value (filled in later for validation)';
COMMENT ON COLUMN predictions.error_ensemble IS 'Prediction error = actual_value - prediction_ensemble';
