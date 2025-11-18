"""
Database module for prediction logging with connection pooling
"""
import os
import psycopg2
from psycopg2 import pool
from psycopg2.extras import Json
from typing import Optional, List, Dict
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# Database connection parameters from environment
DATABASE_URL = os.getenv("DATABASE_URL")
DB_ENABLED = DATABASE_URL is not None

# Connection pool configuration
MIN_CONNECTIONS = int(os.getenv("DB_POOL_MIN_CONN", "2"))
MAX_CONNECTIONS = int(os.getenv("DB_POOL_MAX_CONN", "10"))

# Global connection pool
_connection_pool: Optional[pool.SimpleConnectionPool] = None


def _initialize_pool():
    """Initialize the connection pool"""
    global _connection_pool
    if _connection_pool is None and DB_ENABLED:
        try:
            _connection_pool = pool.SimpleConnectionPool(
                MIN_CONNECTIONS,
                MAX_CONNECTIONS,
                DATABASE_URL
            )
            logger.info(f"Initialized connection pool (min={MIN_CONNECTIONS}, max={MAX_CONNECTIONS})")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise


def get_pool() -> Optional[pool.SimpleConnectionPool]:
    """Get the connection pool, initializing if necessary"""
    if _connection_pool is None and DB_ENABLED:
        _initialize_pool()
    return _connection_pool


def close_pool():
    """Close all connections in the pool"""
    global _connection_pool
    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None
        logger.info("Closed connection pool")


@contextmanager
def get_db_connection():
    """
    Context manager for database connections from the pool

    This implementation uses connection pooling to avoid creating
    a new connection for each request, significantly improving
    performance under load.
    """
    if not DB_ENABLED:
        yield None
        return

    conn = None
    connection_pool = get_pool()

    if connection_pool is None:
        yield None
        return

    try:
        # Get connection from pool
        conn = connection_pool.getconn()
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn and connection_pool:
            # Return connection to pool instead of closing
            connection_pool.putconn(conn)


def log_prediction(
    input_sequence: List[List[float]],
    predictions: Dict[str, float],
    latencies: Dict[str, float],
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    model_version: Optional[str] = None
) -> Optional[int]:
    """
    Log a prediction to the database

    Args:
        input_sequence: Input sequence as list of lists
        predictions: Dict with keys: ann, gru, lstm, transformer, ensemble
        latencies: Dict with latency for each model in ms
        request_id: Optional request tracking ID
        user_id: Optional user ID
        model_version: Optional model version string

    Returns:
        Prediction ID or None if database is disabled
    """
    if not DB_ENABLED:
        logger.debug("Database logging disabled (no DATABASE_URL)")
        return None

    try:
        with get_db_connection() as conn:
            if conn is None:
                return None

            cur = conn.cursor()

            sql = """
                INSERT INTO predictions (
                    request_id,
                    user_id,
                    input_sequence,
                    sequence_length,
                    prediction_ann,
                    prediction_gru,
                    prediction_lstm,
                    prediction_transformer,
                    prediction_ensemble,
                    latency_ms,
                    latency_ann_ms,
                    latency_gru_ms,
                    latency_lstm_ms,
                    latency_transformer_ms,
                    model_version
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id
            """

            total_latency = sum(latencies.values())

            cur.execute(sql, (
                request_id,
                user_id,
                Json(input_sequence),  # Store as JSONB
                len(input_sequence),
                predictions['ann'],
                predictions['gru'],
                predictions['lstm'],
                predictions['transformer'],
                predictions['ensemble'],
                total_latency,
                latencies.get('ann', 0),
                latencies.get('gru', 0),
                latencies.get('lstm', 0),
                latencies.get('transformer', 0),
                model_version
            ))

            prediction_id = cur.fetchone()[0]
            cur.close()

            logger.info(f"Logged prediction {prediction_id}")
            return prediction_id

    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")
        return None


def update_actual_value(prediction_id: int, actual_value: float) -> bool:
    """
    Update the actual value for a prediction and compute errors

    Args:
        prediction_id: ID of the prediction
        actual_value: Observed actual value

    Returns:
        True if successful, False otherwise
    """
    if not DB_ENABLED:
        return False

    try:
        with get_db_connection() as conn:
            if conn is None:
                return False

            cur = conn.cursor()

            sql = """
                UPDATE predictions
                SET
                    actual_value = %s,
                    error_ann = actual_value - prediction_ann,
                    error_gru = actual_value - prediction_gru,
                    error_lstm = actual_value - prediction_lstm,
                    error_transformer = actual_value - prediction_transformer,
                    error_ensemble = actual_value - prediction_ensemble
                WHERE id = %s
            """

            cur.execute(sql, (actual_value, prediction_id))
            cur.close()

            logger.info(f"Updated actual value for prediction {prediction_id}")
            return True

    except Exception as e:
        logger.error(f"Failed to update actual value: {e}")
        return False


def get_recent_predictions(limit: int = 100) -> List[Dict]:
    """
    Get recent predictions

    Args:
        limit: Number of predictions to return

    Returns:
        List of prediction dictionaries
    """
    if not DB_ENABLED:
        return []

    try:
        with get_db_connection() as conn:
            if conn is None:
                return []

            cur = conn.cursor()

            sql = """
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
                LIMIT %s
            """

            cur.execute(sql, (limit,))
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            cur.close()

            return [dict(zip(columns, row)) for row in rows]

    except Exception as e:
        logger.error(f"Failed to get recent predictions: {e}")
        return []


def get_model_performance() -> Optional[Dict]:
    """
    Get aggregated model performance metrics

    Returns:
        Dictionary with performance metrics or None
    """
    if not DB_ENABLED:
        return None

    try:
        with get_db_connection() as conn:
            if conn is None:
                return None

            cur = conn.cursor()

            sql = """
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
                WHERE actual_value IS NOT NULL
            """

            cur.execute(sql)
            columns = [desc[0] for desc in cur.description]
            row = cur.fetchone()
            cur.close()

            if row:
                return dict(zip(columns, row))
            return None

    except Exception as e:
        logger.error(f"Failed to get model performance: {e}")
        return None
