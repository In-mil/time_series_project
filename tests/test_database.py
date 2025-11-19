"""
Tests for database module
"""
import pytest
from unittest.mock import MagicMock, patch, call
import psycopg2
from psycopg2 import pool


class TestDatabaseConfiguration:
    """Tests for database configuration and initialization"""

    @patch.dict('os.environ', {'DATABASE_URL': 'postgresql://user:pass@localhost/db'})
    def test_database_enabled_when_url_provided(self):
        """Test that database is enabled when DATABASE_URL is provided"""
        # Need to reimport to pick up new env vars
        import importlib
        from service import database
        importlib.reload(database)

        assert database.DB_ENABLED is True
        assert database.DATABASE_URL == 'postgresql://user:pass@localhost/db'

    @patch.dict('os.environ', {}, clear=True)
    def test_database_disabled_when_no_url(self):
        """Test that database is disabled when DATABASE_URL is not provided"""
        import importlib
        from service import database
        importlib.reload(database)

        assert database.DB_ENABLED is False
        assert database.DATABASE_URL is None

    @patch.dict('os.environ', {
        'DATABASE_URL': 'postgresql://localhost/db',
        'DB_POOL_MIN_CONN': '5',
        'DB_POOL_MAX_CONN': '20'
    })
    def test_pool_configuration_from_env(self):
        """Test that pool configuration is read from environment"""
        import importlib
        from service import database
        importlib.reload(database)

        assert database.MIN_CONNECTIONS == 5
        assert database.MAX_CONNECTIONS == 20


class TestConnectionPool:
    """Tests for connection pool management"""

    @patch.dict('os.environ', {'DATABASE_URL': 'postgresql://localhost/db'})
    @patch('psycopg2.pool.SimpleConnectionPool')
    def test_initialize_pool_creates_pool(self, mock_pool_class):
        """Test that _initialize_pool creates a connection pool"""
        from service import database
        import importlib
        importlib.reload(database)

        # Initialize pool
        database._initialize_pool()

        # Verify pool was created with correct parameters
        mock_pool_class.assert_called_once()
        call_args = mock_pool_class.call_args
        assert call_args[0][0] == database.MIN_CONNECTIONS
        assert call_args[0][1] == database.MAX_CONNECTIONS
        assert call_args[0][2] == 'postgresql://localhost/db'

    @patch.dict('os.environ', {'DATABASE_URL': 'postgresql://localhost/db'})
    @patch('psycopg2.pool.SimpleConnectionPool')
    def test_get_pool_initializes_if_needed(self, mock_pool_class):
        """Test that get_pool initializes pool if not exists"""
        from service import database
        import importlib
        importlib.reload(database)

        # Reset global pool
        database._connection_pool = None

        # Get pool
        result = database.get_pool()

        # Verify pool was created
        mock_pool_class.assert_called_once()
        assert result is not None

    @patch.dict('os.environ', {}, clear=True)
    def test_get_pool_returns_none_when_disabled(self):
        """Test that get_pool returns None when database is disabled"""
        from service import database
        import importlib
        importlib.reload(database)

        result = database.get_pool()
        assert result is None

    @patch.dict('os.environ', {'DATABASE_URL': 'postgresql://localhost/db'})
    def test_close_pool_closes_all_connections(self):
        """Test that close_pool closes all connections"""
        from service import database
        import importlib
        importlib.reload(database)

        # Create mock pool
        mock_pool = MagicMock()
        database._connection_pool = mock_pool

        # Close pool
        database.close_pool()

        # Verify closeall was called
        mock_pool.closeall.assert_called_once()
        assert database._connection_pool is None


class TestDatabaseConnection:
    """Tests for database connection context manager"""

    @patch.dict('os.environ', {}, clear=True)
    def test_get_db_connection_yields_none_when_disabled(self):
        """Test that connection context yields None when database is disabled"""
        from service import database
        import importlib
        importlib.reload(database)

        with database.get_db_connection() as conn:
            assert conn is None

    @patch.dict('os.environ', {'DATABASE_URL': 'postgresql://localhost/db'})
    @patch('psycopg2.pool.SimpleConnectionPool')
    def test_get_db_connection_commits_on_success(self, mock_pool_class):
        """Test that connection commits transaction on success"""
        from service import database
        import importlib
        importlib.reload(database)

        # Create mock connection
        mock_conn = MagicMock()
        mock_pool = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_pool_class.return_value = mock_pool
        database._connection_pool = None

        # Use connection
        with database.get_db_connection() as conn:
            assert conn is mock_conn

        # Verify commit and putconn were called
        mock_conn.commit.assert_called_once()
        mock_pool.putconn.assert_called_once_with(mock_conn)

    @patch.dict('os.environ', {'DATABASE_URL': 'postgresql://localhost/db'})
    @patch('psycopg2.pool.SimpleConnectionPool')
    def test_get_db_connection_rolls_back_on_error(self, mock_pool_class):
        """Test that connection rolls back transaction on error"""
        from service import database
        import importlib
        importlib.reload(database)

        # Create mock connection that raises error
        mock_conn = MagicMock()
        mock_pool = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_pool_class.return_value = mock_pool
        database._connection_pool = None

        # Use connection with error
        with pytest.raises(ValueError):
            with database.get_db_connection() as conn:
                raise ValueError("Test error")

        # Verify rollback and putconn were called
        mock_conn.rollback.assert_called_once()
        mock_pool.putconn.assert_called_once_with(mock_conn)


class TestLogPrediction:
    """Tests for log_prediction function"""

    @patch.dict('os.environ', {}, clear=True)
    def test_log_prediction_returns_none_when_disabled(self):
        """Test that log_prediction returns None when database is disabled"""
        from service import database
        import importlib
        importlib.reload(database)

        result = database.log_prediction(
            input_sequence=[[1.0, 2.0]],
            predictions={'ann': 1.0, 'gru': 1.1, 'lstm': 1.2, 'transformer': 1.3, 'ensemble': 1.15},
            latencies={'ann': 10, 'gru': 12, 'lstm': 11, 'transformer': 15}
        )

        assert result is None

    @patch.dict('os.environ', {'DATABASE_URL': 'postgresql://localhost/db'})
    @patch('psycopg2.pool.SimpleConnectionPool')
    def test_log_prediction_inserts_data(self, mock_pool_class):
        """Test that log_prediction inserts data correctly"""
        from service import database
        import importlib
        importlib.reload(database)

        # Create mock cursor and connection
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = [42]  # Return prediction_id
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        # Create mock pool
        mock_pool = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_pool_class.return_value = mock_pool
        database._connection_pool = None

        # Log prediction
        input_seq = [[1.0, 2.0], [3.0, 4.0]]
        predictions = {
            'ann': 1.0,
            'gru': 1.1,
            'lstm': 1.2,
            'transformer': 1.3,
            'ensemble': 1.15
        }
        latencies = {'ann': 10, 'gru': 12, 'lstm': 11, 'transformer': 15}

        result = database.log_prediction(
            input_sequence=input_seq,
            predictions=predictions,
            latencies=latencies,
            request_id='test-123',
            user_id='user-1',
            model_version='v1.0'
        )

        # Verify result
        assert result == 42

        # Verify cursor.execute was called
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]

        # Check SQL contains INSERT
        assert 'INSERT INTO predictions' in call_args[0]

        # Check parameters
        params = call_args[1]
        assert params[0] == 'test-123'  # request_id
        assert params[1] == 'user-1'    # user_id
        assert params[3] == 2           # sequence_length
        assert params[4] == 1.0         # prediction_ann
        assert params[9] == 48          # total_latency (10+12+11+15)

    @patch.dict('os.environ', {'DATABASE_URL': 'postgresql://localhost/db'})
    @patch('service.database.get_db_connection')
    def test_log_prediction_handles_errors(self, mock_get_conn):
        """Test that log_prediction handles database errors gracefully"""
        from service import database
        import importlib
        importlib.reload(database)

        # Make connection raise error
        mock_get_conn.return_value.__enter__.side_effect = Exception("DB error")

        result = database.log_prediction(
            input_sequence=[[1.0, 2.0]],
            predictions={'ann': 1.0, 'gru': 1.1, 'lstm': 1.2, 'transformer': 1.3, 'ensemble': 1.15},
            latencies={'ann': 10, 'gru': 12, 'lstm': 11, 'transformer': 15}
        )

        # Should return None on error
        assert result is None


class TestUpdateActualValue:
    """Tests for update_actual_value function"""

    @patch.dict('os.environ', {}, clear=True)
    def test_update_actual_value_returns_false_when_disabled(self):
        """Test that update_actual_value returns False when database is disabled"""
        from service import database
        import importlib
        importlib.reload(database)

        result = database.update_actual_value(prediction_id=1, actual_value=5.0)
        assert result is False

    @patch.dict('os.environ', {'DATABASE_URL': 'postgresql://localhost/db'})
    @patch('psycopg2.pool.SimpleConnectionPool')
    def test_update_actual_value_updates_data(self, mock_pool_class):
        """Test that update_actual_value updates data correctly"""
        from service import database
        import importlib
        importlib.reload(database)

        # Create mock cursor and connection
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        # Create mock pool
        mock_pool = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_pool_class.return_value = mock_pool
        database._connection_pool = None

        # Update actual value
        result = database.update_actual_value(prediction_id=42, actual_value=5.5)

        # Verify result
        assert result is True

        # Verify cursor.execute was called
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]

        # Check SQL contains UPDATE
        assert 'UPDATE predictions' in call_args[0]
        assert 'actual_value = %s' in call_args[0]

        # Check parameters
        params = call_args[1]
        assert params[0] == 5.5   # actual_value
        assert params[1] == 42    # prediction_id

    @patch.dict('os.environ', {'DATABASE_URL': 'postgresql://localhost/db'})
    @patch('service.database.get_db_connection')
    def test_update_actual_value_handles_errors(self, mock_get_conn):
        """Test that update_actual_value handles database errors gracefully"""
        from service import database
        import importlib
        importlib.reload(database)

        # Make connection raise error
        mock_get_conn.return_value.__enter__.side_effect = Exception("DB error")

        result = database.update_actual_value(prediction_id=42, actual_value=5.5)

        # Should return False on error
        assert result is False


class TestGetRecentPredictions:
    """Tests for get_recent_predictions function"""

    @patch.dict('os.environ', {}, clear=True)
    def test_get_recent_predictions_returns_empty_when_disabled(self):
        """Test that get_recent_predictions returns empty list when database is disabled"""
        from service import database
        import importlib
        importlib.reload(database)

        result = database.get_recent_predictions(limit=10)
        assert result == []

    @patch.dict('os.environ', {'DATABASE_URL': 'postgresql://localhost/db'})
    @patch('psycopg2.pool.SimpleConnectionPool')
    def test_get_recent_predictions_returns_data(self, mock_pool_class):
        """Test that get_recent_predictions returns data correctly"""
        from service import database
        import importlib
        importlib.reload(database)

        # Create mock cursor with data
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ('id',), ('timestamp',), ('prediction_ensemble',),
            ('prediction_ann',), ('latency_ms',)
        ]
        mock_cursor.fetchall.return_value = [
            (1, '2024-01-01 10:00:00', 5.5, 5.3, 50),
            (2, '2024-01-01 11:00:00', 6.0, 5.8, 55),
        ]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        # Create mock pool
        mock_pool = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_pool_class.return_value = mock_pool
        database._connection_pool = None

        # Get recent predictions
        result = database.get_recent_predictions(limit=10)

        # Verify result
        assert len(result) == 2
        assert result[0]['id'] == 1
        assert result[0]['prediction_ensemble'] == 5.5
        assert result[1]['id'] == 2

        # Verify SQL was called with limit
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]
        assert 'SELECT' in call_args[0]
        assert 'LIMIT' in call_args[0]
        assert call_args[1] == (10,)

    @patch.dict('os.environ', {'DATABASE_URL': 'postgresql://localhost/db'})
    @patch('service.database.get_db_connection')
    def test_get_recent_predictions_handles_errors(self, mock_get_conn):
        """Test that get_recent_predictions handles database errors gracefully"""
        from service import database
        import importlib
        importlib.reload(database)

        # Make connection raise error
        mock_get_conn.return_value.__enter__.side_effect = Exception("DB error")

        result = database.get_recent_predictions(limit=10)

        # Should return empty list on error
        assert result == []


class TestGetModelPerformance:
    """Tests for get_model_performance function"""

    @patch.dict('os.environ', {}, clear=True)
    def test_get_model_performance_returns_none_when_disabled(self):
        """Test that get_model_performance returns None when database is disabled"""
        from service import database
        import importlib
        importlib.reload(database)

        result = database.get_model_performance()
        assert result is None

    @patch.dict('os.environ', {'DATABASE_URL': 'postgresql://localhost/db'})
    @patch('psycopg2.pool.SimpleConnectionPool')
    def test_get_model_performance_returns_metrics(self, mock_pool_class):
        """Test that get_model_performance returns metrics correctly"""
        from service import database
        import importlib
        importlib.reload(database)

        # Create mock cursor with performance data
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ('total_predictions',), ('avg_latency_ann',),
            ('mae_ann',), ('mae_ensemble',)
        ]
        mock_cursor.fetchone.return_value = (100, 12.5, 0.3, 0.25)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        # Create mock pool
        mock_pool = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_pool_class.return_value = mock_pool
        database._connection_pool = None

        # Get performance
        result = database.get_model_performance()

        # Verify result
        assert result is not None
        assert result['total_predictions'] == 100
        assert result['avg_latency_ann'] == 12.5
        assert result['mae_ann'] == 0.3
        assert result['mae_ensemble'] == 0.25

        # Verify SQL was called
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]
        assert 'SELECT' in call_args[0]
        assert 'AVG' in call_args[0]
        assert 'WHERE actual_value IS NOT NULL' in call_args[0]

    @patch.dict('os.environ', {'DATABASE_URL': 'postgresql://localhost/db'})
    @patch('service.database.get_db_connection')
    def test_get_model_performance_handles_no_data(self, mock_get_conn):
        """Test that get_model_performance handles no data gracefully"""
        from service import database
        import importlib
        importlib.reload(database)

        # Create mock cursor with no data
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        result = database.get_model_performance()

        # Should return None when no data
        assert result is None

    @patch.dict('os.environ', {'DATABASE_URL': 'postgresql://localhost/db'})
    @patch('service.database.get_db_connection')
    def test_get_model_performance_handles_errors(self, mock_get_conn):
        """Test that get_model_performance handles database errors gracefully"""
        from service import database
        import importlib
        importlib.reload(database)

        # Make connection raise error
        mock_get_conn.return_value.__enter__.side_effect = Exception("DB error")

        result = database.get_model_performance()

        # Should return None on error
        assert result is None
