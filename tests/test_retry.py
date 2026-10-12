from unittest.mock import MagicMock, patch

import pytest
from alpaca.common.exceptions import APIError
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout

from bot.broker import Broker
from bot.retry import RETRY_DELAYS, retry_transient_reads
from config import AssetClass


@pytest.fixture(autouse=True)
def no_sleep():
    """Never sleep for real; expose the mock so tests can count/assert calls."""
    with patch("bot.retry.time.sleep") as sleep:
        yield sleep


def api_error(status):
    """Build an APIError the way alpaca-py raises it: JSON body + HTTPError."""
    http_error = MagicMock()
    http_error.response.status_code = status
    return APIError(
        '{"code":50010000,"message":"internal server error occurred"}', http_error
    )


class TestRetryTransientReads:
    def test_returns_after_one_connection_error(self, no_sleep, caplog):
        fn = MagicMock(side_effect=[RequestsConnectionError("boom"), 42], __name__="read")
        wrapped = retry_transient_reads(fn)
        with caplog.at_level("WARNING"):
            assert wrapped() == 42
        assert fn.call_count == 2
        assert no_sleep.call_count == 1
        assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1

    def test_persistent_connection_error_reraises_original(self, no_sleep):
        original = RequestsConnectionError("still down")
        fn = MagicMock(side_effect=original, __name__="read")
        wrapped = retry_transient_reads(fn)
        with pytest.raises(RequestsConnectionError) as exc:
            wrapped()
        assert exc.value is original
        assert fn.call_count == len(RETRY_DELAYS) + 1
        assert no_sleep.call_count == len(RETRY_DELAYS)

    def test_timeout_is_retried(self, no_sleep):
        fn = MagicMock(side_effect=[Timeout("slow"), "ok"], __name__="read")
        wrapped = retry_transient_reads(fn)
        assert wrapped() == "ok"
        assert fn.call_count == 2
        assert no_sleep.call_count == 1

    def test_worst_case_delay_stays_small(self):
        assert sum(RETRY_DELAYS) <= 1.5

    def test_api_error_propagates_without_retry(self, no_sleep):
        fn = MagicMock(side_effect=APIError("400"))
        wrapped = retry_transient_reads(fn)
        with pytest.raises(APIError):
            wrapped()
        assert fn.call_count == 1
        assert no_sleep.call_count == 0

    def test_5xx_api_error_is_retried(self, no_sleep):
        fn = MagicMock(side_effect=[api_error(500), "ok"], __name__="read")
        wrapped = retry_transient_reads(fn)
        assert wrapped() == "ok"
        assert fn.call_count == 2
        assert no_sleep.call_count == 1

    def test_persistent_5xx_api_error_reraises_original(self, no_sleep):
        original = api_error(503)
        fn = MagicMock(side_effect=original, __name__="read")
        wrapped = retry_transient_reads(fn)
        with pytest.raises(APIError) as exc:
            wrapped()
        assert exc.value is original
        assert fn.call_count == len(RETRY_DELAYS) + 1
        assert no_sleep.call_count == len(RETRY_DELAYS)

    def test_4xx_api_error_propagates_without_retry(self, no_sleep):
        fn = MagicMock(side_effect=api_error(403), __name__="read")
        wrapped = retry_transient_reads(fn)
        with pytest.raises(APIError):
            wrapped()
        assert fn.call_count == 1
        assert no_sleep.call_count == 0

    def test_api_error_without_status_propagates_without_retry(self, no_sleep):
        fn = MagicMock(side_effect=APIError('{"code":1,"message":"x"}'), __name__="read")
        wrapped = retry_transient_reads(fn)
        with pytest.raises(APIError):
            wrapped()
        assert fn.call_count == 1
        assert no_sleep.call_count == 0

    def test_value_error_propagates_without_retry(self, no_sleep):
        fn = MagicMock(side_effect=ValueError("bad"))
        wrapped = retry_transient_reads(fn)
        with pytest.raises(ValueError):
            wrapped()
        assert fn.call_count == 1
        assert no_sleep.call_count == 0

    def test_args_and_kwargs_forwarded(self, no_sleep):
        fn = MagicMock(return_value="done")
        wrapped = retry_transient_reads(fn)
        assert wrapped("a", b=2) == "done"
        fn.assert_called_once_with("a", b=2)


@pytest.fixture
def broker():
    with patch("bot.broker.TradingClient") as mock_cls:
        b = Broker("key", "secret", paper=True)
        b.client = mock_cls.return_value
        yield b


class TestBrokerReadsRetry:
    def test_account_equity_retries_transient(self, broker, no_sleep):
        broker.client.get_account.side_effect = [
            RequestsConnectionError("aborted"),
            MagicMock(equity="10000"),
        ]
        assert broker.account_equity() == 10000.0
        assert broker.client.get_account.call_count == 2
        assert no_sleep.call_count == 1

    def test_market_is_open_retries_transient(self, broker, no_sleep):
        broker.client.get_clock.side_effect = [
            RequestsConnectionError("aborted"),
            MagicMock(is_open=True),
        ]
        assert broker.market_is_open() is True
        assert broker.client.get_clock.call_count == 2

    def test_has_open_order_retries_transient(self, broker, no_sleep):
        broker.client.get_orders.side_effect = [
            RequestsConnectionError("aborted"),
            [MagicMock()],
        ]
        assert broker.has_open_order("BTC/USD") is True
        assert broker.client.get_orders.call_count == 2


class TestBrokerWritesNotRetried:
    def test_market_order_does_not_retry_on_connection_error(self, broker, no_sleep):
        broker.client.submit_order.side_effect = RequestsConnectionError("aborted")
        with pytest.raises(RequestsConnectionError):
            broker.market_order("BTC/USD", "buy", 0.5, AssetClass.CRYPTO)
        assert broker.client.submit_order.call_count == 1
        assert no_sleep.call_count == 0

    def test_cancel_open_orders_does_not_retry_on_connection_error(self, broker, no_sleep):
        broker.client.get_orders.return_value = [MagicMock(id="oid-1")]
        broker.client.cancel_order_by_id.side_effect = RequestsConnectionError("aborted")
        with pytest.raises(RequestsConnectionError):
            broker.cancel_open_orders("BTC/USD")
        assert broker.client.cancel_order_by_id.call_count == 1
        assert no_sleep.call_count == 0
